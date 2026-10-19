from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from urllib.parse import quote

import streamlit as st

from src.app.client import AgentCallResult, AgentStreamEvent
from src.app.web.streamlit_sources import render_code, render_evidence
from src.app.web.streamlit_state import AssistantChatMessage, ChatMessage
from src.core.answer_schema import AnswerResponse, ContentUnit, SemanticCheck, iter_content_units
from src.core.answer_schema.rendering import inline_code_markdown
from src.core.contracts.outcome import TurnResult


_DEFAULT_ERROR_MESSAGE = "응답을 받지 못했습니다."
_DEFAULT_PROCESSING_MESSAGE = "응답을 준비하고 있습니다."
_STAGE_MESSAGES = {
    "summarize": "이전 맥락을 정리하고 있습니다.",
    "planner": "질문을 분석하고 있습니다.",
    "retrieval": "근거를 수집하고 있습니다.",
    "pre_synthesis_validation": "검색 결과를 확인하고 있습니다.",
    "synthesis": "답변을 작성하고 있습니다.",
    "post_synthesis_validation": "답변의 근거를 확인하고 있습니다.",
    "validation": "근거를 검증하고 있습니다.",
    "action_postprocess": "결과를 정리하고 있습니다.",
}


def render_chat_history(messages: list[ChatMessage], fastapi_url: str) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                for error_message in message.get("transport_errors", []):
                    st.error(error_message)
                result = message["result"]
                if result is not None:
                    render_turn_result(result, fastapi_url)
                elif message.get("request_id"):
                    st.caption(f"오류 식별자: {message['request_id']}")


def render_turn_result(result: TurnResult, fastapi_url: str) -> None:
    """Keep failures and questions separate from checked answer content."""
    if result.status == "needs_input":
        st.info(result.message)
    elif result.problem is not None:
        render = st.error if result.status == "failed" else st.warning
        render(result.problem.message)
        guidance = {
            "none": "",
            "retry_later": "잠시 후 같은 요청으로 다시 시도할 수 있습니다.",
            "fix_configuration": "서비스 설정을 수정해야 합니다. 오류 식별자를 운영자에게 알려 주세요.",
            "supply_information": "확인에 필요한 자료나 조건을 추가해 주세요.",
        }[result.problem.next_action]
        if result.problem.retry_after_seconds is not None:
            guidance = f"{result.problem.retry_after_seconds:g}초 후 다시 시도할 수 있습니다."
        if guidance:
            st.caption(guidance)
    if result.request_id and result.status != "completed":
        label = "요청 식별자" if result.status == "needs_input" else "오류 식별자"
        st.caption(f"{label}: {result.request_id}")
    if result.response is not None:
        render_answer_response(result.response, fastapi_url, turn_incomplete=result.status == "partial")


def render_answer_response(response: AnswerResponse, fastapi_url: str, *, turn_incomplete: bool = False) -> None:
    response = AnswerResponse.model_validate(response.model_dump(mode="json"))
    citations = {citation.evidence.id: citation for citation in response.citations}
    checks = {check.unit_id: check for check in response.checks}
    for issue in response.issues:
        if issue.unit_id is None:
            st.warning(issue.message)

    for index, block in enumerate(response.content.blocks):
        prefix = f"b{index}."
        units = [(path, unit) for path, unit in iter_content_units(response.content) if path.startswith(prefix)]
        if block.type == "paragraph":
            st.markdown(" ".join(_unit_text(unit, citations) for unit in block.content))
        elif block.type == "heading":
            st.markdown("#" * block.level + " " + _unit_text(block.content, citations))
        elif block.type == "list":
            st.markdown("\n".join(
                f"{item_index + 1}. {_unit_text(unit, citations)}" if block.ordered
                else f"- {_unit_text(unit, citations)}"
                for item_index, unit in enumerate(block.items)
            ))
        elif block.type == "code":
            if block.inline:
                st.markdown(inline_code_markdown(block.content.text))
            else:
                render_code(block.content.text, language=block.language or "text")
            _render_basis(block.content)
        elif block.type == "table":
            header = "| " + " | ".join(_table_cell(unit, citations) for unit in block.columns) + " |"
            separator = "| " + " | ".join("---" for _ in block.columns) + " |"
            rows = ["| " + " | ".join(_table_cell(unit, citations) for unit in row) + " |" for row in block.rows]
            st.markdown("\n".join([header, separator, *rows]))

        paths = {path for path, _ in units}
        for issue in response.issues:
            if issue.unit_id in paths:
                st.warning(issue.message)
        for path, unit in units:
            check = checks.get(path)
            if check is not None and check.reference_status == "missing":
                st.warning("이 내용에 연결된 근거를 찾지 못했습니다.")
            elif check is not None and check.support_status == "unsupported":
                st.warning("이 인용문은 보관한 원문과 일치하지 않습니다.")
        _render_citations([unit for _, unit in units], citations)

    _render_validation_scope(response, turn_incomplete=turn_incomplete)

    for action in response.actions:
        label = "파일 저장" if action.kind == "save_text" else "Slack 전송"
        if action.kind == "slack_notify" and action.slack is not None:
            delivery = action.slack
            if delivery.selection is not None:
                recipient = delivery.selection.selector.value
                prefix = "기본 수신자" if delivery.selection.source == "configured_default" else "요청 수신자"
                st.caption(f"{prefix}: {recipient}")
            elif delivery.intent.state == "unresolved":
                st.caption(f"요청 수신자: {delivery.intent.raw_input}")
            if delivery.status == "sent":
                st.success(f"Slack 전송 완료: {delivery.target.channel_id}")
            else:
                failure = delivery.failure
                render = st.warning if delivery.status == "unknown" else st.info if action.status == "skipped" else st.error
                state = "전달 여부 불명" if delivery.status == "unknown" else "전송하지 않음"
                render(f"Slack {state}: {failure.message}")
                guidance = {
                    "correct_input": "보낼 대상 하나의 ID 또는 이메일을 확인해 주세요.",
                    "fix_configuration": "앱 권한과 설정을 확인한 뒤 같은 수신자로 재시도해 주세요.",
                    "retry_same_target": "같은 수신자로 재시도할 수 있습니다.",
                    "verify_delivery": "Slack에서 전달 여부를 확인해 주세요. 자동으로 다시 전송하지 않습니다.",
                }[failure.next_action]
                if failure.retry_after_seconds is not None:
                    guidance = f"{failure.retry_after_seconds}초 이후 " + guidance
                st.caption(guidance)
            continue
        if action.status == "error":
            st.error(f"{label} 실패: {action.error or action.message or '처리하지 못했습니다.'}")
        elif action.status == "skipped":
            st.info(f"{label} 보류: {action.message or '실행 조건을 확인해 주세요.'}")
        elif action.status == "unknown":
            st.warning(f"{label} 결과를 확인할 수 없습니다: {action.error or action.message or '완료 여부를 다시 확인해 주세요.'}")
        elif action.kind == "save_text" and (action.verification != "verified" or action.artifact is None):
            st.warning("저장 결과를 검증할 수 없습니다. 검증된 파일 정보가 없는 이전 기록입니다.")
        else:
            detail = action.message or ""
            st.success(f"{label} 완료" + (f": {detail}" if detail else ""))
            if action.kind == "save_text" and action.artifact is not None:
                expires = datetime.fromtimestamp(action.artifact.expires_at, tz=timezone.utc).astimezone()
                st.caption(f"다운로드 가능 기한: {expires.strftime('%Y-%m-%d %H:%M %Z')}")
                if expires <= datetime.now(tz=timezone.utc):
                    st.warning("저장 파일의 보관 기한이 지났습니다.")
                    continue
                filename = action.artifact.filename
                url = f"{fastapi_url}/download/{quote(filename, safe='')}"
                st.markdown(f"[파일 다운로드 ({filename})]({url})")


def _render_validation_scope(response: AnswerResponse, *, turn_incomplete: bool = False) -> None:
    report = response.semantic_validation
    units = list(iter_content_units(response.content))
    if not response.citations and report is None and not units:
        return
    with st.expander("근거 확인 범위"):
        resolved = sum(check.reference_status == "resolved" for check in response.checks)
        exact = sum(check.support_status == "exact_match" for check in response.checks)
        st.caption(f"원문에 연결된 내용 {resolved}개 · 원문과 일치하는 발췌 {exact}개")
        if report is not None:
            st.caption("Decisions API에서 받은 평가 결과입니다.")
        semantic_checks = {(check.name, check.unit_id): check for check in report.checks} if report is not None else {}
        request_check = semantic_checks.get(("request_fulfilled", None))
        issue_codes = {issue.code for issue in response.issues}
        request_label = "요청 충족"
        if issue_codes & {"retrieved_evidence_incomplete", "evidence_packet_incomplete"}:
            request_label = "확인된 근거 범위의 본문 충족"
            st.caption("일부 근거가 부족해 요청 전체는 미완료입니다. 아래 판정은 확보된 근거로 작성한 본문 범위에 한정됩니다.")
        elif issue_codes & {"answer_incomplete", "source_excerpt_fallback"}:
            request_label = "제공된 본문 범위의 충족"
            st.caption("요청 전체를 완료하지 못한 부분 답변입니다. 아래 판정은 현재 제공된 본문에 대한 평가입니다.")
        elif turn_incomplete:
            request_label = "답변 본문 충족"
            st.caption("전체 요청 처리는 완료되지 않았습니다. 아래 판정은 답변 본문만 평가하며 저장·전송 등의 처리 완료 여부를 뜻하지 않습니다.")
        _render_semantic_result(request_label, request_check, report.threshold if report is not None else None)
        for index, (path, unit) in enumerate(units, start=1):
            preview = " ".join(unit.text.split())
            if len(preview) > 120:
                preview = preview[:117] + "…"
            st.markdown(f"**내용 {index}** · {preview}")
            _render_semantic_result(
                "근거 지지", semantic_checks.get(("grounded", path)),
                report.threshold if report is not None else None,
            )


def _render_semantic_result(label: str, check: SemanticCheck | None, threshold: float | None) -> None:
    if check is None or threshold is None:
        st.caption(f"{label}: 검증 안 됨")
        return
    if check.status == "refused":
        st.caption(f"{label}: 평가 거절 · 충족 확률 없음")
        return
    status = "통과" if check.probability >= threshold else "미통과"
    st.caption(f"{label}: {status} · 충족 확률 {check.probability} · 임곗값 {threshold}")


def _unit_text(unit: ContentUnit, citations: dict) -> str:
    prefix = {"inference": "**해석** · ", "example": "**예시** · ", "excerpt": "**원문 발췌** · "}.get(unit.basis, "")
    labels = " ".join(f"[{citations[ref].number}]" for ref in unit.refs if ref in citations)
    return prefix + unit.text + (f" {labels}" if labels else "")


def _table_cell(unit: ContentUnit, citations: dict) -> str:
    return _unit_text(unit, citations).replace("|", "\\|").replace("\n", "<br>")


def _render_basis(unit: ContentUnit) -> None:
    if unit.basis == "excerpt":
        st.caption("원문 발췌")
        return
    label = {"inference": "해석에 따라 작성한 코드", "example": "코드 예시", "source": "근거를 바탕으로 작성한 코드"}.get(unit.basis, "코드")
    st.caption(label + " · 실행 확인 안 됨")


def _render_citations(units: list[ContentUnit], citations: dict) -> None:
    refs = list(dict.fromkeys(ref for unit in units for ref in unit.refs if ref in citations))
    if not refs:
        return
    with st.container(horizontal=True):
        for ref in refs:
            citation = citations[ref]
            with st.popover(f"근거 [{citation.number}]", help=citation.evidence.snapshot.title):
                render_evidence(citation.evidence)


def process_chat_prompt(
    prompt: str,
    append_user_message: Callable[[ChatMessage], None],
    append_assistant_message: Callable[[ChatMessage], None],
    stream_agent: Callable[[str], Iterable[AgentStreamEvent]],
) -> None:
    append_user_message({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    result: AgentCallResult | None = None
    error_messages: list[str] = []
    request_id = ""
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.markdown("요청을 접수했습니다.")
        for event in stream_agent(prompt):
            request_id = event.observation.request_id or str(event.data.get("request_id") or request_id)
            if event.event == "final_response" and event.result is not None:
                result = event.result
            elif event.event == "error":
                error_message = str(event.data.get("message") or "응답 처리 중 오류가 발생했습니다.")
                if error_message not in error_messages:
                    error_messages.append(error_message)
                status_placeholder.markdown(error_message)
            else:
                progress = _progress_message_for_event(event)
                if progress:
                    status_placeholder.markdown(progress)
        status_placeholder.empty()

    if result is None:
        error_text = "\n\n".join(error_messages) or _DEFAULT_ERROR_MESSAGE
        append_assistant_message({
            "role": "assistant",
            "result": None,
            "transport_errors": [error_text],
            "request_id": request_id,
        })
    else:
        assistant_message: AssistantChatMessage = {
            "role": "assistant",
            "result": TurnResult.model_validate(result.model_dump(exclude={"upload_manifest"})),
        }
        if error_messages:
            assistant_message["transport_errors"] = error_messages
        append_assistant_message(assistant_message)
    st.rerun()


def _progress_message_for_event(event: AgentStreamEvent) -> str:
    if event.event == "request_started":
        return "요청을 접수했습니다."
    if event.event == "progress_snapshot":
        return str(event.data.get("summary") or "").strip() or _DEFAULT_PROCESSING_MESSAGE
    if event.event not in {"stage_started", "heartbeat"}:
        return ""
    return _STAGE_MESSAGES.get(str(event.data.get("stage") or "").strip(), _DEFAULT_PROCESSING_MESSAGE)
