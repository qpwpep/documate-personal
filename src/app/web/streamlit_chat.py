from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from urllib.parse import quote

import streamlit as st

from src.app.client import AgentStreamEvent
from src.app.web.streamlit_sources import render_code, render_evidence
from src.app.web.streamlit_state import AssistantChatMessage, ChatMessage
from src.core.answer_schema import ActionReceipt, AnswerResponse, CodeBlock, ContentUnit, SemanticCheck, iter_content_units
from src.core.answer_schema.rendering import code_display_label, content_unit_markdown, inline_code_markdown, literal_markdown
from src.core.contracts.outcome import TurnResult
from src.core.llm_errors import ExecutionProblem


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


def render_chat_history(
    messages: list[ChatMessage], fastapi_url: str, *, on_save: Callable[[str], None] | None = None,
    on_retry: Callable[[str], None] | None = None, on_edit: Callable[[str], None] | None = None,
    actions_disabled: bool = False,
) -> None:
    for index, message in enumerate(messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                if message.get("interrupted"):
                    st.warning("응답 수신이 중단되었습니다. 서버 작업은 취소되지 않아 처리나 저장·전송이 계속될 수 있습니다.")
                for error_message in message.get("transport_errors", []):
                    st.error(error_message)
                result = message["result"]
                if result is not None:
                    export_result = message.get("export_result")
                    displayed = result
                    if export_result is not None and result.response is not None:
                        # Keep the original body and other actions; show only the latest save attempt.
                        response = result.response.model_copy(update={
                            "actions": [action for action in result.response.actions if action.kind != "save_text"]
                                       + export_result.response.actions,
                        })
                        changes = {"response": response}
                        if export_result.status == "completed" and result.problem is not None and result.problem.code == "save_failed":
                            changes.update(status="completed", problem=None, message="")
                        displayed = result.model_copy(update=changes)
                    render_turn_result(displayed, fastapi_url, citation_scope=f"answer-{index}")
                    if result.answer_id and on_save is not None:
                        export_error = message.get("export_error")
                        if export_error:
                            st.error(export_error)
                        if export_result is not None and export_result.status != "completed":
                            st.caption("‘저장 다시 시도’는 이 답변의 본문과 출처를 그대로 사용합니다.")
                        retry = bool(export_error) or (export_result is not None and export_result.status != "completed")
                        if export_result is None and result.response is not None:
                            retry = retry or any(action.kind == "save_text" and action.status != "success" for action in result.response.actions)
                        save_actions = [action for action in displayed.response.actions if action.kind == "save_text"] if displayed.response else []
                        if not any(_save_download_ready(action) for action in save_actions):
                            expired = any(action.artifact is not None and action.verification == "verified"
                                          and action.artifact.expires_at <= datetime.now(tz=timezone.utc).timestamp()
                                          for action in save_actions)
                            label = "TXT 파일 만들기"
                            if retry:
                                label = "저장 다시 시도"
                            elif expired:
                                label = "TXT 다시 만들기"
                            st.button(label, key=f"save_answer_{index}_{result.answer_id}",
                                      disabled=actions_disabled, on_click=on_save, args=(result.answer_id,))
                elif message.get("request_id"):
                    st.caption(f"오류 식별자: {message['request_id']}")
                if index == len(messages) - 1 and index > 0 and messages[index - 1]["role"] == "user":
                    _render_question_recovery(
                        message, messages[index - 1]["content"], index=index,
                        on_retry=on_retry, on_edit=on_edit, disabled=actions_disabled,
                    )


def _render_question_recovery(
    message: AssistantChatMessage, prompt: str, *, index: int,
    on_retry: Callable[[str], None] | None, on_edit: Callable[[str], None] | None,
    disabled: bool,
) -> None:
    result = message["result"]
    if result is not None:
        if result.status not in {"failed", "partial"}:
            return
        # Delivery receipts have their own recovery path. Replaying the whole
        # utterance could repeat a file save or a Slack delivery that already ran.
        if result.problem.code == "save_failed" or (result.response is not None and result.response.actions):
            return
    can_retry = (
        result is not None
        and result.problem.next_action in {"retry_later", "supply_information"}
        and result.problem.stage not in {"action_postprocess", "response_assembly"}
    )
    if result is None and on_edit is not None and not disabled:
        if message.get("interrupted"):
            st.caption("다른 작업은 ‘새 채팅’에서 시작할 수 있습니다. 이 대화의 질문 원문은 아래 ‘질문 수정’으로 가져올 수 있습니다.")
        st.caption("처리 결과를 확인하지 못했습니다. 저장·전송을 요청했다면 완료 여부를 확인한 뒤 질문을 수정해 주세요.")
    request_id = result.request_id if result is not None else message.get("request_id", "")
    recovery_key = f"{index}_{request_id}"
    with st.container(horizontal=True):
        if can_retry and on_retry is not None:
            st.button("같은 질문 다시 보내기", key=f"retry_question_{recovery_key}",
                      disabled=disabled, on_click=on_retry, args=(prompt,))
        if on_edit is not None:
            st.button("질문 수정", key=f"edit_question_{recovery_key}",
                      disabled=disabled, on_click=on_edit, args=(prompt,))


def render_turn_result(result: TurnResult, fastapi_url: str, *, citation_scope: str = "") -> None:
    """Keep failures and questions separate from checked answer content."""
    partial_body = (result.status == "partial" and result.problem.code != "save_failed"
                    and result.problem.stage != "action_postprocess")
    if result.status == "needs_input":
        st.info(result.message)
    elif partial_body:
        st.warning(_partial_answer_summary(result.response, result.problem))
    elif result.problem is not None:
        render = st.error if result.status == "failed" else st.warning
        render(result.problem.message)
        guidance = _problem_guidance(result.problem)
        if guidance:
            st.caption(guidance)
    if result.request_id and result.status != "completed" and not partial_body:
        label = "요청 식별자" if result.status == "needs_input" else "오류 식별자"
        st.caption(f"{label}: {result.request_id}")
    if result.response is not None:
        render_answer_response(
            result.response, fastapi_url, turn_incomplete=result.status == "partial",
            partial_problem=result.problem if partial_body else None,
            request_id=result.request_id if partial_body else "",
            citation_scope=citation_scope,
        )


def _partial_answer_summary(response: AnswerResponse, problem: ExecutionProblem) -> str:
    global_issues = [issue for issue in response.issues if issue.unit_id is None]
    issue_codes = {issue.code for issue in global_issues}
    provided_scope = next(
        (issue.message for issue in global_issues if issue.code == "source_excerpt_fallback"),
        "현재 제공할 수 있는 본문을 아래에 표시합니다.",
    )
    # General completion notices do not explain the missing condition. Keep
    # every original issue in the details, but prioritize concrete gaps here.
    reasons = list(dict.fromkeys(
        issue.message for issue in global_issues
        if issue.code not in {"answer_incomplete", "source_excerpt_fallback"}
    )) or [problem.message]
    displayed_reasons = "\n".join(f"- {message}" for message in reasons[:2])
    if len(reasons) > 2:
        displayed_reasons += f"\n- 나머지 {len(reasons) - 2}개 한계는 ‘답변의 한계와 처리 정보’에서 확인하세요."
    if "invalid_content_removed" in issue_codes:
        displayed_reasons += "\n\n제외된 항목의 상세 기록은 없습니다."
    guidance = _problem_guidance(problem)
    if problem.next_action == "supply_information":
        if "evidence_packet_incomplete" in issue_codes and "retrieved_evidence_incomplete" not in issue_codes:
            guidance = "질문을 항목별로 나누어 다시 요청해 주세요. 검색한 자료를 답변에 담는 과정에서 생긴 한계입니다."
        elif "source_location_unverified" in issue_codes:
            guidance = "출처에서 원문 위치를 확인하거나, 위치를 확인할 수 있는 원본 파일을 첨부해 주세요."
        elif "invalid_content_removed" in issue_codes:
            guidance = "필요한 항목이 빠졌다면 해당 항목을 지정해 다시 질문해 주세요."
        else:
            guidance = ("위 조건을 확인해 질문을 수정하세요. 필요한 내용이 기존 자료에 있으면 파일명·위치를 지정하고, "
                        "없는 경우에만 자료를 추가해 주세요.")
    return ("**부분 답변** · 요청 전체를 완성하지 못했습니다."
            f"\n\n**제공한 범위** · {provided_scope}"
            f"\n\n**확인하지 못한 내용**\n\n{displayed_reasons}"
            + (f"\n\n**다음 행동** · {guidance}" if guidance else ""))


def _problem_guidance(problem: ExecutionProblem) -> str:
    if problem.retry_after_seconds is not None:
        return f"{problem.retry_after_seconds:g}초 후 다시 시도할 수 있습니다."
    return {
        "none": "",
        "retry_later": "잠시 후 같은 요청으로 다시 시도할 수 있습니다.",
        "fix_configuration": "서비스 설정을 수정해야 합니다. 오류 식별자를 운영자에게 알려 주세요.",
        "supply_information": "확인에 필요한 자료나 조건을 추가해 주세요.",
    }[problem.next_action]


def render_answer_response(
    response: AnswerResponse, fastapi_url: str, *, turn_incomplete: bool = False,
    partial_problem: ExecutionProblem | None = None, request_id: str = "", citation_scope: str = "",
) -> None:
    response = AnswerResponse.model_validate(response.model_dump(mode="json"))
    citations = {citation.evidence.id: citation for citation in response.citations}
    checks = {check.unit_id: check for check in response.checks}
    global_issues = [issue for issue in response.issues if issue.unit_id is None]
    section_units: list[ContentUnit | CodeBlock] = []
    citation_scope = f"{citation_scope}-{response.content_hash}" if citation_scope else response.content_hash
    citation_prefix = f"dm-citation-{citation_scope}-0"

    for index, block in enumerate(response.content.blocks):
        if block.type == "heading" and section_units:
            _render_citations(section_units, citations, citation_prefix=citation_prefix)
            section_units = []
            citation_prefix = f"dm-citation-{citation_scope}-{index}"
        prefix = f"b{index}."
        units = [(path, unit) for path, unit in iter_content_units(response.content) if path.startswith(prefix)]
        if block.type == "code":
            section_units.append(block)
        else:
            section_units.extend(unit for _, unit in units)
        if block.type == "paragraph":
            st.markdown(" ".join(_unit_text(unit, citations, citation_prefix) for unit in block.content))
        elif block.type == "heading":
            st.markdown("#" * block.level + " " + _unit_text(block.content, citations, citation_prefix))
        elif block.type == "list":
            st.markdown("\n".join(
                f"{item_index + 1}. {_unit_text(unit, citations, citation_prefix)}" if block.ordered
                else f"- {_unit_text(unit, citations, citation_prefix)}"
                for item_index, unit in enumerate(block.items)
            ))
        elif block.type == "code":
            if block.inline:
                st.markdown(inline_code_markdown(block.content.text))
            else:
                render_code(block.content.text, language=block.language or "text")
            st.caption(code_display_label(block))
        elif block.type == "table":
            header = "| " + " | ".join(_table_cell(unit, citations, citation_prefix) for unit in block.columns) + " |"
            separator = "| " + " | ".join("---" for _ in block.columns) + " |"
            rows = ["| " + " | ".join(_table_cell(unit, citations, citation_prefix) for unit in row) + " |" for row in block.rows]
            st.markdown("\n".join([header, separator, *rows]))

        unit_by_path = dict(units)
        unit_issues = [issue for issue in response.issues if issue.unit_id in unit_by_path]
        if unit_issues:
            with st.expander(f"이 내용의 한계 ({len(unit_issues)}개)"):
                for issue in unit_issues:
                    preview = " ".join(unit_by_path[issue.unit_id].text.split())
                    st.caption("관련 내용: " + (preview[:117] + "…" if len(preview) > 120 else preview))
                    st.warning(f"답변에 기록된 한계: {issue.message}")
        for path, unit in units:
            check = checks.get(path)
            if check is not None and check.reference_status == "missing":
                st.warning("이 내용에 연결된 근거를 찾지 못했습니다.")
            elif check is not None and check.support_status == "unsupported":
                st.warning("이 인용문은 보관한 원문과 일치하지 않습니다.")
    _render_citations(section_units, citations, citation_prefix=citation_prefix)

    if global_issues or partial_problem is not None:
        with st.expander("답변의 한계와 처리 정보"):
            if partial_problem is not None:
                st.markdown(partial_problem.message)
            for issue in global_issues:
                st.warning(f"답변에 기록된 한계: {issue.message}")
            if request_id:
                st.caption(f"오류 식별자: {request_id}")
    _render_validation_scope(
        response, turn_incomplete=turn_incomplete, show_limit_summary=partial_problem is None,
    )
    render_action_receipts(response, fastapi_url)


def render_action_receipts(response: AnswerResponse, fastapi_url: str) -> None:
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
            st.error("파일 저장 실패: 기존 답변은 유지됩니다. 아래 '저장 다시 시도'로 같은 답변을 저장할 수 있습니다."
                     if action.kind == "save_text" else
                     f"{label} 실패: {action.error or action.message or '처리하지 못했습니다.'}")
        elif action.status == "skipped":
            st.info(f"{label} 보류: {action.message or '실행 조건을 확인해 주세요.'}")
        elif action.status == "unknown":
            st.warning(f"{label} 결과를 확인할 수 없습니다: {action.error or action.message or '완료 여부를 다시 확인해 주세요.'}")
        elif action.kind == "save_text" and (action.verification != "verified" or action.artifact is None):
            st.warning("저장 결과를 검증할 수 없습니다. 검증된 파일 정보가 없는 이전 기록입니다.")
        elif action.kind == "save_text" and action.artifact is not None:
            expires = datetime.fromtimestamp(action.artifact.expires_at, tz=timezone.utc).astimezone()
            st.caption(f"다운로드 가능 기한: {expires.strftime('%Y-%m-%d %H:%M %Z')}")
            if not _save_download_ready(action):
                st.warning("저장 파일의 보관 기한이 지났습니다. 답변은 유지됩니다. 내려받으려면 TXT를 다시 만들어 주세요.")
                continue
            st.success("파일 준비 완료 — 아래에서 내려받으세요.")
            url = f"{fastapi_url}/download/{quote(action.artifact.filename, safe='')}"
            st.link_button("TXT 다운로드", url, type="primary")
        else:
            detail = action.message or ""
            st.success(f"{label} 완료" + (f": {detail}" if detail else ""))


def _save_download_ready(action: ActionReceipt) -> bool:
    return (action.kind == "save_text" and action.status == "success"
            and action.verification == "verified" and action.artifact is not None
            and action.artifact.expires_at > datetime.now(tz=timezone.utc).timestamp())


def _render_validation_scope(
    response: AnswerResponse, *, turn_incomplete: bool = False, show_limit_summary: bool = True,
) -> None:
    report = response.semantic_validation
    units = list(iter_content_units(response.content))
    if not response.citations and report is None and not units:
        return
    resolved = sum(check.reference_status == "resolved" for check in response.checks)
    excerpt_paths = {path for path, unit in units if unit.basis == "excerpt"}
    exact = sum(check.unit_id in excerpt_paths and check.support_status == "exact_match" for check in response.checks)
    semantic_checks = {(check.name, check.unit_id): check for check in report.checks} if report is not None else {}
    request_check = semantic_checks.get(("request_fulfilled", None))
    issue_codes = {issue.code for issue in response.issues}
    quote_summary = f"직접 인용 {len(excerpt_paths)}개 중 원문 일치 {exact}개" if excerpt_paths else "직접 인용 없음"
    st.caption(f"출처가 연결된 내용 {resolved}개 · {quote_summary}")
    limits = list(dict.fromkeys(issue.message for issue in response.issues if issue.unit_id is None))
    if "invalid_content_removed" in issue_codes:
        limits.append("제외된 항목의 상세 기록은 없습니다.")
    unit_limit_count = sum(issue.unit_id is not None for issue in response.issues)
    if unit_limit_count:
        limits.append(f"내용별 한계 {unit_limit_count}개는 해당 내용 아래에서 확인할 수 있습니다.")
    if request_check is not None and request_check.status == "refused":
        limits.append("요청 조건을 자동으로 확인하지 못했습니다.")
    elif request_check is not None and request_check.probability < report.threshold:
        limits.append("답변 본문의 요청 조건 충족을 확인하지 못했습니다.")
    if limits and show_limit_summary:
        st.caption("확인할 한계: " + " ".join(limits))
    if "invalid_content_removed" in issue_codes and show_limit_summary:
        st.caption("필요한 항목이 빠졌다면 해당 항목을 지정해 다시 질문해 주세요.")
    elif "source_location_unverified" in issue_codes and show_limit_summary:
        st.caption("출처에서 원문 위치를 확인하거나, 위치를 확인할 수 있는 원본 파일을 첨부해 주세요.")

    with st.expander("검증 상세"):
        if resolved:
            st.caption(f"출처 연결: 내용 {resolved}개에 출처가 연결되어 있습니다. ‘근거’에서 사용한 원문을 확인할 수 있습니다.")
        else:
            st.caption("출처 연결: 이 답변에 연결된 출처가 없습니다.")
        st.caption(quote_summary + ". 직접 인용은 보관한 원문과 일치하는지 확인합니다.")
        if any(block.type == "code" for block in response.content.blocks):
            st.caption("실행 확인: 코드 형식으로 표시된 내용의 실행 결과는 확인하지 않았습니다.")
        if limits:
            st.caption("확인하지 못한 내용: " + " ".join(limits))
        st.caption("자동 확인은 답변 전체의 정확도나 실행 성공을 보증하지 않습니다.")
        st.caption("본문 확인과 저장·전송의 처리 결과는 별도로 표시됩니다.")
        if issue_codes & {"retrieved_evidence_incomplete", "evidence_packet_incomplete"}:
            st.caption("이 답변에 기록된 한계: 일부 근거가 부족합니다. 아래 판정은 확보된 근거로 작성한 본문 범위에 한정됩니다.")
        elif issue_codes & {"answer_incomplete", "source_excerpt_fallback"}:
            st.caption("이 답변에 기록된 한계: 일부 내용만 제공한 부분 답변입니다. 아래 판정은 제공된 본문 범위에 한정됩니다.")
        if turn_incomplete:
            st.caption("이번 요청 처리는 완료되지 않았습니다. 남은 작업은 안내와 저장·전송 결과를 확인해 주세요.")
        with st.expander("기술 상세"):
            if report is not None:
                st.caption("Decisions API에서 받은 평가 결과입니다. 수치는 개별 조건의 충족 여부에 대한 자동 평가입니다.")
            _render_semantic_result("답변 본문 조건 충족", request_check, report.threshold if report is not None else None)
            format_check = semantic_checks.get(("format_fulfilled", None))
            if format_check is not None:
                _render_semantic_result("답변 서식 조건 충족", format_check, report.threshold)
            for index, (path, unit) in enumerate(units, start=1):
                preview = " ".join(unit.text.split())
                if len(preview) > 120:
                    preview = preview[:117] + "…"
                if unit.source_location is not None:
                    preview = literal_markdown(preview)
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


def _unit_text(unit: ContentUnit, citations: dict, citation_prefix: str) -> str:
    prefix = {"inference": "**해석** · ", "example": "**예시** · ", "excerpt": "**원문 발췌** · "}.get(unit.basis, "")
    labels = " ".join(f"[[{citations[ref].number}]](#{citation_prefix}-{citations[ref].number})"
                      for ref in unit.refs if ref in citations)
    return prefix + content_unit_markdown(unit) + (f" {labels}" if labels else "")


def _table_cell(unit: ContentUnit, citations: dict, citation_prefix: str) -> str:
    return _unit_text(unit, citations, citation_prefix).replace("|", "\\|").replace("\n", "<br>")


def _render_citations(context: list[ContentUnit | CodeBlock], citations: dict, *, citation_prefix: str) -> None:
    units = [item.content if isinstance(item, CodeBlock) else item for item in context]
    refs = list(dict.fromkeys(ref for unit in units for ref in unit.refs if ref in citations))
    if not refs:
        return
    with st.container(horizontal=True):
        for ref in refs:
            citation = citations[ref]
            target = f"{citation_prefix}-{citation.number}"
            with st.container(key=target, width="content"):
                with st.popover(f"근거 [{citation.number}]", help=citation.evidence.snapshot.title):
                    render_evidence(
                        citation.evidence, cited_texts=[unit.text for unit in units if ref in unit.refs],
                        answer_context=[item if isinstance(item, CodeBlock) else content_unit_markdown(item)
                                        for item, unit in zip(context, units) if ref in unit.refs],
                        search_key=target,
                    )


def progress_message_for_event(event: AgentStreamEvent) -> str:
    if event.event == "request_started":
        return "요청을 접수했습니다."
    if event.event == "progress_snapshot":
        return str(event.data.get("summary") or "").strip() or _DEFAULT_PROCESSING_MESSAGE
    if event.event not in {"stage_started", "heartbeat"}:
        return ""
    return _STAGE_MESSAGES.get(str(event.data.get("stage") or "").strip(), _DEFAULT_PROCESSING_MESSAGE)
