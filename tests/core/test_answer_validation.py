from src.core.answer_schema import AnswerDocument, export_answer_text, finalize_answer, iter_content_units, text_document
from src.core.contracts import ResponseState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import RequestContract
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.helpers import make_decisions_client
from src.runtime.nodes.validation.evidence_validator import assess_validation, build_validation_snapshot
from src.runtime.nodes.validation.policy import apply_validation_outcome


def _snapshot(document: AnswerDocument):
    contract = RequestContract(request_id="validation-test")
    return build_validation_snapshot(
        user_input="자료를 설명해 주세요",
        planner_output=PlannerOutput(use_retrieval=False, tasks=[]),
        parsed_hits=[],
        current_attempt_retrieval_errors=[],
        current_attempt_retrieval_diagnostics=[],
        response_result=finalize_answer(document, []),
        evidence_packet=[],
        request_contract=contract,
        response_request_id=contract.request_id,
        response_contract_revision=contract.revision,
    )


def _evidence(text="retries = 5\n", source_type="upload"):
    snapshot = build_snapshot(
        source_uri="upload:///job.py" if source_type == "upload" else "https://docs.example.org/retries",
        title="job.py" if source_type == "upload" else "Reference",
        media_type="text/x-python" if source_type == "upload" else "text/html",
        source_type=source_type,
        content=text,
        parser="test",
        parser_version="1",
    )
    element = DocumentElement(
        element_id="content", kind="code", text=text, language="python",
        anchors=[SourceAnchor(kind="code", line_start=1, line_end=1)],
    )
    return build_evidence(snapshot=snapshot, element=element)


def _hit(evidence):
    return SearchHit(evidence=evidence, score=RetrievalScore(metric="rank", raw=1, direction="lower"), rank=1)


def test_validation_covers_list_code_and_table_cell_text():
    """검증을 마친 답변은 근거 없는 목록·코드·표 셀을 제거하고 유효한 본문을 보존한다."""
    invalid_units = [
        {"text": text, "basis": "source", "refs": [f"missing-{index}"]}
        for index, text in enumerate(["잘못된 목록", "unsupported_code()", "잘못된 표 값"])
    ]
    retained = text_document("유지할 안내")
    label = {"text": "항목", "basis": "interaction", "refs": []}
    document = AnswerDocument.model_validate({"blocks": [
        *retained.model_dump(mode="json")["blocks"],
        {"type": "list", "items": [invalid_units[0]]},
        {"type": "code", "language": "python", "content": invalid_units[1]},
        {"type": "table", "columns": [label], "rows": [[invalid_units[2]]]},
    ]})
    contract = RequestContract()
    state = build_graph_state_input(
        user_input="자료를 설명해 주세요", request_contract=contract,
        response=ResponseState(result=finalize_answer(document, []),
                               request_id=contract.request_id, contract_revision=contract.revision),
        retry={"max_retries": 0, "max_content_repairs": 0},
    )

    updates = make_post_synthesis_validation_node(False, decisions_client=make_decisions_client())(state)

    result = updates["response"].result
    assert updates["response"].kind == "answer"
    assert not updates["retry"].needs_retry
    assert result.content == retained
    assert export_answer_text(result) == "유지할 안내"
    assert all(unit["text"] not in result.model_dump_json() for unit in invalid_units)
    assert any(issue.code == "invalid_content_removed" for issue in result.issues)


def test_mismatching_excerpt_falls_back_to_the_exact_original_text():
    """원문과 다른 발췌를 정상으로 처리하지 않고 실제 원문으로 대체한다."""
    evidence = _evidence()
    document = AnswerDocument.model_validate({"blocks": [{
        "type": "code", "language": "python",
        "content": {"text": "retries = 99", "basis": "excerpt", "refs": [evidence.id]},
    }]})
    snapshot = _snapshot(document)
    snapshot.evidence_packet = [evidence]
    snapshot.parsed_hits = [_hit(evidence)]

    assessment = assess_validation(snapshot)
    updates = apply_validation_outcome(snapshot=snapshot, assessment=assessment, attempt=1, needs_retry=False)

    assert assessment.checked_result.checks[0].support_status == "unsupported"
    assert assessment.retry_reason == "missing_content"
    result = updates["response"].result
    excerpts = [unit.text for _, unit in iter_content_units(result.content) if unit.basis == "excerpt"]
    assert excerpts == [evidence.excerpt]
    assert "retries = 99" not in export_answer_text(result)
    assert any(check.support_status == "exact_match" for check in result.checks)
