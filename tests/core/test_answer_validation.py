from src.core.answer_schema import AnswerDocument, export_answer_text, finalize_answer, iter_content_units
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput
from src.runtime.nodes.validation.evidence_validator import assess_validation, build_validation_snapshot
from src.runtime.nodes.validation.policy import apply_validation_outcome


def _snapshot(document: AnswerDocument):
    return build_validation_snapshot(
        user_input="자료를 설명해 주세요",
        planner_output=PlannerOutput(use_retrieval=False, tasks=[]),
        parsed_hits=[],
        current_attempt_retrieval_errors=[],
        current_attempt_retrieval_diagnostics=[],
        response_result=finalize_answer(document, []),
        evidence_packet=[],
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
    """목록·코드·비교표 셀의 표시 본문도 문단과 같은 참조 검사를 받는다."""
    unit = {"text": "검사할 내용", "basis": "source", "refs": ["missing"]}
    label = {"text": "항목", "basis": "interaction", "refs": []}
    document = AnswerDocument.model_validate({"blocks": [
        {"type": "list", "items": [unit]},
        {"type": "code", "language": "python", "content": unit},
        {"type": "table", "columns": [label], "rows": [[unit]]},
    ]})

    assessment = assess_validation(_snapshot(document))

    assert assessment.invalid_unit_paths == {"b0.items.0", "b1.content", "b2.rows.0.0"}




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








