from src.core.answer_schema import AnswerDocument, export_answer_text, finalize_answer, iter_content_units
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import AnswerContract, ContentRequirement, ContractEvidence, RequestContract
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


def test_validation_checks_the_displayed_leaf_reference():
    """화면에 표시되는 문단의 존재하지 않는 참조를 탐지한다."""
    document = AnswerDocument.model_validate({"blocks": [{
        "type": "paragraph",
        "content": [{"text": "확인되지 않은 설명", "basis": "source", "refs": ["missing"]}],
    }]})

    assessment = assess_validation(_snapshot(document))

    assert assessment.retry_reason == "unresolved_references"
    assert len(assessment.invalid_unit_paths) == 1
    assert assessment.checked_result.checks[0].reference_status == "missing"


def test_interaction_is_not_misrepresented_as_semantically_verified():
    """근거가 필요 없는 안내 문구를 의미 검증 완료로 표시하지 않는다."""
    document = AnswerDocument.model_validate({"blocks": [{
        "type": "paragraph",
        "content": [{"text": "확인할 자료를 올려 주세요.", "basis": "interaction", "refs": []}],
    }]})

    assessment = assess_validation(_snapshot(document))

    assert assessment.retry_reason is None
    assert assessment.checked_result.checks[0].reference_status == "not_required"
    assert assessment.checked_result.checks[0].support_status == "not_evaluated"


def test_recovery_removes_invalid_text_instead_of_restoring_an_old_section():
    """삭제한 내용 단위를 이전 본문에서 복원하지 않는다."""
    document = AnswerDocument.model_validate({"blocks": [{
        "type": "paragraph",
        "content": [
            {"text": "유지할 안내", "basis": "interaction", "refs": []},
            {"text": "삭제해야 할 잘못된 본문", "basis": "source", "refs": ["missing"]},
        ],
    }]})
    snapshot = _snapshot(document)

    updates = apply_validation_outcome(
        snapshot=snapshot,
        assessment=assess_validation(snapshot),
        attempt=1,
        needs_retry=False,
    )

    result = updates["response"].result
    assert result.content.blocks[0].content[0].text == "유지할 안내"
    assert "삭제해야 할 잘못된 본문" not in result.content.model_dump_json()
    assert all(check.reference_status != "missing" for check in result.checks)

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


def test_reference_resolution_does_not_claim_semantic_verification():
    """참조가 해석되어도 원문 내용의 의미적 지지까지 검증됐다고 표시하지 않는다."""
    evidence = _evidence()
    document = AnswerDocument.model_validate({"blocks": [{
        "type": "paragraph",
        "content": [{"text": "재시도 횟수는 다섯 번입니다.", "basis": "source", "refs": [evidence.id]}],
    }]})
    snapshot = _snapshot(document)
    snapshot.evidence_packet = [evidence]

    assessment = assess_validation(snapshot)

    assert assessment.retry_reason is None
    assert assessment.checked_result.checks[0].reference_status == "resolved"
    assert assessment.checked_result.checks[0].support_status == "not_evaluated"


def test_hit_not_supplied_to_synthesis_cannot_resolve_a_displayed_reference():
    """검색 로그에만 존재한 자료가 합성 본문의 참조를 사후 정당화하지 않는다."""
    evidence = _evidence()
    document = AnswerDocument.model_validate({"blocks": [{
        "type": "paragraph",
        "content": [{"text": "자료에 적힌 내용", "basis": "source", "refs": [evidence.id]}],
    }]})
    snapshot = _snapshot(document)
    snapshot.parsed_hits = [_hit(evidence)]
    snapshot.response_result = finalize_answer(document, [evidence])

    assessment = assess_validation(snapshot)

    assert assessment.retry_reason == "unresolved_references"
    assert assessment.checked_result.citations == []
    assert assessment.checked_result.checks[0].reference_status == "missing"


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


def test_recovery_removes_an_invalid_table_row_without_breaking_table_shape():
    """잘못된 근거를 가진 행을 없애고 남은 비교표의 열 구조를 보존한다."""
    valid = {"text": "남길 내용", "basis": "interaction", "refs": []}
    invalid = {"text": "삭제할 셀", "basis": "source", "refs": ["missing"]}
    document = AnswerDocument.model_validate({"blocks": [{
        "type": "table", "columns": [valid, valid], "rows": [[valid, valid], [valid, invalid]],
    }]})
    snapshot = _snapshot(document)

    updates = apply_validation_outcome(
        snapshot=snapshot, assessment=assess_validation(snapshot), attempt=1, needs_retry=False,
    )

    assert len(updates["response"].result.content.blocks[0].rows) == 1
    assert "삭제할 셀" not in export_answer_text(updates["response"].result)


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


def test_recovery_does_not_silently_drop_requested_code():
    """잘못된 코드 블록을 제거하면 코드 요청을 충족한 것처럼 남은 문단만 반환하지 않는다."""
    evidence = _evidence()
    document = AnswerDocument.model_validate({"blocks": [
        {"type": "paragraph", "content": [{"text": "코드를 확인합니다.", "basis": "interaction", "refs": []}]},
        {"type": "code", "language": "python", "content": {"text": "bad()", "basis": "source", "refs": ["missing"]}},
    ]})
    snapshot = _snapshot(document)
    snapshot.user_input = "코드 예시를 보여줘"
    snapshot.request_contract = RequestContract(
        request_id="validation-test",
        answer=AnswerContract(content=(ContentRequirement(kind="code_example", mode="required", evidence_ids=("r1",)),)),
        evidence=(ContractEvidence(id="r1", turn_id="current", quote="코드 예시를 보여줘", scope="answer.content.code_example", interpretation="instruction"),),
    )
    snapshot.parsed_hits = [_hit(evidence)]
    snapshot.evidence_packet = [evidence]

    updates = apply_validation_outcome(
        snapshot=snapshot, assessment=assess_validation(snapshot), attempt=1, needs_retry=False,
    )

    assert any(issue.code == "answer_incomplete" for issue in updates["response"].result.issues)


def test_post_validation_records_reference_failure_as_validation_diagnostic():
    """본문 참조 실패를 검색 오류와 구분되는 검증 오류 코드로 기록한다."""
    from src.core.contracts import ResponseState, RuntimeState
    from src.runtime.nodes.validation import make_post_synthesis_validation_node

    document = AnswerDocument.model_validate({"blocks": [{
        "type": "paragraph", "content": [{"text": "bad", "basis": "source", "refs": ["missing"]}],
    }]})
    contract = RequestContract()
    state = {"runtime": RuntimeState(user_input="설명해줘", request_contract=contract),
             "response": ResponseState(result=finalize_answer(document, []), request_id=contract.request_id, contract_revision=contract.revision)}

    updates = make_post_synthesis_validation_node(False)(state)

    assert updates["retry"].retry_reason == "unresolved_references"
    assert "VALIDATION_UNRESOLVED_REFERENCES" in updates["debug"].error_codes


def test_pre_validation_preserves_planner_failure_without_claiming_upload_is_missing():
    """검색 계획 실패 후속 질문은 업로드 누락으로 잘못 진단하지 않는다."""
    from src.core.contracts import PlannerState, RuntimeState
    from src.core.contracts.debug import PlannerDiagnostic, RetryState
    from src.runtime.nodes.validation import make_pre_synthesis_validation_node

    followup = "검색 계획을 만들지 못했습니다. 다시 요청해 주세요."
    state = {
        "runtime": RuntimeState(user_input="검색한 뒤 저장해줘"),
        "planner": PlannerState(guided_followup=followup, diagnostics=PlannerDiagnostic(reason="planner_unavailable")),
        "retry": RetryState(attempt=1, needs_retry=True, retry_reason="no_evidence", failed_routes=["docs"]),
    }

    updates = make_pre_synthesis_validation_node(False)(state)

    assert export_answer_text(updates["response"].result) == followup
    assert updates["retry"].retry_reason is None
    assert not updates["retry"].needs_retry
    assert updates["retry"].failed_routes == []


def test_reusing_an_answer_preserves_its_original_reference_requirement():
    """이전 답변의 검색 근거 요구 조건을 현재 저장 요청이 축소하지 않는다."""
    evidence = _evidence()
    document = AnswerDocument.model_validate({"blocks": [{
        "type": "code", "language": "python",
        "content": {"text": "retries = 5", "basis": "example", "refs": [evidence.id]},
    }]})
    snapshot = _snapshot(document)
    snapshot.user_input = "방금 답변을 저장해줘"
    snapshot.evidence_packet = [evidence]
    snapshot.response_result = finalize_answer(document, [evidence], retrieval_required=True)

    assessment = assess_validation(snapshot)

    assert assessment.checked_result.retrieval_required
    assert assessment.checked_result.checks[0].reference_status == "resolved"
