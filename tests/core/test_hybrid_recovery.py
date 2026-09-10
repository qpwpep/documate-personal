from src.core.answer_schema import AnswerDocument, finalize_answer, iter_content_units
from src.core.contracts import RetrievalDiagnostic
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.runtime.nodes.retry import build_retry_update
from src.core.contracts.debug import RetryState
from src.runtime.nodes.validation.evidence_validator import assess_retrieval_quality, assess_validation, build_validation_snapshot
from src.runtime.nodes.validation.policy import apply_validation_outcome
from tests.core.test_answer_validation import _evidence, _hit


def _snapshot(document, docs, upload):
    contract = RequestContract()
    return build_validation_snapshot(
        user_input="공식 자료와 업로드 자료의 차이를 설명해 주세요.",
        planner_output=PlannerOutput(use_retrieval=True, tasks=[
            RetrievalTask(route="docs", query="official", k=3), RetrievalTask(route="upload", query="upload", k=3),
        ]),
        parsed_hits=[_hit(docs), _hit(upload)],
        current_attempt_retrieval_errors=[],
        current_attempt_retrieval_diagnostics=[],
        response_result=finalize_answer(document, [docs, upload], retrieval_required=True),
        evidence_packet=[docs, upload],
        request_contract=contract,
        response_request_id=contract.request_id,
        response_contract_revision=contract.revision,
    )


def test_hybrid_reference_coverage_accepts_a_paragraph_without_fixed_sections():
    """두 출처를 인용한 자연스러운 문단에 고정된 section 레이아웃을 강제하지 않는다."""
    docs, upload = _evidence("default = 3\n", "official"), _evidence("retries = 5\n")
    document = AnswerDocument.model_validate({"blocks": [{
        "type": "paragraph", "content": [
            {"text": "공식 기본값은 3입니다.", "basis": "source", "refs": [docs.id]},
            {"text": "업로드 설정은 5이므로 기본값과 다릅니다.", "basis": "inference", "refs": [docs.id, upload.id]},
        ],
    }]})

    assessment = assess_validation(_snapshot(document, docs, upload))

    assert assessment.retry_reason is None
    assert assessment.missing_route_coverage == []
    assert all(check.support_status == "not_evaluated" for check in assessment.checked_result.checks)


def test_missing_hybrid_route_falls_back_to_source_excerpts_without_inventing_a_comparison():
    """비교 근거가 답변에서 빠졌으면 비교 문장을 꾸미지 않고 원문 발췌와 제한을 제공한다."""
    docs, upload = _evidence("default = 3\n", "official"), _evidence("retries = 5\n")
    document = AnswerDocument.model_validate({"blocks": [{
        "type": "paragraph", "content": [{"text": "공식 설명만 사용했습니다.", "basis": "source", "refs": [docs.id]}],
    }]})
    snapshot = _snapshot(document, docs, upload)
    assessment = assess_validation(snapshot)

    updates = apply_validation_outcome(snapshot=snapshot, assessment=assessment, attempt=1, needs_retry=False)

    assert assessment.missing_route_coverage == ["upload"]
    result = updates["response"].result
    assert [unit.text for _, unit in iter_content_units(result.content) if unit.basis == "excerpt"] == [
        docs.excerpt, upload.excerpt,
    ]
    assert any(issue.code == "answer_incomplete" for issue in result.issues)
    assert all(unit.basis != "inference" for _, unit in iter_content_units(result.content))


def test_retrieval_retry_preserves_successful_upload_hits():
    """공식 검색만 실패했을 때 성공한 업로드 근거를 다음 시도에 보존한다."""
    docs, upload = _evidence("default = 3\n", "official"), _evidence("retries = 5\n")
    snapshot = _snapshot(AnswerDocument(), docs, upload)
    snapshot.parsed_hits = [_hit(upload)]
    snapshot.evidence_by_route = {"docs": [], "upload": [upload]}
    snapshot.current_attempt_retrieval_diagnostics = [
        RetrievalDiagnostic(route="upload", tool="upload_search", status="success"),
    ]
    assessment = assess_retrieval_quality(snapshot)

    needs_retry, retry, _ = build_retry_update(
        retry_context=RetryState(), retry_reason=assessment.retry_reason,
        planner_output=snapshot.planner_output, retrieval_errors=[], score_avg=None,
        failed_routes=assessment.failed_routes, current_attempt_hits=snapshot.parsed_hits,
        current_attempt_retrieval_diagnostics=snapshot.current_attempt_retrieval_diagnostics,
    )

    assert needs_retry
    assert retry.failed_routes == ["docs"]
    assert retry.preserved_hits == [_hit(upload).model_dump(mode="json")]
    assert [item.route for item in retry.preserved_retrieval_diagnostics] == ["upload"]


def test_explicit_zero_retry_budget_is_preserved():
    """0회로 설정된 검색 재시도 예산을 기본값으로 대체하지 않는다."""
    planner = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="docs", query="docs", k=3)])

    needs_retry, retry, _ = build_retry_update(
        retry_context=RetryState(max_retries=0), retry_reason="no_evidence",
        planner_output=planner, retrieval_errors=[], score_avg=None,
    )

    assert not needs_retry
    assert retry.max_retries == 0
    assert retry.attempt == 0


def test_displayed_content_repair_reuses_each_route_once():
    """본문 오류는 공식·업로드·혼합 출처 모두에서 같은 검색 결과로 한 번 재합성한다."""
    for routes, reason in (
        (["docs"], "unresolved_references"),
        (["upload"], "missing_content"),
        (["docs", "upload"], "missing_route_coverage"),
    ):
        hits = [_hit(_evidence(source_type="official" if route == "docs" else "upload")) for route in routes]
        planner = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route=route, query=route, k=3) for route in routes])
        arguments = {
            "retry_reason": reason, "planner_output": planner, "retrieval_errors": [],
            "score_avg": None, "current_attempt_hits": hits,
        }

        needs_retry, retry, _ = build_retry_update(retry_context=RetryState(), **arguments)

        assert needs_retry, routes
        assert retry.retry_scope == "reuse_hits_resynthesize"
        assert retry.failed_routes == []
        assert retry.preserved_hits == [hit.model_dump(mode="json") for hit in hits]
        assert retry.attempt == 1
        needs_retry_again, exhausted, _ = build_retry_update(retry_context=retry, **arguments)
        assert not needs_retry_again
        assert exhausted.attempt == 1


def test_upload_with_no_evidence_does_not_repeat_retrieval():
    """업로드 검색 결과가 없을 때 불필요하게 같은 검색을 반복하지 않는다."""
    planner = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="upload", query="upload", k=3)])

    needs_retry, retry, _ = build_retry_update(
        retry_context=RetryState(), retry_reason="no_evidence",
        planner_output=planner, retrieval_errors=[], score_avg=None, failed_routes={"upload"},
    )

    assert not needs_retry
    assert retry.attempt == 0
