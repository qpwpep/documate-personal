"""Retrieval recovery keeps evidence ownership and retries only recoverable searches."""

import pytest

from src.core.contracts import RetryState
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from tests.core.helpers import build_test_state


def _verdict(hit, status):
    from src.core.contracts.graph_state import EvidenceVerdict

    return EvidenceVerdict(
        requirement_id=hit.requirement_id, evidence_id=hit.evidence.id, status=status,
        probability={"passed": 0.99, "failed": 0.1, "refused": None}[status],
    )


def _mixed_evidence_state(*, rejected_status="refused"):
    from tests.core.test_decisions_validation import _evidence, _state

    state = _state()
    accepted = state["retrieval"].hit_log[0]
    rejected = accepted.model_copy(update={"evidence": _evidence(
        "Path.exists() creates missing directories without checking whether they exist."
    ), "rank": 2})
    state["retrieval"] = state["retrieval"].model_copy(update={
        "hit_log": [accepted, rejected],
        "evidence_verdicts": [_verdict(accepted, "passed"), _verdict(rejected, rejected_status)],
    })
    return state, accepted, rejected


def test_synthesis_and_validation_filter_verdicts_per_requirement_without_erasing_history():
    """같은 원문도 요구별 판정을 적용하고, 검증 전 새 후보와 통과 근거를 유지한다."""
    from src.runtime.nodes.synthesis.context import build_synthesis_context
    from src.runtime.nodes.validation.snapshot import collect_validation_snapshot
    from tests.core.test_decisions_validation import _evidence

    state, accepted, rejected = _mixed_evidence_state()
    second_task = state["planner"].output.tasks[0].model_copy(update={"requirement_id": "path_exists_other"})
    shared_but_failed = accepted.model_copy(update={"requirement_id": second_task.requirement_id})
    unjudged = accepted.model_copy(update={"evidence": _evidence(
        "Path.exists() returns False when the path does not exist."
    ), "rank": 3})
    state["planner"].output.tasks.append(second_task)
    raw_hits = [accepted, rejected, shared_but_failed, unjudged]
    state["retrieval"] = state["retrieval"].model_copy(update={
        "hit_log": raw_hits,
        "evidence_verdicts": [*state["retrieval"].evidence_verdicts, _verdict(shared_but_failed, "failed")],
    })

    context = build_synthesis_context(state=state)
    snapshot, errors = collect_validation_snapshot(state)

    expected = {(accepted.requirement_id, accepted.evidence.id), (unjudged.requirement_id, unjudged.evidence.id)}
    assert {(hit.requirement_id, hit.evidence.id) for hit in context.hits} == expected
    assert {(hit.requirement_id, hit.evidence.id) for hit in snapshot.parsed_hits} == expected
    assert errors == []
    assert state["retrieval"].hit_log == raw_hits


@pytest.mark.parametrize("rejected_status", ["failed", "refused"])
def test_dispatch_refresh_passes_only_accepted_previous_hits_and_retains_them_with_new_candidates(rejected_status):
    """한 요구를 재검색할 때 통과한 원문을 유지하고 실패 원문만 도구 입력에서 제외한다."""
    from src.core.contracts import RetrievalDiagnostic
    from src.core.contracts.retrieval import RetrievalResult
    from tests.core.test_decisions_validation import _evidence

    state, accepted, rejected = _mixed_evidence_state(rejected_status=rejected_status)
    candidate = accepted.model_copy(update={"evidence": _evidence(
        "Path.exists() reports whether a file or directory is present and never creates it."
    ), "rank": 3})
    calls = []

    def docs_search(**kwargs):
        calls.append(kwargs)
        return RetrievalResult(hits=[candidate], diagnostics=RetrievalDiagnostic(
            route="docs", status="success", answerability="covered",
        ))

    state["retry"] = RetryState(
        attempt=1, retry_reason="no_evidence", failed_requirement_ids=[accepted.requirement_id],
        hit_start_index=len(state["retrieval"].hit_log),
    )
    updates = make_retrieve_dispatch_node(docs_search, None, False)(state)

    assert len(calls) == 1
    assert calls[0]["refresh"] is True
    assert calls[0]["previous_hits"] == [accepted]
    assert updates["retrieval"].hit_log[:2] == [accepted, rejected]
    assert {hit.evidence.id for hit in updates["retrieval"].hit_log[2:]} == {accepted.evidence.id, candidate.evidence.id}
    assert updates["retrieval"].evidence_verdicts == state["retrieval"].evidence_verdicts


@pytest.mark.parametrize("rejected_status", ["failed", "refused"])
def test_refresh_returning_the_same_rejected_source_does_not_restore_it_to_synthesis(rejected_status):
    """재검색이 실패 원문을 다시 반환해도 새 합성과 검증에는 통과·새 원문만 들어간다."""
    from src.core.contracts import RetrievalDiagnostic
    from src.core.contracts.retrieval import RetrievalResult
    from src.runtime.nodes.synthesis.context import build_synthesis_context
    from src.runtime.nodes.validation.snapshot import collect_validation_snapshot
    from tests.core.test_decisions_validation import _evidence

    state, accepted, rejected = _mixed_evidence_state(rejected_status=rejected_status)
    candidate = accepted.model_copy(update={"evidence": _evidence(
        "Path.exists() tests the existing path without creating files or directories."
    ), "rank": 3})

    def docs_search(**kwargs):
        return RetrievalResult(hits=[rejected, candidate], diagnostics=RetrievalDiagnostic(
            route="docs", status="success", answerability="covered",
        ))

    state["retry"] = RetryState(
        attempt=1, retry_reason="no_evidence", failed_requirement_ids=[accepted.requirement_id],
        hit_start_index=len(state["retrieval"].hit_log),
    )
    state.update(make_retrieve_dispatch_node(docs_search, None, False)(state))

    context = build_synthesis_context(state=state)
    snapshot, errors = collect_validation_snapshot(state)
    expected = {accepted.evidence.id, candidate.evidence.id}
    assert {hit.evidence.id for hit in context.hits} == expected
    assert {hit.evidence.id for hit in snapshot.parsed_hits} == expected
    assert errors == []
    assert state["retrieval"].hit_log[:2] == [accepted, rejected]


def test_duplicate_request_reuse_does_not_restore_refused_evidence():
    """같은 검색 재사용도 거절 원문을 되살리지 않고 통과 원문만 새 시도에 전달한다."""
    from src.runtime.nodes.retrieval.executor import retrieval_fingerprint

    state, accepted, rejected = _mixed_evidence_state()
    task = state["planner"].output.tasks[0]
    state["debug"].retrieval_diagnostics[0] = state["debug"].retrieval_diagnostics[0].model_copy(update={
        "request_fingerprint": retrieval_fingerprint(task), "evidence_count": 2,
    })

    def unexpected_search(**kwargs):
        pytest.fail("An unchanged, completed request must reuse accepted evidence")

    updates = make_retrieve_dispatch_node(unexpected_search, None, False)(state)

    assert updates["retrieval"].hit_log[:2] == [accepted, rejected]
    assert updates["retrieval"].hit_log[2:] == [accepted]
    diagnostic = updates["debug"].retrieval_diagnostics[-1]
    assert diagnostic.reused is True
    assert diagnostic.evidence_count == 1


@pytest.mark.parametrize("retry_reason", [None, "tool_error", "no_evidence"])
def test_dispatch_retries_timeout_after_partial_success_without_repeating_completed_http_query(monkeypatch, retry_reason):
    """A successful partial result is not a completed search when its next provider request failed."""
    import json
    import requests

    from src.infra.settings import AppSettings
    from src.infra.tools.docs_search.tool import build_docs_search_tool

    queries = []
    source_url = "https://numpy.org/doc/2.4/reference/generated/numpy.reshape.html"

    def response(url, data):
        value = requests.Response()
        value.status_code = 200
        value.url = url
        value._content = json.dumps(data).encode()
        value._content_consumed = True
        return value

    def post(url, **kwargs):
        queries.append(kwargs["json"]["query"])
        if len(queries) == 2:
            raise requests.Timeout("provider timeout")
        aspect = "order" if len(queries) == 1 else "copy"
        return response(url, {"results": [{
            "url": source_url, "title": "numpy.reshape", "score": 0.9,
            "raw_content": f"# numpy.reshape\nnumpy.reshape(a, shape, {aspect}=None)\n{aspect}: Controls array layout.\n",
        }]})

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "head", lambda url, **kwargs: response(url, {}))
    tool = build_docs_search_tool(AppSettings(_env_file=None, tavily_api_key="test", openai_api_key="test"))
    dispatch = make_retrieve_dispatch_node(tool, None, False)
    task = RetrievalTask(route="docs", query="numpy.reshape order", k=4,
                         requirement={"library": "numpy", "symbols": ["numpy.reshape"], "aspects": ["order", "copy"]})
    state = build_test_state({"request_contract": RequestContract(),
                              "planner_output": PlannerOutput(use_retrieval=True, tasks=[task])})
    state.update(dispatch(state))
    first_hit = state["retrieval"].hit_log[0]
    first = state["debug"].retrieval_diagnostics[-1]
    assert (first.status, first.answerability, first.error_code) == ("success", "partial", "RETRIEVAL_DOCS_TIMEOUT")
    state["retry"] = RetryState(attempt=1, retry_reason=retry_reason, failed_requirement_ids=[task.requirement_id])
    state.update(dispatch(state))
    final = state["debug"].retrieval_diagnostics[-1]
    assert (final.reused, final.answerability, final.error_code) == (False, "covered", None)
    assert len(queries) == 3
    assert queries[0] == task.query
    assert queries[1] == queries[2]
    assert queries[1] == task.query + " copy"
    assert first_hit.evidence in [hit.evidence for hit in state["retrieval"].hit_log[1:]]
