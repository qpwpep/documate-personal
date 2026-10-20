"""Retrieval recovery keeps evidence ownership and retries only recoverable searches."""

import json

import httpx
import pytest
from openai import OpenAI

from src.core.contracts import RetryState
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.infra.decisions import DecisionsClient
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from tests.core.helpers import build_test_state
from tests.synthesis_fixtures import prepare_synthesis_state, synthesis_excerpt_limits


def test_search_recovery_stops_after_one_supplemental_attempt():
    from src.runtime.nodes.retry import build_retry_update

    task = RetrievalTask(route="docs", query="numpy.reshape", k=4,
                         requirement={"library": "numpy", "symbols": ["numpy.reshape"]})
    plan = PlannerOutput(use_retrieval=True, tasks=[task])
    needs_retry, retry, _ = build_retry_update(
        retry_context=RetryState(max_retries=5), retry_reason="no_evidence",
        planner_output=plan, score_avg=None, failed_requirement_ids=[task.requirement_id],
    )
    assert needs_retry and retry.attempt == 1

    needs_retry, retry, _ = build_retry_update(
        retry_context=retry, retry_reason="no_evidence",
        planner_output=plan, score_avg=None, failed_requirement_ids=[task.requirement_id],
    )
    assert not needs_retry
    assert retry.attempt == 1
    assert retry.terminal_failed_requirement_ids == [task.requirement_id]


def _verdict(hit, status):
    from src.core.contracts.graph_state import EvidenceVerdict

    return EvidenceVerdict(
        requirement_id=hit.requirement_id, evidence_id=hit.evidence.id, status=status,
        probability={"passed": 0.99, "failed": 0.1, "refused": None}[status],
        fingerprint="fixture-source-decision", model="gpt-6-luna", prompt_fingerprint="fixture-instructions",
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
    })
    prepare_synthesis_state(state, decisions=_rejecting_decisions(rejected_status))
    return state, accepted, rejected


def _rejecting_decisions(status="failed"):
    """Reject the deliberately unrelated fixture through the actual provider boundary."""
    def respond(request):
        payload = json.loads(request.content)
        context = json.loads(payload["input"])
        rejected = {item["name"] for item in context.get("evidence", [])
                    if "creates missing directories" in item["source"]["excerpt"]
                    or item["task"]["requirement_id"] == "path_exists_other"}
        answers = []
        for question in payload["questions"]:
            name = question["name"]
            blocked = name.split("_support_")[0] in rejected
            answers.append({"type": "refusal", "name": name} if blocked and status == "refused" else
                           {"type": "predicate", "name": name, "probability": 0.1 if blocked else 0.99})
        return httpx.Response(200, json={"model": "gpt-6-luna", "answers": answers,
                                        "usage": {"input_tokens": 32, "output_tokens": 0, "total_tokens": 32}})

    return DecisionsClient(OpenAI(api_key="test-key", base_url="https://provider.test/v1", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(respond))))


def test_prevalidation_keeps_raw_candidates_and_selects_verdicts_per_requirement():
    """재검색은 통과 원문을 유지하고 사전 검증은 새 지침으로 거절 원문도 재평가할 수 있다."""
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

    snapshot, errors = collect_validation_snapshot(state)

    assert {(hit.requirement_id, hit.evidence.id) for hit in snapshot.parsed_hits} == {
        (hit.requirement_id, hit.evidence.id) for hit in raw_hits
    }
    assert errors == []
    prepare_synthesis_state(state, decisions=_rejecting_decisions())
    packet = state["retrieval"].prepared_evidence.normal
    assert {item.id for item in packet.evidence} == {accepted.evidence.id, unjudged.evidence.id}
    assert all(second_task.requirement_id not in ids for ids in packet.requirement_map.values())
    assert state["retrieval"].hit_log == raw_hits


def test_refresh_evaluates_new_evidence_after_lexically_stronger_candidates_are_rejected(monkeypatch):
    """과거 거절 후보가 검증 한도를 채워도 새 검색의 유효 근거를 평가한다."""
    import requests

    from src.infra.settings import AppSettings
    from src.infra.tools.docs_search.tool import build_docs_search_tool
    from src.runtime.nodes.validation import make_pre_synthesis_validation_node

    source_url = "https://numpy.org/doc/stable/reference/generated/numpy.reshape.html"
    previous_results = [{
        "url": f"https://numpy.org/doc/stable/reference/stale-index-{index}.html",
        "title": "numpy.reshape order copy", "score": 0.99,
        "raw_content": f"numpy.reshape order copy index entry {index}. This index has no API details.",
    } for index in range(12)]
    new_result = {
        "url": source_url, "title": "Reshaping arrays", "score": 0.9,
        "raw_content": "# numpy.reshape\nThe traversal mode selects row-major or column-major index traversal. "
                       "The result shares memory when possible; otherwise a new array is allocated.",
    }
    queries = []

    def post(url, **kwargs):
        queries.append(kwargs["json"]["query"])
        response = requests.Response()
        response.status_code = 200
        response.url = url
        response._content = json.dumps({"results": previous_results if len(queries) == 1 else [new_result]}).encode()
        response._content_consumed = True
        return response

    evaluated_sources = []

    def respond(request):
        payload = json.loads(request.content)
        context = json.loads(payload["input"])
        evaluated_sources.extend(item["source"]["uri"] for item in context.get("evidence", []))
        entries = {item["name"]: item for collection in ("evidence", "requirements", "packets")
                   for item in context.get(collection, [])}
        answers = []
        for question in payload["questions"]:
            entry = entries[question["name"].split("_support_")[0]]
            usable = source_url in json.dumps(entry)
            answers.append({"type": "predicate", "name": question["name"], "probability": 0.99 if usable else 0.1})
        return httpx.Response(200, json={"model": "gpt-6-luna", "answers": answers,
                                        "usage": {"input_tokens": 32, "output_tokens": 0, "total_tokens": 32}})

    monkeypatch.setattr(requests, "post", post)
    tool = build_docs_search_tool(AppSettings(_env_file=None, tavily_api_key="test", openai_api_key="test"))
    dispatch = make_retrieve_dispatch_node(tool, None, False)
    decisions = DecisionsClient(OpenAI(
        api_key="test-key", base_url="https://provider.test/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ))
    validate = make_pre_synthesis_validation_node(
        False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=decisions,
    )
    task = RetrievalTask(route="docs", query="numpy.reshape order copy", k=4,
                         requirement={"library": "numpy", "symbols": ["numpy.reshape"], "aspects": ["order", "copy"]})
    state = build_test_state({
        "user_input": "numpy.reshape의 order와 copy 동작을 설명해 주세요.",
        "request_contract": RequestContract(), "planner_output": PlannerOutput(use_retrieval=True, tasks=[task]),
    })
    state.update(dispatch(state))
    previous_hits = list(state["retrieval"].hit_log)
    assert len(previous_hits) == 12
    state.update(validate(state))
    assert len(state["retrieval"].evidence_verdicts) == 12
    assert all(item.status == "failed" for item in state["retrieval"].evidence_verdicts)
    assert state["retry"].needs_retry is True
    assert state["retry"].failed_requirement_ids == [task.requirement_id]

    state.update(dispatch(state))
    assert len(queries) == 2
    assert state["retrieval"].hit_log[:len(previous_hits)] == previous_hits
    assert source_url in {hit.evidence.snapshot.source_uri for hit in state["retrieval"].hit_log}
    state.update(validate(state))

    assert source_url in evaluated_sources
    packet = state["retrieval"].prepared_evidence.normal
    assert {item.snapshot.source_uri for item in packet.evidence} == {source_url}
    assert packet.failed_requirement_ids == []
    assert state["retry"].needs_retry is False
    assert state["retry"].failed_requirement_ids == []
    assert state["retry"].terminal_failed_requirement_ids == []
    assert state["retry"].retry_reason is None
    assert state["retry"].terminal_retrieval_reason is None


@pytest.mark.parametrize("rejected_status", ["failed", "refused"])
def test_refresh_returning_the_same_rejected_source_does_not_restore_it_to_synthesis(rejected_status):
    """재검색이 실패 원문을 다시 반환해도 새 합성과 검증에는 통과·새 원문만 들어간다."""
    from src.core.contracts import RetrievalDiagnostic
    from src.core.contracts.retrieval import RetrievalResult
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

    snapshot, errors = collect_validation_snapshot(state)
    expected = {accepted.evidence.id, candidate.evidence.id}
    current = state["retrieval"].hit_log[state["retry"].hit_start_index:]
    assert {hit.evidence.id for hit in snapshot.parsed_hits} == {hit.evidence.id for hit in current}
    assert errors == []
    prepare_synthesis_state(state, decisions=_rejecting_decisions(rejected_status))
    assert {item.id for item in state["retrieval"].prepared_evidence.normal.evidence} == expected
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
    assert updates["retrieval"].hit_log[2:] == [accepted, rejected]
    diagnostic = updates["debug"].retrieval_diagnostics[-1]
    assert diagnostic.reused is True
    assert diagnostic.evidence_count == 2
    state.update(updates)
    prepare_synthesis_state(state, decisions=_rejecting_decisions("refused"))
    assert [item.id for item in state["retrieval"].prepared_evidence.normal.evidence] == [accepted.evidence.id]


def test_complementary_upload_refresh_can_complete_coverage_with_preserved_sources():
    """새 검색 결과 단독으로는 partial이어도 통과 원문과 합친 실제 범위를 평가한다."""
    from types import SimpleNamespace

    from src.core.contracts import RetrievalState
    from src.infra.tools.local_rag.tool import build_upload_search_tool
    from src.runtime.nodes.validation import make_pre_synthesis_validation_node
    from tests.core.helpers import make_decisions_client
    from tests.core.test_answer_validation import _evidence, _hit

    task = RetrievalTask(route="upload", query="alpha beta", k=1, requirement_id="parts",
                         requirement={"match": "topic", "aspects": ["alpha", "beta"]})
    accepted = _hit(_evidence("alpha = 1\n")).model_copy(update={"requirement_id": task.requirement_id})
    replacement = _evidence("beta = 2\n")
    vectorstore = SimpleNamespace(similarity_search_with_score=lambda query, **kwargs: [(replacement, 0.1)])
    state = build_test_state({
        "user_input": "alpha와 beta를 설명해 주세요.", "request_contract": RequestContract(),
        "planner_output": PlannerOutput(use_retrieval=True, tasks=[task]),
        "retriever": SimpleNamespace(vectorstore=vectorstore),
        "retrieval": RetrievalState(hit_log=[accepted], evidence_verdicts=[_verdict(accepted, "passed")]),
        "retry": RetryState(attempt=1, max_retries=1, retry_reason="no_evidence",
                            failed_requirement_ids=[task.requirement_id], hit_start_index=1),
    })
    state.update(make_retrieve_dispatch_node(None, build_upload_search_tool(), False)(state))
    assert state["debug"].retrieval_diagnostics[-1].answerability == "partial"

    state.update(make_pre_synthesis_validation_node(False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=make_decisions_client())(state))

    assert state["retry"].needs_retry is False
    assert state["retry"].failed_requirement_ids == []
    assert state["retry"].terminal_failed_requirement_ids == []
    assert state["retry"].retry_reason is None
    assert {item.evidence_id for item in state["retrieval"].evidence_verdicts if item.status == "passed"} == {
        accepted.evidence.id, replacement.id,
    }


@pytest.mark.parametrize("rejected_status", ["failed", "refused"])
def test_rejected_source_cannot_keep_a_previously_covered_upload_requirement_complete(rejected_status):
    """원문 제외로 필요한 항목이 빠지면 과거 covered 진단으로 누락을 가리지 않는다."""
    from src.core.contracts import RetrievalDiagnostic, RetrievalState
    from src.runtime.nodes.validation import make_pre_synthesis_validation_node
    from tests.core.helpers import make_decisions_client
    from tests.core.test_answer_validation import _evidence, _hit

    task = RetrievalTask(route="upload", query="alpha beta", k=2, requirement_id="parts",
                         requirement={"match": "topic", "aspects": ["alpha", "beta"]})
    accepted, rejected = [_hit(_evidence(text)).model_copy(update={"requirement_id": task.requirement_id})
                          for text in ("alpha = 1\n", "beta = 2\n")]
    state = build_test_state({
        "user_input": "alpha와 beta를 설명해 주세요.", "request_contract": RequestContract(),
        "planner_output": PlannerOutput(use_retrieval=True, tasks=[task]),
        "retrieval": RetrievalState(hit_log=[accepted, rejected],
                                    evidence_verdicts=[_verdict(accepted, "passed"), _verdict(rejected, rejected_status)]),
        "retrieval_diagnostics": [RetrievalDiagnostic(route="upload", requirement_id=task.requirement_id,
                                                       status="success", answerability="covered")],
    })

    decisions = make_decisions_client(
        {"evidence_1": 0.1, "requirement_0": 0.1, "request_ready": 0.1},
        refused_names={"evidence_1"} if rejected_status == "refused" else set(),
    )
    state.update(make_pre_synthesis_validation_node(
        False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=decisions,
    )(state))

    assert state["retry"].needs_retry is True
    assert state["retry"].failed_requirement_ids == [task.requirement_id]
    assert state["retry"].preserved_hits == [accepted]


def test_complete_retained_coverage_does_not_hide_a_partial_provider_failure():
    """범위를 합쳐 충족해도 검색 중 보고된 기술적 실패는 완료로 바꾸지 않는다."""
    from src.core.contracts import RetrievalDiagnostic, RetrievalState
    from src.runtime.nodes.validation import make_pre_synthesis_validation_node
    from tests.core.helpers import make_decisions_client
    from tests.core.test_answer_validation import _evidence, _hit

    task = RetrievalTask(route="upload", query="alpha", k=1, requirement_id="part",
                         requirement={"match": "topic", "aspects": ["alpha"]})
    accepted = _hit(_evidence("alpha = 1\n")).model_copy(update={"requirement_id": task.requirement_id})
    state = build_test_state({
        "user_input": "alpha를 설명해 주세요.", "request_contract": RequestContract(),
        "planner_output": PlannerOutput(use_retrieval=True, tasks=[task]),
        "retrieval": RetrievalState(hit_log=[accepted], evidence_verdicts=[_verdict(accepted, "passed")]),
        "retrieval_diagnostics": [RetrievalDiagnostic(route="upload", requirement_id=task.requirement_id,
            status="success", answerability="partial", error_code="LOCAL_RAG_FAILED")],
        "retry": RetryState(max_retries=0),
    })

    state.update(make_pre_synthesis_validation_node(False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=make_decisions_client())(state))

    assert state["retry"].terminal_failed_requirement_ids == [task.requirement_id]
    assert state["retry"].terminal_retrieval_reason == "tool_error"


@pytest.mark.parametrize("same_query", [False, True], ids=["rewritten-query", "same-query"])
def test_dispatch_retries_rewritten_query_timeout_and_keeps_previous_candidates(monkeypatch, same_query):
    """A failed rewritten query remains retryable without discarding earlier original sources."""
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
    assert (first.status, first.answerability, first.error_code) == ("success", "unknown", None)
    revised = task.model_copy(update={"query": task.query if same_query else task.query + " copy"})
    state["planner"] = state["planner"].model_copy(update={"output": PlannerOutput(use_retrieval=True, tasks=[revised])})
    state["retry"] = RetryState(attempt=1, retry_reason="no_evidence", failed_requirement_ids=[task.requirement_id])
    state.update(dispatch(state))
    failed = state["debug"].retrieval_diagnostics[-1]
    assert (failed.status, failed.answerability, failed.error_code) == ("success", "unknown", "RETRIEVAL_DOCS_TIMEOUT")
    state["retry"] = RetryState(attempt=2, retry_reason="no_evidence", failed_requirement_ids=[task.requirement_id])
    state.update(dispatch(state))
    final = state["debug"].retrieval_diagnostics[-1]
    assert (final.reused, final.answerability, final.error_code) == (False, "unknown", None)
    assert len(queries) == 3
    assert queries[0] == task.query
    assert queries[1] == queries[2]
    assert queries[1] == revised.query
    assert first_hit.evidence in [hit.evidence for hit in state["retrieval"].hit_log[1:]]
