"""Partial Decisions failures retain approved sources across real graph nodes."""
from __future__ import annotations

from functools import partial
import json

import httpx
import pytest
from openai import OpenAI

from src.core.answer_schema import export_answer_text
from src.core.contracts import RetryState
from src.core.contracts.debug import RetrievalDiagnostic
from src.core.contracts.retrieval import RetrievalResult
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.infra.decisions import DecisionsClient
from src.infra.llm_boundary import capture_llm_diagnostics
from src.runtime.agent_runtime.response_assembler import ResponseAssembler
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.core.helpers import _CapturePlannerLLM, build_test_state, make_decisions_client
from tests.core.test_answer_validation import _evidence, _hit
from tests.core.test_pending_action_delivery import _contract
from tests.synthesis_fixtures import synthesis_excerpt_limits


make_pre_synthesis_validation_node = partial(make_pre_synthesis_validation_node, excerpt_limits=synthesis_excerpt_limits())


class DecisionProvider:
    """Replace provider HTTP; keep Decisions parsing and validation policies real."""

    def __init__(self, *, rejected=(), required_text=None, refusal=True):
        self.rejected = set(rejected)
        self.required_text = dict(required_text or {})
        self.refusal = refusal
        self.source_requests = []
        self.requirement_requests = []
        self.http = httpx.Client(transport=httpx.MockTransport(self.respond))
        self.client = DecisionsClient(OpenAI(
            api_key="test-key", base_url="https://provider.test/v1", http_client=self.http, max_retries=0,
        ))

    def respond(self, request):
        assert request.url.path == "/v1/decisions"
        payload = json.loads(request.content)
        context = json.loads(payload["input"])
        candidates = {item["name"]: item for item in context.get("evidence", [])}
        requirements = {item["name"]: item for item in context.get("requirements", []) if "task" in item}
        self.source_requests.extend(candidates.values())
        if requirements:
            self.requirement_requests.append(list(requirements.values()))
        answers = []
        for question in payload["questions"]:
            name = question["name"]
            probability = 0.99
            if name in candidates and candidates[name]["source"]["id"] in self.rejected:
                if self.refusal:
                    answers.append({"type": "refusal", "name": name})
                    continue
                probability = 0.1
            if name in requirements:
                requirement = requirements[name]
                expected = self.required_text.get(requirement["task"]["requirement_id"])
                if expected and not any(expected in source["excerpt"] for source in requirement["sources"]):
                    probability = 0.1
            answers.append({"type": "predicate", "name": name, "probability": probability})
        return httpx.Response(200, json={
            "model": "gpt-6-luna", "answers": answers,
            "usage": {"input_tokens": 32, "output_tokens": 0, "total_tokens": 32},
        })


@pytest.fixture
def decision_provider():
    providers = []

    def create(**kwargs):
        provider = DecisionProvider(**kwargs)
        providers.append(provider)
        return provider

    with capture_llm_diagnostics():
        yield create
    for provider in providers:
        provider.http.close()


class CitedBodyModel:
    def __init__(self):
        self.messages = []

    def with_structured_output(self, *args, **kwargs):
        return self

    def invoke(self, messages, *, timeout=None):
        self.messages = messages
        packet_text = next(str(message.content) for message in messages
                           if str(message.content).startswith("[Evidence Packet]"))
        packet = json.loads(packet_text[packet_text.index("[", len("[Evidence Packet]")):])
        return {"blocks": [{"type": "paragraph", "content": [{
            "text": "The verified sources explain the available behavior.",
            "basis": "source", "refs": [source["id"] for source in packet],
        }]}]}


def _task(requirement_id, *, route="docs", match="topic"):
    return RetrievalTask(route=route, query=requirement_id, requirement_id=requirement_id, k=3,
                         requirement={"match": match, "symbols": ["rejected_definition"] if match == "definition" else []})


def _result(evidence):
    return RetrievalResult(hits=[_hit(item) for item in evidence], diagnostics=RetrievalDiagnostic(
        status="success", answerability="covered", evidence_count=len(evidence),
    ))


def _state(tasks, *, max_retries=1):
    return build_test_state({
        "user_input": "Explain all requested source behavior, then save and send the answer",
        "request_contract": _contract(save="requested", slack="requested"),
        "planner_output": PlannerOutput(use_retrieval=True, tasks=tasks),
        "retry": RetryState(max_retries=max_retries, max_content_repairs=0, original_tasks=tasks),
    })


def _synthesize(state):
    model = CitedBodyModel()
    state.update(make_synthesize_node(model)(state))
    return model


@pytest.mark.parametrize("refusal", [True, False], ids=["refusal", "below-threshold"])
def test_one_rejected_source_does_not_retry_a_requirement_covered_by_its_other_source(decision_provider, refusal):
    task = _task("available")
    good = _evidence("available VERIFIED_SOURCE", source_type="official")
    bad = _evidence("available REJECTED_SOURCE", source_type="official")
    state = _state([task])
    provider = decision_provider(rejected={bad.id}, refusal=refusal)
    state.update(make_retrieve_dispatch_node(lambda **kwargs: _result([good, bad]), None, False)(state))

    state.update(make_pre_synthesis_validation_node(False, decisions_client=provider.client)(state))
    model = _synthesize(state)

    assert not state["retry"].needs_retry
    assert state["retry"].attempt == 0
    assert [item.id for item in state["response"].evidence_packet] == [good.id]
    assert {item["source"]["id"] for item in provider.source_requests} == {good.id, bad.id}
    assert [source["id"] for source in provider.requirement_requests[-1][0]["sources"]] == [good.id]
    assert "VERIFIED_SOURCE" in str(model.messages)
    assert "REJECTED_SOURCE" not in str(model.messages)


def test_retry_preserves_passed_source_within_failed_requirement_and_skips_successful_requirement(decision_provider):
    missing, covered = _task("missing"), _task("covered")
    kept = _evidence("missing VERIFIED_OLD_SOURCE", source_type="official")
    rejected = _evidence("missing REJECTED_SOURCE", source_type="official")
    other = _evidence("covered VERIFIED_OTHER_SOURCE", source_type="official")
    fresh = _evidence("missing NEW_REQUIRED_DETAIL", source_type="official")
    searches = []

    def search(*, query, **kwargs):
        searches.append(query)
        return _result({"missing": [kept, rejected], "covered": [other], "missing revised": [fresh]}[query])

    state = _state([missing, covered])
    provider = decision_provider(rejected={rejected.id}, required_text={"missing": "NEW_REQUIRED_DETAIL"})
    dispatch = make_retrieve_dispatch_node(search, None, False)
    validate = make_pre_synthesis_validation_node(False, decisions_client=provider.client)
    state.update(dispatch(state))
    state.update(validate(state))

    assert state["retry"].needs_retry
    assert state["retry"].failed_requirement_ids == ["missing"]
    assert {hit.evidence.id for hit in state["retry"].preserved_hits} == {kept.id, other.id}
    rewrite = _CapturePlannerLLM({"rewrites": [{"requirement_id": "missing", "query": "missing revised", "k": 3}]})
    state.update(make_planner_node(None, False, llm_planner_retry=rewrite)(state))
    state.update(dispatch(state))
    state.update(validate(state))
    model = _synthesize(state)

    assert searches.count("missing") == searches.count("missing revised") == searches.count("covered") == 1
    assert not state["retry"].needs_retry
    assert {item.id for item in state["response"].evidence_packet} == {kept.id, other.id, fresh.id}
    evaluated_ids = [item["source"]["id"] for item in provider.source_requests]
    assert evaluated_ids.count(kept.id) == evaluated_ids.count(other.id) == evaluated_ids.count(rejected.id) == 1
    assert fresh.id in evaluated_ids
    assert "REJECTED_SOURCE" not in str(model.messages)


def test_unrefreshable_upload_failure_does_not_leak_when_docs_alone_are_researched(decision_provider):
    docs, upload = _task("docs"), _task("upload", route="upload", match="definition")
    docs = docs.model_copy(update={"query": "axis"})
    docs_bad = _evidence("axis REJECTED_DOC_SOURCE", source_type="official")
    upload_bad = _evidence("def rejected_definition():\n    return 'REJECTED_UPLOAD_SOURCE'\n")
    docs_good = _evidence("axis selects a dimension RECOVERED_DOC_SOURCE", source_type="official")
    searches = []

    def search_docs(*, query, **kwargs):
        searches.append(query)
        return _result([docs_good if query == "axis selection" else docs_bad])

    def search_upload(**kwargs):
        searches.append("upload")
        return _result([upload_bad])

    state = _state([docs, upload])
    provider = decision_provider(rejected={docs_bad.id, upload_bad.id}, required_text={
        "docs": "RECOVERED_DOC_SOURCE", "upload": "MISSING_DEFINITION",
    })
    dispatch = make_retrieve_dispatch_node(search_docs, search_upload, False)
    validate = make_pre_synthesis_validation_node(False, decisions_client=provider.client)
    state.update(dispatch(state))
    state.update(validate(state))

    assert state["retry"].needs_retry
    assert state["retry"].failed_requirement_ids == ["docs"]
    assert state["retry"].terminal_failed_requirement_ids == ["upload"]
    assert state["retry"].terminal_retrieval_reason == "no_evidence"
    assert not state["retry"].preserved_hits
    rewrite = _CapturePlannerLLM({"rewrites": [{"requirement_id": "docs", "query": "axis selection", "k": 3}]})
    state.update(make_planner_node(None, False, llm_planner_retry=rewrite)(state))
    state.update(dispatch(state))
    state.update(validate(state))
    model = _synthesize(state)

    assert searches.count("upload") == searches.count("axis") == searches.count("axis selection") == 1
    assert state["retry"].terminal_failed_requirement_ids == ["upload"]
    assert not state["retry"].needs_retry
    assert [item.id for item in state["response"].evidence_packet] == [docs_good.id]
    assert "REJECTED_UPLOAD_SOURCE" not in str(model.messages)
    assert "REJECTED_DOC_SOURCE" not in str(model.messages)


@pytest.mark.parametrize("max_retries", [0, 1], ids=["retry-disabled", "retry-also-rejected"])
def test_exhausted_retrieval_returns_only_passed_citations_and_prevents_delivery(decision_provider, max_retries):
    covered, missing = _task("covered"), _task("missing")
    good = _evidence("covered VERIFIED_SOURCE", source_type="official")
    bad = _evidence("missing REJECTED_SOURCE", source_type="official")
    retry_bad = _evidence("missing STILL_REJECTED_SOURCE", source_type="official")
    searches = []

    def search(*, query, **kwargs):
        searches.append(query)
        return _result([{"covered": good, "missing": bad, "missing revised": retry_bad}[query]])

    state = _state([covered, missing], max_retries=max_retries)
    provider = decision_provider(rejected={bad.id, retry_bad.id}, required_text={"missing": "MISSING_REQUIRED_DETAIL"})
    dispatch = make_retrieve_dispatch_node(search, None, False)
    validate = make_pre_synthesis_validation_node(False, decisions_client=provider.client)
    state.update(dispatch(state))
    state.update(validate(state))
    if max_retries:
        assert state["retry"].needs_retry
        rewrite = _CapturePlannerLLM({"rewrites": [{"requirement_id": "missing", "query": "missing revised", "k": 3}]})
        state.update(make_planner_node(None, False, llm_planner_retry=rewrite)(state))
        state.update(dispatch(state))
        state.update(validate(state))
    model = _synthesize(state)
    state.update(make_post_synthesis_validation_node(False, decisions_client=make_decisions_client())(state))
    delivered = []
    state.update(make_action_postprocess_node(
        lambda **kwargs: delivered.append("save"), lambda **kwargs: delivered.append("slack"), False,
    )(state))
    outcome = ResponseAssembler().assemble(response=state, debug_info={})

    assert outcome["status"] == "partial"
    assert state["response"].kind == "failure"
    assert state["retry"].terminal_failed_requirement_ids == ["missing"]
    assert not state["retry"].needs_retry
    assert state["retry"].attempt == max_retries
    assert searches.count("covered") == searches.count("missing") == 1
    assert searches.count("missing revised") == max_retries
    assert [citation.evidence.id for citation in state["response"].result.citations] == [good.id]
    assert "verified sources" in export_answer_text(state["response"].result)
    assert "answer_incomplete" in {issue.code for issue in state["response"].result.issues}
    assert "REJECTED_SOURCE" not in str(model.messages)
    assert delivered == []
    assert not state["runtime"].pending_action.body_prepared
