"""Public node contracts for preserving independent evidence requirements."""

from src.core.contracts import RetryState
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import RequestContract
from src.runtime.nodes.planner import make_planner_node
from tests.core.helpers import build_test_state
from src.core.evidence import SearchHit
from src.core.planner_schema import RetrievalTask
import json
import requests
from src.infra.settings import AppSettings
from src.infra.tools import build_tool_registry
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
import pytest


def test_dispatch_keeps_independent_official_candidates_with_mixed_provider_results(monkeypatch):
    """Each docs task retains its allowed library sources without claiming semantic coverage."""
    tasks = [
        RetrievalTask(route="docs", query="numpy.concatenate", k=2,
                      requirement={"library": "numpy", "symbols": ["numpy.concatenate"], "match": "definition"}),
        RetrievalTask(route="docs", query="pandas.concat", k=2,
                      requirement={"library": "pandas", "symbols": ["pandas.concat"], "match": "definition"}),
    ]
    sources = [
        {"url": "https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
         "title": "numpy.concatenate", "content": "numpy.concatenate joins arrays along an existing axis.", "score": 0.9},
        {"url": "https://pandas.pydata.org/docs/reference/api/pandas.concat.html",
         "title": "pandas.concat", "content": "pandas.concat combines pandas objects along an axis.", "score": 0.9},
    ]
    requests_seen = []

    def post(url, **kwargs):
        requests_seen.append(kwargs["json"])
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"results": sources}).encode()
        return response

    def head(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response.url = url
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "head", head)
    registry = build_tool_registry(AppSettings(_env_file=None, openai_api_key="test", tavily_api_key="test"))
    dispatch = make_retrieve_dispatch_node(registry.tavily_search_tool, registry.upload_search_tool, False)

    result = dispatch(build_test_state({
        "user_input": "Compare both official sources", "planner_output": PlannerOutput(use_retrieval=True, tasks=tasks),
    }))

    hits = [SearchHit.model_validate(hit) for hit in result["retrieval"].hit_log]
    assert len(hits) == 2
    by_requirement = {hit.requirement_id: hit for hit in hits}
    for task, source in zip(tasks, sources, strict=True):
        hit = by_requirement[task.requirement_id]
        assert hit.evidence.snapshot.source_uri == source["url"]
        assert hit.evidence.element.text == source["content"]
    diagnostics = result["debug"].retrieval_diagnostics
    assert {diagnostic.requirement_id for diagnostic in diagnostics} == {task.requirement_id for task in tasks}
    assert all(diagnostic.answerability == "unknown" and diagnostic.status == "success" for diagnostic in diagnostics)
    assert all(diagnostic.missing_requirements == [] for diagnostic in diagnostics)
    assert all(diagnostic.evidence_count == diagnostic.final_evidence_count == 1 for diagnostic in diagnostics)
    assert all(diagnostic.filtered_cross_domain_count == 1 for diagnostic in diagnostics)
    assert len(requests_seen) == 2
    assert {tuple(request["include_domains"]) for request in requests_seen} == {("numpy.org",), ("pandas.pydata.org",)}


class _RewriteModel:
    def __init__(self, rewrites):
        self.rewrites = rewrites

    def invoke(self, _messages):
        return {"rewrites": self.rewrites}


def _rewrite_state():
    failed = RetrievalTask(route="docs", query="numpy reshape old", k=2, requirement_id="failed",
                           requirement={"library": "numpy", "symbols": ["numpy.reshape"], "version": "1.26",
                                        "aspects": ["order"], "evidence_needs": ["parameters"]})
    successful = RetrievalTask(route="docs", query="pandas concat", k=2, requirement_id="successful")
    contract = RequestContract()
    state = build_test_state({"user_input": "continue", "request_contract": contract,
                             "planner_output": PlannerOutput(use_retrieval=True, tasks=[failed, successful]),
                             "retry": RetryState(attempt=1, failed_requirement_ids=[failed.requirement_id],
                                                  original_tasks=[failed.model_dump(), successful.model_dump()])})
    return state, failed, successful, contract


def test_retrieval_rewrite_rejects_an_unknown_requirement():
    from src.core.llm_errors import LLMCallError
    rewrites = [{"requirement_id": "unknown", "query": "new", "k": 3}]
    state, failed, successful, contract = _rewrite_state()
    with pytest.raises(LLMCallError) as caught:
        make_planner_node(None, False, llm_planner_retry=_RewriteModel(rewrites))(state)
    assert caught.value.problem.code == "model_output_invalid"
    assert state["retry"].original_tasks == [failed, successful]
    assert state["runtime"].request_contract == contract
