"""Public node contracts for preserving independent evidence requirements."""
from langchain_core.messages import HumanMessage

from src.core.contracts.boundary.planner import parse_planner_output
from src.core.contracts import RetrievalDiagnostic, RetryState
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import RequestContract, WireRequestContract, UnresolvedBody, MissingInformation
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.retry import build_retrieval_feedback, format_retry_context_for_planner
from tests.core.helpers import _CapturePlannerLLM, build_test_state
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import SearchHit, RetrievalScore, build_evidence
from src.core.planner_schema import RetrievalTask
from src.runtime.nodes.validation import make_pre_synthesis_validation_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from src.core.answer_schema import AnswerDocument, finalize_answer
import json
import requests
from src.infra.settings import AppSettings
from src.infra.tools import build_tool_registry
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
import pytest
from pydantic import ValidationError


def _official_hit(task):
    text = "numpy.concatenate joins arrays along axis."
    snapshot = build_snapshot(source_uri="https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
                              title="numpy.concatenate", media_type="text/plain", source_type="official",
                              content=text, parser="test", parser_version="1")
    return SearchHit(evidence=build_evidence(snapshot=snapshot, element=DocumentElement(element_id="body", kind="paragraph", text=text)),
                     score=RetrievalScore(metric="provider_score", raw=0.9, normalized=0.9, direction="higher"), rank=1,
                     requirement_id=task.requirement_id)










def test_dispatch_keeps_independent_official_sources_and_coverage_with_mixed_provider_results(monkeypatch):
    """Each docs task retains only its own library's source and consistent coverage diagnostics."""
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
    from src.infra.tools.docs_search.url_validation import validate_doc_url
    validate_doc_url.cache_clear()
    registry = build_tool_registry(AppSettings(_env_file=None, openai_api_key="test", tavily_api_key="test"))
    dispatch = make_retrieve_dispatch_node(registry.tavily_search_tool, registry.upload_search_tool, False)

    try:
        result = dispatch(build_test_state({
            "user_input": "Compare both official sources", "planner_output": PlannerOutput(use_retrieval=True, tasks=tasks),
        }))
    finally:
        validate_doc_url.cache_clear()

    hits = [SearchHit.model_validate(hit) for hit in result["retrieval"].hit_log]
    assert len(hits) == 2
    by_requirement = {hit.requirement_id: hit for hit in hits}
    for task, source in zip(tasks, sources, strict=True):
        hit = by_requirement[task.requirement_id]
        assert hit.evidence.snapshot.source_uri == source["url"]
        assert hit.evidence.element.text == source["content"]
    diagnostics = result["debug"].retrieval_diagnostics
    assert {diagnostic.requirement_id for diagnostic in diagnostics} == {task.requirement_id for task in tasks}
    assert all(diagnostic.answerability == "covered" and diagnostic.status == "success" for diagnostic in diagnostics)
    assert all(diagnostic.evidence_count == diagnostic.final_evidence_count == 1 for diagnostic in diagnostics)
    assert all(diagnostic.filtered_cross_domain_count == 1 for diagnostic in diagnostics)
    assert len(requests_seen) == 2
    assert {tuple(request["include_domains"]) for request in requests_seen} == {("numpy.org",), ("pandas.pydata.org",)}










def test_planner_keeps_independent_official_sources_on_the_same_route():
    """Two official-source requirements remain independently executable."""
    errors = []
    plan = parse_planner_output({"use_retrieval": True, "tasks": [
        {"route": "docs", "query": "numpy.concatenate axis", "k": 2},
        {"route": "docs", "query": "pandas.concat axis", "k": 3},
    ]}, errors)
    assert errors == []
    assert [(t.route, t.query, t.k) for t in plan.tasks] == [
        ("docs", "numpy.concatenate axis", 2), ("docs", "pandas.concat axis", 3),
    ]










def test_task_query_is_normalized_at_construction():
    task = RetrievalTask(route="docs", query="  numpy\n\treshape   axis  ", k=3)
    assert task.query == "numpy reshape axis"
