"""Documentation requirements survive the actual model evidence budget."""

import json

import pytest
import requests

from src.core.contracts import PlannerState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.evidence import SearchHit
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.infra.settings import AppSettings
from src.infra.tools.docs_search.tool import build_docs_search_tool
from src.infra.tools.docs_search.url_validation import validate_doc_url
from src.runtime.nodes.synthesis.budgets import ExcerptLimits, RetrievedEvidenceBudget, resolve_evidence_budgets
from src.runtime.nodes.synthesis.context import build_synthesis_context, prepare_synthesis_inputs


def _retrieve(monkeypatch, body, *, need, symbol="numpy.mean", aspects=None):
    """Only HTTP is fake; search selection and synthesis preparation are real."""
    document_symbol = symbol.replace("np.", "numpy.")
    source = {
        "url": f"https://numpy.org/doc/stable/reference/generated/{document_symbol}.html",
        "title": document_symbol, "content": body, "raw_content": body, "score": 0.9,
    }

    def post(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"results": [source]}).encode()
        return response

    def head(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response.url = url
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "head", head)
    validate_doc_url.cache_clear()
    needs = [need] if isinstance(need, str) else need
    task = RetrievalTask(
        route="docs", query=f"{symbol} {' '.join(needs)}", k=1,
        requirement={"library": "numpy", "symbols": [symbol], "evidence_needs": needs, "aspects": aspects or []},
    )
    try:
        result = build_docs_search_tool(AppSettings(_env_file=None, tavily_api_key="test", openai_api_key="test"))(
            task.query, requirement=task.requirement, k=task.k,
        )
    finally:
        validate_doc_url.cache_clear()
    assert result["diagnostics"]["answerability"] == "covered"
    hits = [SearchHit.model_validate(hit).model_copy(update={"requirement_id": task.requirement_id})
            for hit in result["hits"]]
    return task, hits


def _prepare(task, hits, *, excerpt_chars=1800, max_items=8):
    state = build_graph_state_input(
        user_input=task.query, planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task])),
        request_contract=RequestContract(),
        retrieval={"hit_log": [hit.model_dump(mode="json") for hit in hits]},
    )
    prepared = prepare_synthesis_inputs(
        state=state, context=build_synthesis_context(state=state),
        budget=RetrievedEvidenceBudget(excerpt_chars, 6000, max_items), max_turns=6,
    )
    requirement_message = next(str(message.content) for message in prepared.model_messages
                               if str(message.content).startswith("[Retrieval Requirements]"))
    records = json.loads(requirement_message[requirement_message.index("\n[") + 1:])
    return prepared, records[0]["coverage"]












