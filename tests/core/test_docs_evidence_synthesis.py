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

    monkeypatch.setattr(requests, "post", post)
    needs = [need] if isinstance(need, str) else need
    task = RetrievalTask(
        route="docs", query=f"{symbol} {' '.join(needs)}", k=1,
        requirement={"library": "numpy", "symbols": [symbol], "evidence_needs": needs, "aspects": aspects or []},
    )
    result = build_docs_search_tool(AppSettings(_env_file=None, tavily_api_key="test", openai_api_key="test"))(
        task.query, requirement=task.requirement, k=task.k,
    )
    assert result.diagnostics.answerability == "covered"
    hits = [SearchHit.model_validate(hit).model_copy(update={"requirement_id": task.requirement_id})
            for hit in result.hits]
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














def test_parameter_aspect_does_not_displace_a_separate_required_example(monkeypatch):
    """A covered parameter cannot repeatedly defeat the still-missing example passage."""
    description = (
        "order: C reads and writes the last axis first, while F reads and writes the first axis first. "
        "A follows Fortran order for a Fortran contiguous array and C order otherwise. "
        + "The index order controls reading and writing values without changing the original array data. " * 6
        + "\n"
    )
    examples = "Examples\n>>> numpy.reshape([1, 2, 3, 4], (2, 2))\narray([[1, 2], [3, 4]])\n"
    body = (
        "# numpy.reshape\nnumpy.reshape(a, shape, order='C')\n\nParameters\n"
        + description + "\nNotes\n" + "Additional background about storage and memory layout. " * 25
        + "\n\n" + examples
    )
    assert len(body) > 1800
    task, hits = _retrieve(monkeypatch, body, need=["examples", "explanation", "parameters"],
                           symbol="numpy.reshape", aspects=["order"])

    prepared, coverage = _prepare(task, hits)

    assert coverage["missing_aspects"] == []
    assert coverage["missing_needs"] == []
    assert prepared.missing_requirement_ids == []
    assert any(description.strip() in item.excerpt for item in prepared.evidence_packet)
    assert any(examples.strip() in item.excerpt for item in prepared.evidence_packet)
    assert len(prepared.evidence_packet) >= 2
    assert all(len(item.excerpt) <= 1800 for item in prepared.evidence_packet)
    assert sum(len(item.excerpt) for item in prepared.evidence_packet) <= 6000
    for selected in prepared.evidence_packet:
        assert any(
            original.evidence.snapshot == selected.snapshot
            and original.evidence.element == selected.element
            and original.evidence.selection.start <= selected.selection.start
            and selected.selection.end <= original.evidence.selection.end
            for original in hits
        )
        assert selected.excerpt == body[selected.selection.start:selected.selection.end]
