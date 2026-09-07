"""Public node contracts for preserving independent evidence requirements."""
from langchain_core.messages import HumanMessage

from src.core.contracts.boundary.planner import parse_planner_output
from src.core.contracts import RetrievalDiagnostic, RetryState
from src.core.planner_schema import PlannerOutput
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.retry import format_retry_context_for_planner
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
















def test_one_planned_requirement_cannot_mix_an_enclosing_definition_and_a_callee():
    """Each planned symbol has its own role instead of sharing a possibly wrong match mode."""
    with pytest.raises(ValidationError):
        RetrievalTask(route="upload", query="handler의 persist 호출을 설명", k=3,
                      requirement={"symbols": ["handler", "persist"], "match": "definition"})


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
