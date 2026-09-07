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










def test_replanning_changes_queries_without_dropping_original_sources_or_versions():
    """A retry cannot silently remove or weaken an original evidence requirement."""
    first = RetrievalTask(route="docs", query="numpy reshape old", k=1, requirement_id="numpy",
                          requirement={"library": "numpy", "symbols": ["numpy.reshape"], "version": "1.26"})
    second = RetrievalTask(route="docs", query="pandas concat", k=1, requirement_id="pandas",
                           requirement={"library": "pandas", "symbols": ["pandas.concat"]})
    revised = first.model_copy(update={"query": "numpy reshape reference", "requirement": first.requirement.model_copy(update={"version": None})})
    state = build_test_state({"user_input": "Compare requested sources", "retry_context": {
        "attempt": 1, "original_tasks": [first.model_dump(), second.model_dump()],
    }})
    result = make_planner_node(_CapturePlannerLLM(PlannerOutput(use_retrieval=True, tasks=[revised])), False)(state)
    tasks = result["planner"].output.tasks
    assert [(task.requirement_id, task.requirement) for task in tasks] == [(first.requirement_id, first.requirement), (second.requirement_id, second.requirement)]
    assert tasks[0].query == "numpy reshape reference"


def test_planner_does_not_promote_guessed_parameter_values_to_required_evidence():
    """Unrequested model guesses cannot make a valid source fail a harder invented requirement."""
    query = "engine.reshape의 mode 매개변수를 설명해줘"
    guessed = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(
        route="docs", query="engine.reshape mode invented_value documentation", k=3,
        requirement={"symbols": ["engine.reshape"], "aspects": ["mode", "invented_value"], "match": "symbol"},
    )])
    result = make_planner_node(_CapturePlannerLLM(guessed), False)(build_test_state({"user_input": query}))
    task = result["planner"].output.tasks[0]
    assert task.requirement.aspects == ["mode"]
    assert task.query == guessed.tasks[0].query


def test_explicitly_requested_parameter_value_survives_requirement_grounding():
    """The same term remains a real requirement when the user actually asks about it."""
    query = "engine.reshape mode=invented_value의 지원 여부를 확인해줘"
    requested = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(
        route="docs", query=query, k=3,
        requirement={"symbols": ["engine.reshape"], "aspects": ["mode", "invented_value"], "match": "symbol"},
    )])
    result = make_planner_node(_CapturePlannerLLM(requested), False)(build_test_state({"user_input": query}))
    assert result["planner"].output.tasks[0].requirement == requested.tasks[0].requirement


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


def test_planner_returns_the_missing_reference_question_without_retrieval():
    """An unresolved referent produces the intended clarification, not a failed search."""
    question = "어떤 라이브러리와 버전을 비교할까요?"
    model = _CapturePlannerLLM({"use_retrieval": False, "tasks": [], "clarification_question": question})
    result = make_planner_node(model, verbose=False)(build_test_state({
        "user_input": "그거 최신 버전에서 어떻게 바뀌었어?",
        "messages": [HumanMessage(content="그거 최신 버전에서 어떻게 바뀌었어?")],
    }))
    assert result["planner"].guided_followup == question
    assert result["planner"].diagnostics.reason == "clarification_required"
    assert not result["planner"].output.use_retrieval


def test_retry_prompt_exposes_prior_query_and_filter_failure_without_source_text():
    """Replanning receives the failed request and cause rather than just its route."""
    state = build_test_state({
        "user_input": "NumPy reshape의 order를 설명해줘",
        "planner_output": PlannerOutput(use_retrieval=True, tasks=[
            {"route": "docs", "query": "np.reshape order", "k": 3},
        ]),
        "debug": {"retrieval_diagnostics": [RetrievalDiagnostic(
            route="docs", query="np.reshape order", status="no_result",
            warnings=["identifier_coverage_incomplete"], provider_result_count=3,
        )]},
    })
    prompt = format_retry_context_for_planner(state, RetryState(
        attempt=1, retry_reason="no_evidence", failed_routes=["docs"],
        retrieval_feedback="identifier missing after filtering",
    ))
    assert "np.reshape order" in prompt
    assert "identifier_coverage_incomplete" in prompt
    assert "identifier missing after filtering" in prompt
