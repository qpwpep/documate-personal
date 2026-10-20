"""Evidence preparation reserves shared Decisions capacity for final packet checks."""
from __future__ import annotations

from src.core.contracts import DebugState, PlannerState, ResponseState, RetrievalDiagnostic, RetrievalState
from src.core.evidence import RetrievalScore, SearchHit
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.infra.llm_boundary import capture_llm_diagnostics, request_call_budget
from src.runtime.nodes.validation import make_pre_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary, _evidence, _state
from tests.synthesis_fixtures import synthesis_excerpt_limits


def _many_candidates_state(*, revision="initial"):
    aspects = ["axis", "dtype", "out", "keepdims", "where", "ddof", "correction", "mean"]
    tasks = [RetrievalTask(
        route="docs", query=f"{symbol} signature parameters examples explanation", k=10,
        requirement_id=f"api_{index}", requirement={
            "library": "numpy", "symbols": [symbol], "match": "symbol",
            "aspects": aspects, "evidence_needs": ["signature", "parameters", "examples", "explanation"],
        },
    ) for index, symbol in enumerate(["numpy.mean", "numpy.sum", "numpy.var", "numpy.std"])]
    hits = []
    for task in tasks:
        symbol = task.requirement.symbols[0]
        for index in range(20):
            evidence = _evidence(
                f"{symbol}: {revision} documentation passage {index}. "
                "Signature, parameters, examples and explanation for " + ", ".join(aspects) + ".",
                subject=symbol,
                source_uri=f"https://numpy.org/doc/stable/reference/generated/{symbol}.html#{revision}-{index}",
            )
            hits.append(SearchHit(
                evidence=evidence, requirement_id=task.requirement_id, rank=index + 1,
                score=RetrievalScore(metric="rank", raw=index + 1, direction="lower"),
            ))
    state = _state()
    state["runtime"] = state["runtime"].model_copy(update={
        "user_input": "각 API의 시그니처, 매개변수, 사용 예제와 동작을 비교해 주세요: "
        + ", ".join(task.requirement.symbols[0] for task in tasks),
    })
    state["planner"] = PlannerState(output=PlannerOutput(use_retrieval=True, tasks=tasks))
    state["retrieval"] = RetrievalState(hit_log=hits)
    state["response"] = ResponseState()
    state["debug"] = DebugState(retrieval_diagnostics=[RetrievalDiagnostic(
        route="docs", requirement_id=task.requirement_id, status="success", answerability="covered",
        evidence_count=20,
    ) for task in tasks])
    return state


def _assert_ready_for_every_requirement(state):
    required = {task.requirement_id for task in state["planner"].output.tasks}
    assert state["retry"].needs_retry is False
    assert state["retry"].retry_reason is None
    assert state["retry"].failed_requirement_ids == []
    assert state["response"].problem is None
    retrieval = state["retrieval"]
    assert {verdict.requirement_id for verdict in retrieval.evidence_verdicts
            if verdict.status == "passed"} == required
    prepared = retrieval.prepared_evidence
    assert prepared is not None
    assert prepared.compact is not None
    for packet in (prepared.normal, prepared.compact):
        assert packet.failed_requirement_ids == []
        assert {requirement for requirements in packet.requirement_map.values()
                for requirement in requirements} == required
        assert packet.evidence


def test_many_candidates_can_finish_packet_checks_after_retrieval_with_the_same_budget():
    """The original request's call cap also covers preparation after one re-search."""
    state = _many_candidates_state()
    boundary = DecisionsBoundary({
        name: 0.1 for index in range(4)
        for name in (f"requirement_{index}", f"compact_requirement_{index}")
    })
    node = make_pre_synthesis_validation_node(
        False, decisions_client=boundary, excerpt_limits=synthesis_excerpt_limits(),
    )

    with capture_llm_diagnostics() as diagnostics:
        state.update(node(state))
        assert state["retry"].needs_retry is True
        assert state["retry"].retry_reason == "no_evidence"
        assert set(state["retry"].failed_requirement_ids) == {"api_0", "api_1", "api_2", "api_3"}
        first_calls = len(boundary.requests)

        # A new search supplies different evidence; retain the prior judgments
        # and request context so both preparations really share the same budget.
        refreshed = _many_candidates_state(revision="expanded")
        state["retrieval"] = state["retrieval"].model_copy(update={
            "hit_log": refreshed["retrieval"].hit_log,
        })
        boundary.probabilities.clear()
        state.update(node(state))
        budget = request_call_budget("pre_synthesis_validation")

    _assert_ready_for_every_requirement(state)
    assert diagnostics == []
    assert budget.max_calls == 6
    assert first_calls < len(boundary.requests) == budget.calls <= budget.max_calls
    assert all("expanded" in evidence.excerpt
               for evidence in state["retrieval"].prepared_evidence.normal.evidence)


def test_optional_support_questions_cannot_block_one_sources_relevance_and_packet_checks():
    """Unasked support hints stay unknown instead of becoming cached rejections."""
    state, boundary = _state(), DecisionsBoundary()
    task = state["planner"].output.tasks[0].model_copy(deep=True)
    task.requirement.aspects = [f"option_{index}" for index in range(1200)]
    state["planner"] = PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task]))
    state["response"] = ResponseState()
    node = make_pre_synthesis_validation_node(
        False, decisions_client=boundary, excerpt_limits=synthesis_excerpt_limits(),
    )

    with capture_llm_diagnostics() as diagnostics:
        state.update(node(state))
        budget = request_call_budget("pre_synthesis_validation")

    _assert_ready_for_every_requirement(state)
    verdict, = state["retrieval"].evidence_verdicts
    assert verdict.status == "passed"
    assert "relevance" in verdict.supported_labels
    assert 1 < len(verdict.supported_labels) < 1 + len(task.requirement.aspects)
    # The fake returns only positive answers. Every stored judgment must come
    # from an actual answer, including when many support questions were omitted.
    cache = state["retrieval"].decision_cache
    answered_questions = sum(len(request["questions"]) for request in boundary.requests)
    assert len(cache) == answered_questions
    assert all(entry.probability == 0.99 for entry in cache)
    assert diagnostics == []
    assert budget.calls == len(boundary.requests) <= 6


def test_cached_source_judgments_keep_the_prepared_packet_when_no_new_question_budget_remains():
    """Cached approvals do not consume the allowance reserved for later checks."""
    state, boundary = _many_candidates_state(), DecisionsBoundary()
    node = make_pre_synthesis_validation_node(
        False, decisions_client=boundary, excerpt_limits=synthesis_excerpt_limits(),
    )

    with capture_llm_diagnostics() as diagnostics:
        state.update(node(state))
        _assert_ready_for_every_requirement(state)
        prepared = state["retrieval"].prepared_evidence
        verdicts = state["retrieval"].evidence_verdicts
        first_calls = len(boundary.requests)

        state.update(node(state))
        budget = request_call_budget("pre_synthesis_validation")

    _assert_ready_for_every_requirement(state)
    assert state["retrieval"].prepared_evidence == prepared
    assert state["retrieval"].evidence_verdicts == verdicts
    assert diagnostics == []
    assert budget.calls == len(boundary.requests) == first_calls
    assert budget.calls <= 6
