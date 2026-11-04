"""Keep the failed answer check distinct from an already known retrieval gap."""
from __future__ import annotations

import json

import pytest

from src.core.answer_schema import AnswerDocument, export_answer_text, finalize_answer, text_document
from src.core.contracts import GraphState, PlannerState, ResponseState, RetrievalDiagnostic, RetrievalState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.evidence import RetrievalScore, SearchHit
from src.core.llm_errors import make_problem
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import AnswerContract, LayoutRequirement, RequestContract, SentenceCountRequirement
from src.runtime.agent_runtime.response_assembler import ResponseAssembler
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary, _evidence, _state, read_decision_input
from tests.synthesis_fixtures import synthesis_excerpt_limits


def _partial_state() -> GraphState:
    state = _state(repairs=0)
    state["planner"].output.tasks.append(RetrievalTask(
        route="docs", query="pathlib.Path.is_dir behavior", k=1, requirement_id="path_is_dir",
        requirement={"library": "Python", "symbols": ["pathlib.Path.is_dir"], "match": "symbol"},
    ))
    state["debug"].retrieval_diagnostics.append(RetrievalDiagnostic(
        route="docs", requirement_id="path_is_dir", status="no_result", answerability="missing", evidence_count=0,
    ))
    state["retry"] = RetryState(
        max_retries=0, max_content_repairs=0, terminal_failed_requirement_ids=["path_is_dir"],
        terminal_retrieval_reason="no_evidence",
    )
    state["response"].normal_evidence_missing_requirement_ids = ["path_is_dir"]
    return state


@pytest.mark.parametrize("invalid_reference", [False, True])
def test_exhausted_answer_validation_never_publishes_uncited_prepared_sources(invalid_reference):
    task = RetrievalTask(
        route="docs", query="Python 3.9 Path.stat() accepts follow_symlinks=False?", k=2,
        requirement_id="pathstat",
        requirement={"library": "Python", "symbols": ["pathlib.Path.stat"], "version": "3.9", "match": "symbol"},
    )
    primary = _evidence(
        "Path.stat() returns an os.stat_result object.", subject="pathlib.Path.stat",
        source_uri="https://docs.python.org/3.9/library/pathlib.html#pathlib.Path.stat",
    )
    unrelated = _evidence(
        "asyncio.run() now uses shutdown_default_executor().", subject="asyncio.run",
        source_uri="https://docs.python.org/3.9/whatsnew/3.9.html#asyncio",
    )
    packet = [primary, unrelated]
    contract = RequestContract()
    document = text_document(
        "Python 3.9 Path.stat() does not accept follow_symlinks=False.", basis="source",
        refs=["missing-source" if invalid_reference else primary.id],
    )
    state = build_graph_state_input(
        user_input=task.query, request_contract=contract,
        planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task])),
        retrieval=RetrievalState(hit_log=[SearchHit(
            evidence=item, requirement_id=task.requirement_id, rank=rank,
            score=RetrievalScore(metric="relevance", raw=0.99, direction="higher"),
        ) for rank, item in enumerate(packet, 1)]),
        retry=RetryState(max_retries=0, max_content_repairs=0),
        response=ResponseState(
            result=finalize_answer(document, packet, retrieval_required=True), evidence_packet=packet,
            evidence_requirement_map={item.id: [task.requirement_id] for item in packet},
            request_id=contract.request_id, contract_revision=contract.revision,
        ),
    )

    updates = make_post_synthesis_validation_node(
        False, decisions_client=DecisionsBoundary({"requirement_0_target_analysis_supported": 0.33}),
    )(state)
    outcome = ResponseAssembler().assemble(response={**state, **updates})

    assert outcome.status == "failed"
    assert outcome.problem.code == ("model_output_invalid" if invalid_reference else "answer_validation_failed")
    assert outcome.response is None
    assert updates["response"].result.citations == []
    assert updates["response"].evidence_packet == []
    assert all(unrelated.excerpt not in message.content for message in updates["messages"])


@pytest.mark.parametrize("predicate", ["request_fulfilled", "grounded_0", "requirement_0_target_analysis_supported"])
def test_rejected_partial_answer_reports_validation_failure_and_keeps_retrieval_gap(predicate):
    state = _partial_state()
    original_body = state["response"].result.content
    decisions = DecisionsBoundary({predicate: 0.1})

    state.update(make_post_synthesis_validation_node(False, decisions_client=decisions)(state))
    outcome = ResponseAssembler().assemble(response=state)

    assert outcome.problem.code == "answer_validation_failed"
    assert outcome.problem.stage == "validation"
    assert outcome.problem.next_action == "retry_later"
    assert outcome.status == "failed"
    assert outcome.response is None
    assert state["response"].result.content != original_body
    assert state["response"].result.citations == []
    assert "retrieved_evidence_incomplete" in {issue.code for issue in state["response"].result.issues}
    assert any(question["name"] == predicate for request in decisions.requests for question in request["questions"])


def test_local_output_condition_failure_has_validation_cause_even_with_terminal_gap():
    state = _partial_state()
    contract = state["runtime"].request_contract
    state["runtime"].request_contract = contract.model_copy(update={
        "answer": AnswerContract(format=(LayoutRequirement(kind="list", mode="required"),)),
    })

    state.update(make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state))
    outcome = ResponseAssembler().assemble(response=state)

    assert outcome.problem.code == "answer_validation_failed"
    assert outcome.problem.stage == "validation"
    assert "retrieved_evidence_incomplete" in {issue.code for issue in state["response"].result.issues}


def test_accepted_partial_answer_keeps_retrieval_cause_and_body():
    state = _partial_state()
    original_body = state["response"].result.content

    state.update(make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state))
    outcome = ResponseAssembler().assemble(response=state)

    assert outcome.problem.code == "evidence_insufficient"
    assert outcome.problem.stage == "retrieval"
    assert outcome.status == "partial"
    assert state["response"].result.content == original_body


@pytest.mark.parametrize("filtered_is_valid", [False, True])
def test_filtered_partial_is_checked_as_the_exact_final_body(filtered_is_valid):
    state = _partial_state()
    response = state["response"]
    rejected_claim = "Path.exists() creates missing directories automatically."
    rejected = text_document(rejected_claim, basis="source", refs=[response.evidence_packet[0].id])
    state["response"] = response.model_copy(update={
        "result": finalize_answer(
            AnswerDocument(blocks=[*response.result.content.blocks, *rejected.blocks]),
            response.evidence_packet, retrieval_required=True,
        ),
    })

    class RevisionDecisions(DecisionsBoundary):
        def evaluate(self, **kwargs):
            checked_body = any("rendered_response" in read_decision_input(call["input"]) for call in self.requests)
            self.probabilities = ({"request_fulfilled": 1.0 if filtered_is_valid else 0.1}
                                  if checked_body else {"grounded_1": 0.1})
            return super().evaluate(**kwargs)

    decisions = RevisionDecisions()
    state.update(make_post_synthesis_validation_node(False, decisions_client=decisions)(state))
    outcome = ResponseAssembler().assemble(response=state)

    body_inputs = [read_decision_input(call["input"]) for call in decisions.requests
                   if "rendered_response" in read_decision_input(call["input"])]
    assert len(body_inputs) == 2
    assert rejected_claim in json.dumps(body_inputs[0]["rendered_response"])
    assert rejected_claim not in json.dumps(body_inputs[1]["rendered_response"])
    assert rejected_claim not in export_answer_text(state["response"].result)
    assert state["retry"].needs_retry is False
    if filtered_is_valid:
        assert outcome.status == "partial"
        assert outcome.problem.code == "evidence_insufficient"
        assert outcome.response.semantic_validation.passed
    else:
        assert outcome.status == "failed"
        assert outcome.problem.code == "answer_validation_failed"
        assert outcome.response is None
    assert any("revision=filtered" in event for event in state["debug"].validation_events)


def test_locally_rejected_draft_keeps_prepared_judgments_for_its_filtered_body():
    state = _state(repairs=0)
    source = state["response"].evidence_packet[0]
    retained = text_document("Path.exists() returns True.", basis="source", refs=[source.id])
    retained.blocks[0].content[0].sentence_scope = 1
    contract = state["runtime"].request_contract
    state["runtime"] = state["runtime"].model_copy(update={
        "request_contract": contract.model_copy(update={
            "answer": AnswerContract(sentence_counts=[SentenceCountRequirement(minimum=1, maximum=1)]),
        }),
    })
    draft = state["response"].model_copy(update={
        "result": finalize_answer(retained, [source], retrieval_required=True),
    })
    pre = DecisionsBoundary({"requirement_0_target_identified": 0.94,
                             "requirement_0_target_analysis_supported": 0.93})
    state.update(make_pre_synthesis_validation_node(
        False, decisions_client=pre, excerpt_limits=synthesis_excerpt_limits(),
    )(state))
    unsupported = text_document("Unsupported second claim.", basis="source", refs=["missing-source"])
    unsupported.blocks[0].content[0].sentence_scope = 1
    state["response"] = draft.model_copy(update={
        "result": finalize_answer(AnswerDocument(blocks=[*retained.blocks, *unsupported.blocks]),
                                  [source], retrieval_required=True),
    })
    post = DecisionsBoundary({"requirement_0_target_identified": 0.1,
                              "requirement_0_target_analysis_supported": 0.1})

    updates = make_post_synthesis_validation_node(False, decisions_client=post)(state)

    assert all("requirements" not in read_decision_input(call["input"]) for call in post.requests)
    checked = updates["response"].result
    assert updates["response"].kind == "answer"
    assert checked.content == retained
    assert [citation.evidence for citation in checked.citations] == [source]
    assert checked.semantic_validation.passed
    probabilities = {check.name: check.probability for check in checked.semantic_validation.checks}
    assert probabilities["requirement_target_identified:path_exists"] == 0.94
    assert probabilities["requirement_target_analysis_supported:path_exists"] == 0.93


@pytest.mark.parametrize("code", ["provider_unavailable", "model_output_incomplete"])
def test_existing_provider_or_generation_failure_keeps_its_cause_and_gap(code):
    state = _partial_state()
    problem = make_problem(code, "synthesis")
    state["response"].problem = problem
    decisions = DecisionsBoundary({"request_fulfilled": 0.1})

    state.update(make_post_synthesis_validation_node(False, decisions_client=decisions)(state))
    outcome = ResponseAssembler().assemble(response=state)

    assert outcome.problem == problem
    assert "retrieved_evidence_incomplete" in {issue.code for issue in outcome.response.issues}
    assert decisions.requests == []


def test_unresolved_source_reference_remains_invalid_model_output():
    state = _partial_state()
    state["response"].result = finalize_answer(
        text_document("Path.exists() returns True.", basis="source", refs=["missing-source"]),
        state["response"].evidence_packet, retrieval_required=True,
    )

    state.update(make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state))
    outcome = ResponseAssembler().assemble(response=state)

    assert outcome.problem.code == "model_output_invalid"
    assert outcome.problem.stage == "validation"
    assert outcome.status == "failed"
    assert outcome.response is None
    assert "retrieved_evidence_incomplete" in {issue.code for issue in state["response"].result.issues}
