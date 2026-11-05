"""Validation observations preserve rejected inputs without changing execution policy."""
from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from src.core.answer_schema import AnswerDocument, finalize_answer, text_document
from src.core.contracts import RetrievalState
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.llm_errors import LLMCallError
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_bodies import BoundInputText, TransformInputBody
from src.core.request_contracts import RequestContract
from src.runtime.agent_runtime.validation_diagnostics import capture_validation_diagnostics
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.core.test_bound_transform_validation_scope import _bound_edit_state
from tests.core.test_decisions_validation import DecisionsBoundary, _evidence, _previous_answer_state, _state
from tests.synthesis_fixtures import synthesis_excerpt_limits


def _distinct_packet_state():
    state = _state()
    hit = state["retrieval"].hit_log[0]
    state["retrieval"] = RetrievalState(hit_log=[
        hit.model_copy(update={"rank": index + 1, "evidence": _evidence(
            f"Path.exists documentation passage {index}. "
            + "Path.exists returns False for a missing path. " * 15,
        )}) for index in range(8)
    ])
    return state


def _pre_node(boundary):
    return make_pre_synthesis_validation_node(
        False, decisions_client=boundary, excerpt_limits=synthesis_excerpt_limits(),
    )


def test_preparation_retains_rejected_compact_packet_and_requirement_judgment():
    state = _distinct_packet_state()
    boundary = DecisionsBoundary({"compact_requirement_0_target_analysis_supported": 0.2})

    with capture_validation_diagnostics() as recorder:
        updates = _pre_node(boundary)(state)
    diagnostics = recorder.snapshot()

    assert updates["retrieval"].prepared_evidence.compact is None
    observation, = diagnostics.observations
    assert observation.revision == "preparation"
    assert observation.status == "completed"
    assert observation.user_input == state["runtime"].user_input
    assert observation.request_contract == state["runtime"].request_contract
    assert observation.tasks == state["planner"].output.tasks
    assert observation.candidates
    packets = {packet.mode: packet for packet in observation.packets}
    assert set(packets) == {"normal", "compact"}
    assert packets["normal"].evidence_ids != packets["compact"].evidence_ids
    assert packets["compact"].failed_requirement_ids == ["path_exists"]
    assert packets["normal"].failed_requirement_ids == []
    assert [diagnostics.evidence[identity] for identity in packets["normal"].evidence_ids] == (
        updates["retrieval"].prepared_evidence.normal.evidence
    )
    rejected, = [predicate for predicate in observation.predicates
                 if predicate.name == "compact_requirement_0_target_analysis_supported"]
    assert rejected.requirement_id == "path_exists"
    assert rejected.packet_mode == "compact"
    assert rejected.criterion == "target_analysis_supported"
    assert rejected.status == "evaluated"
    assert rejected.probability == 0.2


def test_preparation_preserves_original_normal_packet_when_compact_is_promoted():
    state = _distinct_packet_state()
    boundary = DecisionsBoundary({"requirement_0_target_analysis_supported": 0.2})

    with capture_validation_diagnostics() as recorder:
        updates = _pre_node(boundary)(state)
    diagnostics = recorder.snapshot()

    prepared = updates["retrieval"].prepared_evidence
    assert prepared.normal == prepared.compact
    observation, = diagnostics.observations
    packets = {packet.mode: packet for packet in observation.packets}
    assert packets["normal"].failed_requirement_ids == ["path_exists"]
    assert packets["compact"].failed_requirement_ids == []
    assert packets["normal"].evidence_ids != packets["compact"].evidence_ids
    assert packets["compact"].evidence_ids == [item.id for item in prepared.normal.evidence]


def test_post_validation_preserves_generated_and_filtered_revisions_with_refusal():
    state = _state(repairs=0)
    response = state["response"]
    rejected_text = "Path.exists() creates missing directories automatically."
    document = AnswerDocument(blocks=[
        *response.result.content.blocks,
        *text_document(rejected_text, basis="source", refs=[response.evidence_packet[0].id]).blocks,
    ])
    generated = finalize_answer(document, response.evidence_packet, retrieval_required=True)
    state["response"] = response.model_copy(update={"result": generated})
    boundary = DecisionsBoundary(refused_names={"grounded_1"})

    with capture_validation_diagnostics() as recorder:
        updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)
    diagnostics = recorder.snapshot()

    assert updates["response"].kind == "answer"
    assert updates["response"].result.content == response.result.content
    assert updates["response"].result.semantic_validation.passed
    original, filtered = diagnostics.observations
    assert [original.revision, filtered.revision] == ["generated", "filtered"]
    assert original.body == generated.content
    assert original.body_hash == generated.content_hash
    assert filtered.body == response.result.content
    assert filtered.body_hash == response.result.content_hash
    assert original.body_hash != filtered.body_hash
    refused, = [predicate for predicate in original.predicates if predicate.name == "grounded_1"]
    assert refused.unit_id == "b1.content.0"
    assert refused.status == "refused"
    assert refused.probability is None
    assert [diagnostics.evidence[identity] for identity in original.packets[0].evidence_ids] == response.evidence_packet
    assert original.status == filtered.status == "completed"


def test_post_provider_failure_retains_body_packet_and_completed_source_judgment():
    class BodyFailureBoundary(DecisionsBoundary):
        def _respond(self, request):
            payload = json.loads(request.content)
            if any(question["name"] == "request_fulfilled" for question in payload["questions"]):
                self.requests.append({**payload, "stage": self._stage, "attempt": self._attempt})
                return httpx.Response(400, json={"error": {
                    "message": "Invalid provider request", "type": "invalid_request_error",
                    "code": "invalid_request_error",
                }})
            return super()._respond(request)

    state = _state()
    original = state["response"].result
    boundary = BodyFailureBoundary()

    with capture_validation_diagnostics() as recorder:
        with pytest.raises(LLMCallError):
            make_post_synthesis_validation_node(False, decisions_client=boundary)(state)
    diagnostics = recorder.snapshot()

    observation, = diagnostics.observations
    assert observation.status == "failed"
    assert observation.failure_type == "LLMCallError"
    assert observation.body == original.content
    assert observation.body_hash == original.content_hash
    assert [diagnostics.evidence[identity] for identity in observation.packets[0].evidence_ids] == state["response"].evidence_packet
    completed, = [predicate for predicate in observation.predicates if predicate.name == "grounded_0"]
    assert completed.status == "evaluated"
    assert completed.probability == 0.99
    assert completed.unit_id == "b0.content.0"
    assert state["response"].result == original


def test_recording_keeps_product_outcome_and_provider_requests_unchanged():
    plain_state, captured_state = _state(), _state()
    plain_boundary = DecisionsBoundary({"request_fulfilled": 0.2})
    captured_boundary = DecisionsBoundary({"request_fulfilled": 0.2})
    plain = make_post_synthesis_validation_node(False, decisions_client=plain_boundary)(plain_state)

    with capture_validation_diagnostics() as recorder:
        captured = make_post_synthesis_validation_node(False, decisions_client=captured_boundary)(captured_state)

    assert captured == plain
    assert captured_boundary.requests == plain_boundary.requests
    assert recorder.snapshot().observations


def test_repeated_preparation_records_cache_results_and_execution_scopes_are_isolated():
    state, boundary = _state(), DecisionsBoundary()
    node = _pre_node(boundary)

    with capture_validation_diagnostics() as recorder:
        state.update(node(state))
        request_count = len(boundary.requests)
        state.update(node(state))
    first = recorder.snapshot()

    assert len(boundary.requests) == request_count
    assert len(first.observations) == 2
    assert all(predicate.cached for predicate in first.observations[1].predicates)
    assert {predicate.status for predicate in first.observations[1].predicates} == {"evaluated"}
    with capture_validation_diagnostics() as another:
        assert another.snapshot().observations == []
    assert another.snapshot().evidence == {}
    assert len(recorder.snapshot().observations) == 2


def test_acknowledgment_records_skip_without_model_judgments():
    state, boundary = _state(retrieval=False), DecisionsBoundary()
    contract = RequestContract(request_id="acknowledge", body={"kind": "acknowledge"})
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract})

    with capture_validation_diagnostics() as recorder:
        updates = _pre_node(boundary)(state)

    observation, = recorder.snapshot().observations
    assert observation.status == "skipped"
    assert observation.skip_reason
    assert observation.predicates == []
    assert boundary.requests == []
    assert updates["retry"].needs_retry is False


def test_local_contract_failure_keeps_rejected_body_without_semantic_probabilities():
    state, boundary = _state(), DecisionsBoundary()
    contract = RequestContract(
        request_id=state["runtime"].request_contract.request_id,
        answer={"sentence_counts": [{"minimum": 2, "maximum": 2, "evidence_ids": ["two-sentences"]}]},
        evidence=[{"id": "two-sentences", "turn_id": "user-request", "quote": "두 문장",
                   "scope": "answer.sentence_counts", "interpretation": "instruction"}],
    )
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract})
    original = state["response"].result

    with capture_validation_diagnostics() as recorder:
        updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    observation, = recorder.snapshot().observations
    assert observation.body == original.content
    assert any(failure.startswith("sentence_count:") for failure in observation.local_failures)
    assert observation.predicates == []
    assert boundary.requests == []
    assert updates["retry"].needs_retry is True


def test_candidate_records_exact_complementary_context_given_to_its_predicate():
    source_text = 'import pathlib\nexists = pathlib.Path("report.txt").exists()\n'
    source = build_snapshot(
        source_uri="upload:///context.py", title="context.py", media_type="text/x-python",
        source_type="upload", content=source_text, parser="python-ast", parser_version="1",
    )
    element = DocumentElement(element_id="context", kind="code", language="python", text=source_text)
    import_end = len("import pathlib\n")
    imported = build_evidence(snapshot=source, element=element, start=0, end=import_end)
    invocation = build_evidence(snapshot=source, element=element, start=import_end, end=len(source_text))
    task = RetrievalTask(route="upload", requirement_id="path_check", query="Explain the exists assignment.",
                         k=2, requirement={"match": "topic"})
    state, boundary = _state(), DecisionsBoundary()
    state["planner"] = state["planner"].model_copy(update={
        "output": PlannerOutput(use_retrieval=True, tasks=[task]),
    })
    state["retrieval"] = RetrievalState(hit_log=[SearchHit(
        evidence=evidence, requirement_id=task.requirement_id, rank=index,
        score=RetrievalScore(metric="test", normalized=0.99, direction="higher"),
    ) for index, evidence in enumerate([imported, invocation], 1)])

    with capture_validation_diagnostics() as recorder:
        _pre_node(boundary)(state)

    source_call = next(json.loads(call["input"]) for call in boundary.requests
                       if "evidence" in json.loads(call["input"]))
    actual = next(entry for entry in source_call["evidence"] if entry["source_id"] == invocation.id)
    assert actual["context_source_ids"] == [imported.id]
    diagnostics = recorder.snapshot()
    observation, = diagnostics.observations
    candidate = next(candidate for candidate in observation.candidates if candidate.evidence_id == invocation.id)
    assert candidate.context_evidence_ids == [imported.id]
    assert diagnostics.evidence[candidate.context_evidence_ids[0]].excerpt == "import pathlib\n"


def test_observation_distinguishes_search_rewrite_from_original_semantic_task():
    state, boundary = _state(), DecisionsBoundary()
    original = state["planner"].output.tasks[0]
    rewritten = original.model_copy(update={"query": "exists official Python documentation return value", "k": 6})
    state["retry"] = state["retry"].model_copy(update={"original_tasks": [original]})
    state["planner"] = state["planner"].model_copy(update={
        "output": PlannerOutput(use_retrieval=True, tasks=[rewritten]),
    })

    with capture_validation_diagnostics() as recorder:
        _pre_node(boundary)(state)

    observation, = recorder.snapshot().observations
    assert observation.tasks[0].query == rewritten.query
    assert observation.semantic_tasks[0].query == original.query
    assert observation.semantic_tasks[0].requirement_id == original.requirement_id
    for call in boundary.requests:
        payload = json.loads(call["input"])
        assert payload["tasks"][original.requirement_id]["query"] == observation.semantic_tasks[0].query


def test_later_http_batch_failure_preserves_completed_predicates_in_the_same_evaluation():
    class SecondBatchFailureBoundary(DecisionsBoundary):
        max_questions_per_call = 1

        def _respond(self, request):
            if self.requests:
                self.error_status = 400
            return super()._respond(request)

    state, boundary = _state(), SecondBatchFailureBoundary()
    original = state["planner"].output.tasks[0]
    task = original.model_copy(update={
        "requirement": original.requirement.model_copy(update={"evidence_needs": ["explanation"]}),
    })
    state["planner"] = state["planner"].model_copy(update={
        "output": PlannerOutput(use_retrieval=True, tasks=[task]),
    })
    # The second retrieval attempt reserves packet checks but no further retry.
    state["retry"] = state["retry"].model_copy(update={"attempt": 1})

    with capture_validation_diagnostics() as recorder:
        with pytest.raises(LLMCallError):
            _pre_node(boundary)(state)

    assert len(boundary.requests) == 2
    assert [request["questions"][0]["name"] for request in boundary.requests] == [
        "evidence_0", "evidence_0_support_1",
    ]
    observation, = recorder.snapshot().observations
    assert observation.status == "failed"
    completed, pending = observation.predicates
    assert completed.name == "evidence_0"
    assert completed.status == "evaluated"
    assert completed.probability == 0.99
    assert pending.name == "evidence_0_support_1"
    assert pending.status == "pending"
    assert pending.probability is None


def test_source_selection_keeps_raw_candidates_that_do_not_fit_the_bounded_pool():
    state, boundary = _state(), DecisionsBoundary()
    original_hit = state["retrieval"].hit_log[0]
    hits = [original_hit.model_copy(update={
        "rank": index + 1,
        "evidence": _evidence(f"Path.exists returns False for missing path {index}."),
    }) for index in range(13)]
    state["retrieval"] = RetrievalState(hit_log=hits)

    with capture_validation_diagnostics() as recorder:
        _pre_node(boundary)(state)

    source_call = next(json.loads(call["input"]) for call in boundary.requests
                       if "evidence" in json.loads(call["input"]))
    evaluated_ids = {entry["source_id"] for entry in source_call["evidence"]}
    raw_ids = {hit.evidence.id for hit in hits}
    assert evaluated_ids < raw_ids
    diagnostics = recorder.snapshot()
    observation, = diagnostics.observations
    recorded_ids = {candidate.evidence_id for candidate in observation.candidates}
    assert raw_ids <= recorded_ids
    assert all(diagnostics.evidence[identity] for identity in raw_ids)
    keys = [(candidate.requirement_id, candidate.evidence_id, candidate.role)
            for candidate in observation.candidates]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("refused", [False, True], ids=["rejected", "refused"])
def test_cached_source_exclusion_preserves_candidate_and_original_judgment(refused):
    state = _state()
    source = state["retrieval"].hit_log[0].evidence
    boundary = DecisionsBoundary({"evidence_0": 0.1}, refused_names={"evidence_0"} if refused else set())
    node = _pre_node(boundary)

    with capture_validation_diagnostics() as recorder:
        state.update(node(state))
        first_request_count = len(boundary.requests)
        state.update(node(state))

    assert len(boundary.requests) == first_request_count
    diagnostics = recorder.snapshot()
    _, repeated = diagnostics.observations
    candidate, = [candidate for candidate in repeated.candidates if candidate.evidence_id == source.id]
    assert candidate.requirement_id == "path_exists"
    assert diagnostics.evidence[candidate.evidence_id] == source
    excluded, = [predicate for predicate in repeated.predicates
                  if predicate.evidence_id == source.id and predicate.criterion == "source_relevance"]
    assert excluded.cached is True
    assert excluded.status == ("refused" if refused else "evaluated")
    assert excluded.probability == (None if refused else 0.1)


@pytest.mark.parametrize("source_kind", ["answer", "input"])
def test_unchanged_required_transform_records_local_failure_without_provider_calls(source_kind):
    state = _bound_edit_state()
    original = state["runtime"].previous_response
    if source_kind == "input":
        text = "마지막 줄: 42"
        contract = state["runtime"].request_contract
        state["runtime"] = state["runtime"].model_copy(update={
            "request_contract": contract.model_copy(update={"body": TransformInputBody(
                source=BoundInputText(turn_id="current", text=text, start=0, end=len(text),
                                     content_hash=hashlib.sha256(text.encode()).hexdigest()),
                instruction=contract.body.instruction, change_requirement="required",
            )}),
        })
    state["response"] = state["response"].model_copy(update={"result": original})
    plain_boundary, recorded_boundary = DecisionsBoundary(), DecisionsBoundary()
    expected = make_post_synthesis_validation_node(False, decisions_client=plain_boundary)(state)

    with capture_validation_diagnostics() as recorder:
        updates = make_post_synthesis_validation_node(False, decisions_client=recorded_boundary)(state)

    assert updates == expected
    assert updates["retry"].needs_retry is True
    assert updates["retry"].retry_reason == "missing_content"
    assert plain_boundary.requests == recorded_boundary.requests == []
    observation, = recorder.snapshot().observations
    assert observation.body == original.content
    assert observation.skip_reason == "unchanged_transformation"
    assert observation.predicates == []
    assert "transform_unchanged" in observation.local_failures


def test_exact_answer_copy_records_deterministic_skip_without_provider_calls():
    state, source = _previous_answer_state("copy_answer")
    boundary = DecisionsBoundary()

    with capture_validation_diagnostics() as recorder:
        updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert updates["response"].kind == "answer"
    assert updates["response"].result == source
    assert boundary.requests == []
    observation, = recorder.snapshot().observations
    assert observation.body == source.content
    assert observation.predicates == []
    assert observation.local_failures == []
    assert observation.status == "skipped"
    assert observation.skip_reason == "deterministic_exact_copy"
