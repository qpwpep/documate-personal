"""Published evidence scope follows the checked answer through reuse and recovery."""

import pytest

from langchain_core.messages import HumanMessage

from src.core.answer_schema import AnswerDocument, finalize_answer, text_document
from src.core.contracts import PendingAction, PlannerState, RuntimeState
from src.core.contracts.debug import RetryState
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import BoundAnswerReference, CopyAnswerBody, RequestContract, TransformAnswerBody, UnresolvedBody
from src.runtime.agent_runtime.debug_collector import DebugCollector
from src.runtime.agent_runtime.response_assembler import ResponseAssembler
from src.runtime.nodes.actions.node import make_action_postprocess_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.core.test_synthesis_validation import ModelBoundary, _hit, _state as retrieval_state


def _source_answer(*texts):
    evidence = [_hit(text).evidence for text in texts]
    document = AnswerDocument.model_validate({"blocks": [{
        "type": "paragraph", "content": [
            {"text": item.excerpt, "basis": "source", "refs": [item.id]} for item in evidence
        ],
    }]})
    return finalize_answer(document, evidence, retrieval_required=True)


def _state(contract, *, previous=None, pending=None):
    return {
        "runtime": RuntimeState(user_input="Rewrite the selected answer.", request_contract=contract,
                                previous_response=previous, pending_action=pending),
        "planner": PlannerState(output=PlannerOutput(use_retrieval=False, tasks=[])),
        "messages": [HumanMessage(content="Rewrite the selected answer.")],
        "retry": RetryState(max_retries=0),
    }


def _publish(state):
    debug = DebugCollector().build(response=state, updated_messages=state["messages"],
                                   graph_total_ms=0, upload_retriever_build_ms=None)
    return ResponseAssembler().assemble(response=state, debug_info=debug)


def test_pending_copy_retains_its_selected_source_after_actions_clear_the_pending_slot():
    """A copied pending answer publishes its captured source even after successful completion clears pending state."""
    source = _source_answer("The configured value is 3.")
    pending = PendingAction(contract=RequestContract(), response=source, body_prepared=True)
    contract = RequestContract(body=CopyAnswerBody(
        source=BoundAnswerReference(ref="pending", response_hash=source.content_hash),
    ))
    state = _state(contract, pending=pending)
    state.update(make_synthesize_node(ModelBoundary(error=AssertionError("copy must not generate")))(state))
    state.update(make_post_synthesis_validation_node(False)(state))
    state.update(make_action_postprocess_node(None, None, False)(state))

    published = _publish(state)

    assert state["runtime"].pending_action is None
    assert published["response"] == source.model_dump(mode="json")
    assert published["debug"]["observed_hits"] == []
    assert published["debug"]["tool_calls"] == []
    assert published["debug"]["answer_provenance"] == {
        "version": 1, "body_kind": "copy_answer", "response_hash": source.content_hash,
        "request_id": contract.request_id, "contract_revision": contract.revision,
        "save_operation_binding_sha256": None,
        "source": {"ref": "pending", "response_hash": source.content_hash,
                   "citation_ids": [citation.evidence.id for citation in source.citations]},
        "evidence_packet": [citation.evidence.model_dump(mode="json") for citation in source.citations],
    }


def test_partial_transform_publishes_the_checked_body_and_complete_parent_identity():
    """Removing an invalid transformed unit keeps the selected parent's full citation identity and the final checked hash."""
    source = _source_answer("The first setting is 3.", "The second setting is 5.")
    first = source.citations[0].evidence
    contract = RequestContract(body=TransformAnswerBody(
        source=BoundAnswerReference(ref="previous", response_hash=source.content_hash), instruction="Summarize it.",
    ))
    document = AnswerDocument.model_validate({"blocks": [{"type": "paragraph", "content": [
        {"text": "The first value is three.", "basis": "source", "refs": [first.id]},
        {"text": "An unsupported extra claim.", "basis": "source", "refs": ["missing"]},
    ]}]})
    state = _state(contract, previous=source)
    state.update(make_synthesize_node(ModelBoundary(malformed=document))(state))
    state.update(make_post_synthesis_validation_node(False)(state))

    published = _publish(state)

    assert state["response"].kind == "answer"
    assert len(published["response"]["citations"]) == 1
    assert "unsupported extra" not in str(published["response"])
    assert published["debug"]["answer_provenance"] == {
        "version": 1, "body_kind": "transform_answer", "response_hash": state["response"].result.content_hash,
        "request_id": contract.request_id, "contract_revision": contract.revision,
        "save_operation_binding_sha256": None,
        "source": {"ref": "previous", "response_hash": source.content_hash,
                   "citation_ids": [citation.evidence.id for citation in source.citations]},
        "evidence_packet": [citation.evidence.model_dump(mode="json") for citation in source.citations],
    }


def test_validation_fallback_publishes_only_the_replacement_packet():
    """A validation fallback keeps the captured parent identity while replacing the allowed evidence with its actual fallback packet."""
    source = _source_answer("The previous setting is 3.")
    current = _hit("The newly retrieved setting is 8.")
    contract = RequestContract(body=TransformAnswerBody(
        source=BoundAnswerReference(ref="previous", response_hash=source.content_hash), instruction="Update it.",
    ))
    state = retrieval_state([current])
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract, "previous_response": source})
    state["retry"] = RetryState(max_retries=0)
    state.update(make_synthesize_node(ModelBoundary(malformed=text_document(
        "An unsupported replacement.", basis="source", refs=["missing"],
    )))(state))
    state.update(make_post_synthesis_validation_node(False)(state))

    published = _publish(state)

    assert state["response"].kind == "failure"
    assert published["debug"]["answer_provenance"]["evidence_packet"] == [current.evidence.model_dump(mode="json")]
    assert published["debug"]["answer_provenance"]["source"] == {
        "ref": "previous", "response_hash": source.content_hash,
        "citation_ids": [citation.evidence.id for citation in source.citations],
    }
    assert published["debug"]["answer_provenance"]["response_hash"] == state["response"].result.content_hash


def test_compact_retry_publishes_the_range_that_reached_the_successful_model():
    """A successful compact attempt exposes its narrower packet instead of the failed attempt or full retrieval range."""
    hit = _hit("source detail " * 300)
    state = retrieval_state([hit])
    compact = ModelBoundary()
    state.update(make_synthesize_node(ModelBoundary(error=TimeoutError("timeout")), compact)(state))
    state.update(make_post_synthesis_validation_node(False)(state))

    published = _publish(state)

    packet = published["debug"]["answer_provenance"]["evidence_packet"]
    assert len(packet) == 1
    assert packet[0]["id"] != hit.evidence.id
    assert packet[0]["selection"] == compact.packet[0]["selection"]
    assert published["debug"]["answer_provenance"]["source"] is None
    assert published["debug"]["answer_provenance"]["body_kind"] == "compose"


def test_missing_source_cannot_publish_a_bound_parent_or_borrow_its_evidence():
    """A missing server-bound answer yields an explicit empty scope instead of invented source provenance."""
    unavailable = _source_answer("An unavailable answer.")
    contract = RequestContract(body=CopyAnswerBody(
        source=BoundAnswerReference(ref="previous", response_hash=unavailable.content_hash),
    ))
    state = _state(contract)
    state.update(make_synthesize_node(ModelBoundary(error=AssertionError("must not generate")))(state))
    state.update(make_post_synthesis_validation_node(False)(state))

    published = _publish(state)

    assert state["response"].kind == "clarification"
    assert published["debug"]["answer_provenance"] == {
        "version": 1, "body_kind": "copy_answer", "response_hash": state["response"].result.content_hash,
        "request_id": contract.request_id, "contract_revision": contract.revision,
        "save_operation_binding_sha256": None,
        "source": None, "evidence_packet": [],
    }


@pytest.mark.parametrize("terminal_path", ["unresolved_answer", "missing_upload", "exhausted_retrieval"])
def test_pre_synthesis_terminal_answers_publish_an_explicit_empty_evidence_scope(terminal_path):
    """A clarification emitted before synthesis has a known empty packet and cannot borrow an earlier answer's evidence."""
    previous = _source_answer("An earlier answer from this session.")
    contract = RequestContract(body=UnresolvedBody(question="Which answer should be saved?")) if terminal_path == "unresolved_answer" else RequestContract()
    state = _state(contract, previous=previous)
    if terminal_path == "unresolved_answer":
        state["planner"] = state["planner"].model_copy(update={"guided_followup": "Which answer should be saved?"})
    else:
        state["planner"] = PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[
            RetrievalTask(route="upload" if terminal_path == "missing_upload" else "docs", query="setting", k=1),
        ]))
        if terminal_path == "missing_upload":
            state["planner"] = state["planner"].model_copy(update={
                "guided_followup": "Attach a code file first.",
                "diagnostics": state["planner"].diagnostics.model_copy(update={"reason": "upload_retriever_missing"}),
            })

    state.update(make_pre_synthesis_validation_node(False)(state))
    published = _publish(state)

    assert state["response"].kind == "clarification"
    assert published["response"]["citations"] == []
    assert published["debug"]["answer_provenance"] == {
        "version": 1, "body_kind": contract.body.kind, "response_hash": state["response"].result.content_hash,
        "request_id": contract.request_id, "contract_revision": contract.revision,
        "save_operation_binding_sha256": None,
        "source": None, "evidence_packet": [],
    }
