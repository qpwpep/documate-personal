import json

import pytest


from langchain_core.messages import AIMessage, HumanMessage

from src.core.answer_schema import export_answer_text, iter_content_units, text_document
from src.core.contracts import PlannerState, RetrievalState, DebugState, RetrievalDiagnostic
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import ExtractBody, RequestContract
from src.core.llm_errors import LLMCallError
from src.runtime.nodes.synthesis import make_synthesize_node
from tests.synthesis_fixtures import prepare_synthesis_state


def _hit(text="Default mode is safe.", *, source="official", rank=1):
    snapshot = build_snapshot(
        source_uri="https://docs.example.com/settings" if source == "official" else "uploads/example.py",
        title="Settings", media_type="text/plain", source_type=source,
        content=text, parser="test", parser_version="1",
    )
    element = DocumentElement(
        element_id="settings", kind="paragraph" if source == "official" else "code", text=text,
        anchors=[SourceAnchor(kind="web" if source == "official" else "code", start=0, end=len(text), precision="exact")],
    )
    return SearchHit(evidence=build_evidence(snapshot=snapshot, element=element), rank=rank,
                     score=RetrievalScore(metric="test", raw=0.01, normalized=0.01, direction="higher"))


def _state(hits, query="Explain the setting"):
    routes = list(dict.fromkeys(hit.evidence.route for hit in hits))
    tasks = [RetrievalTask(route=route, query=query, k=4) for route in routes]
    task_ids = {task.route: task.requirement_id for task in tasks}
    hits = [hit if hit.requirement_id else hit.model_copy(update={"requirement_id": task_ids[hit.evidence.route]}) for hit in hits]
    return build_graph_state_input(
        user_input=query, messages=[HumanMessage(content=query)],
        request_contract=RequestContract(),
        planner=PlannerState(output=PlannerOutput(use_retrieval=bool(routes), tasks=tasks)),
        retrieval=RetrievalState(hit_log=[hit.model_dump(mode="json") for hit in hits]),
    )


class ModelBoundary:
    def __init__(self, *, error=None, malformed=None):
        self.error = error
        self.malformed = malformed
        self.packet = []

    def with_structured_output(self, *args, **kwargs):
        return self

    def invoke(self, messages, *, timeout=None):
        packet_text = next(str(message.content) for message in messages if str(message.content).startswith("[Evidence Packet]"))
        self.packet = json.loads(packet_text[packet_text.index("[", len("[Evidence Packet]")):])
        if self.error:
            raise self.error
        if self.malformed is not None:
            return self.malformed
        if self.packet:
            document = text_document(self.packet[0]["excerpt"], basis="excerpt", refs=[self.packet[0]["id"]])
        else:
            document = text_document("현재 답변입니다.")
        return {"parsed": document.model_dump(mode="json"), "raw": AIMessage(content=""), "parsing_error": None}


def test_malformed_generation_becomes_explicit_original_source_fallback():
    """Invalid model text is never promoted to an answer, while original evidence stays available."""
    hit = _hit()
    updates = make_synthesize_node(ModelBoundary(malformed={"answer": "UNVALIDATED WRONG ANSWER"}))(prepare_synthesis_state(_state([hit])))
    result = updates["response"].result
    assert "UNVALIDATED WRONG ANSWER" not in export_answer_text(result)
    assert hit.evidence.excerpt in export_answer_text(result)
    assert any(unit.basis == "excerpt" for _, unit in iter_content_units(result.content))
    assert updates["debug"].synthesis_errors
    assert all(check.support_status != "unsupported" for check in result.checks)


def test_exact_upload_extraction_needs_no_model_generation():
    """Explicit source extraction uses the same checked document contract without rewriting code."""
    hit = _hit("retries = 3\n", source="upload")
    state = _state([hit], query="retries 코드를 원문 그대로 발췌해줘")
    state["runtime"] = state["runtime"].model_copy(update={
        "request_contract": RequestContract(body=ExtractBody()),
    })
    task = state["planner"].output.tasks[0]
    state["retrieval"] = RetrievalState(hit_log=[hit.model_copy(update={"requirement_id": task.requirement_id}).model_dump(mode="json")])
    state["debug"] = DebugState(retrieval_diagnostics=[RetrievalDiagnostic(
        route="upload", requirement_id=task.requirement_id, status="success", answerability="covered", evidence_count=1,
    )])
    prepare_synthesis_state(state)
    updates = make_synthesize_node(ModelBoundary(error=AssertionError("model should not run")))(state)
    assert updates["debug"].synthesis_errors == []
    assert any(check.support_status == "exact_match" for check in updates["response"].result.checks)
    assert updates["response"].result.citations[0].evidence.excerpt == hit.evidence.excerpt
    assert updates["response"].normal_evidence_missing_requirement_ids == []
    assert updates["response"].result.retrieval_required
    assert updates["response"].result.issues == []


def test_rejected_exact_extraction_uses_repair_feedback_instead_of_repeating_the_same_shortcut():
    """After semantic rejection, the normal recovery path can repair the displayed document."""
    from src.core.contracts import RetryState

    hit = _hit("retries = 3\n", source="upload")
    state = _state([hit], query="retries 코드를 원문 그대로 발췌해줘")
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": RequestContract(body=ExtractBody())})
    task = state["planner"].output.tasks[0]
    state["debug"] = DebugState(retrieval_diagnostics=[RetrievalDiagnostic(
        route="upload", requirement_id=task.requirement_id, status="success", answerability="covered", evidence_count=1,
    )])
    node = make_synthesize_node(ModelBoundary())
    prepare_synthesis_state(state)
    state.update(node(state))
    assert state["response"].result.content.blocks[0].type == "code"
    state["retry"] = RetryState(
        needs_retry=True, retry_scope="reuse_hits_resynthesize", retry_reason="missing_content",
        content_repair_attempt=1, retrieval_feedback="Repair the requested display without changing the selected original.",
    )

    updates = node(state)

    # The HTTP/model boundary returns a repaired presentation of the same original.
    assert updates["response"].result.content.blocks[0].type == "paragraph"
    assert updates["response"].result.content.blocks[0].content[0].text == hit.evidence.excerpt


def test_unprepared_generation_fails_closed_before_model_call():
    model = ModelBoundary(error=AssertionError("an unprepared request cannot call the model"))

    with pytest.raises(LLMCallError) as caught:
        make_synthesize_node(model)(_state([_hit()]))

    assert caught.value.problem.code == "internal_error"
    assert model.packet == []


def test_prepared_source_ranges_are_used_without_reselecting_changed_raw_hits():
    state = prepare_synthesis_state(_state([_hit("The approved source range.")]))
    approved = state["retrieval"].prepared_evidence.normal.evidence
    extra = _hit("An unvalidated replacement must never enter generation.")
    state["retrieval"] = state["retrieval"].model_copy(update={"hit_log": [extra]})
    model = ModelBoundary()

    response = make_synthesize_node(model)(state)["response"]

    assert response.evidence_packet == approved
    assert [item["id"] for item in model.packet] == [f"e{index}" for index in range(1, len(approved) + 1)]
    assert [item["snapshot_id"] for item in model.packet] == [item.snapshot.snapshot_id for item in approved]
    assert [item["selection"] for item in model.packet] == [item.selection.model_dump(mode="json") for item in approved]
    assert [item["excerpt"] for item in model.packet] == [item.excerpt for item in approved]
    assert [citation.evidence.selection for citation in response.result.citations] == [item.selection for item in approved]


@pytest.mark.parametrize("changed", ["revision", "plan"])
def test_stale_preparation_fails_closed(changed):
    state = prepare_synthesis_state(_state([_hit()]))
    if changed == "revision":
        contract = state["runtime"].request_contract
        state["runtime"] = state["runtime"].model_copy(update={
            "request_contract": contract.model_copy(update={"revision": contract.revision + 1}),
        })
    else:
        planner = state["planner"]
        task = planner.output.tasks[0].model_copy(update={"query": "A different request"})
        state["planner"] = planner.model_copy(update={"output": planner.output.model_copy(update={"tasks": [task]})})
    model = ModelBoundary(error=AssertionError("stale preparation cannot call the model"))

    with pytest.raises(LLMCallError) as caught:
        make_synthesize_node(model)(state)

    assert caught.value.problem.code == "internal_error"
    assert model.packet == []
