from langchain_core.messages import HumanMessage

from src.core.answer_schema import ActionReceipt, export_answer_text, finalize_answer, text_document
from src.core.contracts import PlannerState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.runtime.nodes.synthesis import make_synthesize_node


class ModelBoundary:
    def __init__(self, *, unavailable=False):
        self.unavailable = unavailable

    def invoke(self, messages):
        if self.unavailable:
            raise AssertionError("This action reuses an existing document")
        return text_document("현재 전달할 답변입니다.").model_dump(mode="json")


def _state(query, **kwargs):
    return build_graph_state_input(
        user_input=query, messages=[HumanMessage(content=query)],
        planner=PlannerState(output=PlannerOutput(use_retrieval=False, tasks=[])), **kwargs,
    )


def test_save_without_previous_response_generates_current_document():
    """Saving without an earlier answer still produces a self-contained body in this turn."""
    updates = make_synthesize_node(ModelBoundary())(_state("save this answer to txt"))
    assert export_answer_text(updates["response"].result) == "현재 전달할 답변입니다."


def test_missing_slack_destination_is_a_typed_followup():
    """A missing send destination is a single interaction document rather than a delivery receipt."""
    updates = make_synthesize_node(ModelBoundary(unavailable=True))(_state("send this to slack"))
    result = updates["response"].result
    assert "channel_id" in export_answer_text(result)
    assert result.citations == []
    assert result.actions == []
    assert updates["debug"].synthesis_errors == []


def test_saving_previous_response_preserves_its_body_and_source_revision():
    """Reusing an answer preserves its cited document and drops receipts from the earlier turn."""
    snapshot = build_snapshot(
        source_uri="https://docs.example.com", title="Docs", media_type="text/plain",
        source_type="official", content="The value is 3.", parser="test", parser_version="1",
    )
    evidence = build_evidence(snapshot=snapshot, element=DocumentElement(element_id="p1", kind="paragraph", text="The value is 3."))
    previous = finalize_answer(
        text_document("The value is 3.", basis="source", refs=[evidence.id]), [evidence],
        actions=[ActionReceipt(kind="save_text", status="success", file_path="old.txt")],
    )
    updates = make_synthesize_node(ModelBoundary(unavailable=True))(_state("save this answer to txt", previous_response=previous))
    result = updates["response"].result
    assert result.content == previous.content
    assert result.citations == previous.citations
    assert result.actions == []
    assert export_answer_text(result) == export_answer_text(previous)
    assert updates["response"].evidence_packet == [evidence]
    assert updates["debug"].synthesis_errors == []


def test_retrieval_plan_is_not_short_circuited_by_save_wording():
    """A request for new source analysis does not silently reuse the previous answer."""
    state = _state("항목을 확인해서 저장해줘", previous_response=finalize_answer(text_document("previous"), []))
    state["planner"] = PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="upload", query="항목", k=4)]))
    result = make_synthesize_node(ModelBoundary())(state)["response"].result
    assert export_answer_text(result) == "현재 전달할 답변입니다."
