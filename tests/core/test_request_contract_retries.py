import pytest
from langchain_core.messages import HumanMessage

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.contracts import GraphState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.conversation_memory import ConversationMemoryPolicy
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import WireRequestContract
from src.infra.tools.save_text import build_save_text_tool
from src.runtime.make_graph import build_graph
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.session import add_user_message, make_summarize_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node


class PlannerBoundary:
    def __init__(self, previous, save_intent):
        self.previous = previous
        self.save_intent = save_intent

    def invoke(self, messages):
        user = [message for message in messages if isinstance(message, HumanMessage)][-1]
        contract = WireRequestContract.model_validate({
            "body": {"kind": "transform_answer", "source": {"ref": "previous"},
                     "instruction": "세 줄로 줄여줘", "evidence_ids": ["format"]},
            "actions": {"save_text": {"intent": self.save_intent, "evidence_ids": ["action"]}},
            "answer": {"format": [{"kind": "line_count", "mode": "required", "value": 3, "evidence_ids": ["format"]}]},
            "evidence": [
                {"id": "format", "turn_id": user.id or "user:0", "quote": str(user.content), "scope": "answer.format.line_count", "interpretation": "instruction"},
                {"id": "action", "turn_id": user.id or "user:0", "quote": str(user.content), "scope": "actions.save_text",
                 "interpretation": "instruction" if self.save_intent == "requested" else "negation"},
            ],
        })
        return PlannerOutput(use_retrieval=False, tasks=[], request_contract=contract)


class RepairingModelBoundary:
    def __init__(self):
        self.outputs = iter(["하나\n둘\n셋\n넷", "첫 요약\n둘째 요약\n셋째 요약"])

    def invoke(self, _messages):
        return text_document(next(self.outputs)).model_dump(mode="json")


def unavailable_boundary(*_args, **_kwargs):
    raise AssertionError("This request requires neither external lookup nor Slack delivery")


@pytest.mark.parametrize("save_intent", ["requested", "forbidden"])
def test_transform_repair_preserves_contract_and_delivers_only_the_repaired_body(tmp_path, monkeypatch, save_intent):
    """검색 없는 수정도 한 번 재합성하며 저장 의사와 세 줄 조건을 바꾸지 않는다."""
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    previous = finalize_answer(text_document("이전 첫 줄\n이전 둘째 줄\n이전 셋째 줄\n이전 넷째 줄"), [])
    policy = ConversationMemoryPolicy()
    graph = build_graph(
        state_type=GraphState, add_user_node=add_user_message,
        summarize_node=make_summarize_node(None, False, policy=policy),
        planner_node=make_planner_node(PlannerBoundary(previous, save_intent), False),
        retrieve_dispatch_node=make_retrieve_dispatch_node(unavailable_boundary, unavailable_boundary, False),
        synthesize_node=make_synthesize_node(RepairingModelBoundary()),
        pre_synthesis_validation_node=make_pre_synthesis_validation_node(False),
        post_synthesis_validation_node=make_post_synthesis_validation_node(False),
        action_postprocess_node=make_action_postprocess_node(build_save_text_tool(), unavailable_boundary, False),
        memory_policy=policy,
    )
    query = "방금 답변을 세 줄로 줄여 저장해줘" if save_intent == "requested" else "방금 답변을 세 줄로 줄이고 저장하지 마"
    result = graph.invoke(build_graph_state_input(user_input=query, previous_response=previous, retry=RetryState(max_retries=2)))
    answer = result["response"].result
    expected = "첫 요약\n둘째 요약\n셋째 요약"
    assert export_answer_text(answer) == expected
    assert result["response"].kind == "answer"
    assert result["response"].synthesis_attempt == 2
    assert result["runtime"].request_contract.to_wire() == result["planner"].output.request_contract
    assert result["runtime"].request_contract.actions.save_text.intent == save_intent
    files = list(tmp_path.glob("*.txt"))
    if save_intent == "requested":
        assert len(files) == 1
        assert files[0].read_text(encoding="utf-8-sig") == expected
    else:
        assert files == []
        assert answer.actions == []
