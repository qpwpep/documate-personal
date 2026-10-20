"""Displayed partial answers remain usable without reviving their unfinished actions."""
from __future__ import annotations

import json

from src.core.answer_schema import AnswerResponse, export_answer_text
from src.core.contracts import GraphState
from src.core.contracts.debug import RetrievalDiagnostic
from src.core.contracts.retrieval import RetrievalResult
from src.core.conversation_memory import ConversationMemoryPolicy
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.infra.tools.local_rag.tool import build_upload_search_tool
from src.runtime.make_graph import build_graph
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.session import add_user_message
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.core.helpers import _CapturePlannerLLM, make_decisions_client
from tests.core.test_answer_validation import _evidence, _hit
from tests.core.test_pending_action_delivery import _manager, delivery_tools
from tests.core.test_retrieval_partial import NaturalPartialModel
from tests.synthesis_fixtures import synthesis_excerpt_limits


class _PlannerBoundary:
    def invoke(self, messages, *, timeout=None):
        context_message = next(message for message in messages if message.name == "request_context")
        context = json.loads(context_message.content.split("\n", 1)[1])
        turn = context["user_turn_ledger"][-1]
        copying = "그대로" in turn["text"]
        transforming = "요약" in turn["text"]
        acknowledging = turn["text"] == "알겠어"
        contract = {
            "body": ({"kind": "copy_answer", "source": {"ref": "previous"}} if copying else
                     {"kind": "transform_answer", "source": {"ref": "previous"}, "instruction": "부분 답변을 요약해줘"} if transforming else
                     {"kind": "acknowledge"} if acknowledging else
                     {"kind": "compose", "instruction": "Compare the docs and uploaded code"}),
            "actions": {} if acknowledging else {"save_text": {"intent": "requested", "evidence_ids": ["request"]}},
            "evidence": [{"id": "request", "turn_id": turn["turn_id"], "quote": turn["text"],
                          "scope": "current_request", "interpretation": "instruction"}],
            "slack_recipient": {"state": "omitted"},
        }
        return PlannerOutput(
            use_retrieval=not (copying or transforming or acknowledging),
            tasks=[] if copying or transforming or acknowledging else [
                RetrievalTask(route=route, query="axis", k=2, requirement_id=route)
                for route in ("docs", "upload")
            ],
            request_contract=contract,
        ).model_dump(mode="json")


def _partial_manager(save, slack, model=None):
    def docs(**kwargs):
        return RetrievalResult(
            hits=[_hit(_evidence("axis selects the dimension", source_type="official"))],
            diagnostics=RetrievalDiagnostic(status="success", answerability="covered"),
        )

    graph = build_graph(
        GraphState, add_user_message, lambda state: {},
        make_planner_node(_PlannerBoundary(), False, llm_planner_retry=_CapturePlannerLLM({
            "rewrites": [{"requirement_id": "upload", "query": "axis", "k": 2}],
        })),
        make_retrieve_dispatch_node(docs, build_upload_search_tool(), False),
        make_synthesize_node(model or NaturalPartialModel()),
        memory_policy=ConversationMemoryPolicy(),
        pre_synthesis_validation_node=make_pre_synthesis_validation_node(False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=make_decisions_client()),
        post_synthesis_validation_node=make_post_synthesis_validation_node(False, decisions_client=make_decisions_client()),
        action_postprocess_node=make_action_postprocess_node(save, slack, False),
    )
    return _manager(graph)


def test_displayed_partial_can_be_saved_on_a_new_explicit_request(delivery_tools):
    save, slack, deliveries, output = delivery_tools
    manager = _partial_manager(save, slack)

    first = manager.run_agent_flow("공식 문서와 첨부 코드를 비교하고 저장해줘")
    assert first["status"] == "partial"
    partial = AnswerResponse.model_validate(first["response"])
    assert not output.exists()

    followup = manager.run_agent_flow("방금 답변을 그대로 저장해줘")

    assert followup["status"] == "completed"
    copied = AnswerResponse.model_validate(followup["response"])
    assert copied.content == partial.content
    assert "answer_incomplete" in {issue.code for issue in copied.issues}
    assert copied.actions[0].status == "success"
    saved_files = list(output.glob("*.txt"))
    assert len(saved_files) == 1
    assert saved_files[0].read_text(encoding="utf-8-sig") == export_answer_text(copied, include_sources=True)
    assert deliveries == []


def test_partial_answer_summary_receives_and_preserves_original_limits(delivery_tools):
    class SummaryModel(NaturalPartialModel):
        def invoke(self, messages, *, timeout=None):
            source_message = next((message for message in messages
                                   if str(message.content).startswith("[Bound Source Answer]")), None)
            if source_message is None:
                return super().invoke(messages, timeout=timeout)
            self.source = json.loads(str(source_message.content).split("\n", 2)[-1])
            packet_message = next(str(message.content) for message in messages
                                  if str(message.content).startswith("[Evidence Packet]"))
            packet = json.loads(packet_message[packet_message.index("[", len("[Evidence Packet]")):])
            return {"blocks": [{"type": "paragraph", "content": [
                {"text": "The docs explain axis selection.", "basis": "source", "refs": [packet[0]["id"]]},
                {"text": " ".join(issue["message"] for issue in self.source["issues"]),
                 "basis": "interaction", "refs": []},
            ]}]}

    save, slack, deliveries, output = delivery_tools
    model = SummaryModel()
    manager = _partial_manager(save, slack, model)
    first = manager.run_agent_flow("공식 문서와 첨부 코드를 비교하고 저장해줘")
    assert first["status"] == "partial"
    partial = AnswerResponse.model_validate(first["response"])

    followup = manager.run_agent_flow("방금 답변을 요약해서 저장해줘")

    assert followup["status"] == "completed"
    summary = AnswerResponse.model_validate(followup["response"])
    assert model.source["issues"] == [issue.model_dump(mode="json") for issue in partial.issues]
    assert model.source["semantic_validation"] == partial.semantic_validation.model_dump(mode="json")
    assert all(issue.message in export_answer_text(summary) for issue in partial.issues)
    assert summary.actions[0].status == "success"
    assert len(list(output.glob("*.txt"))) == 1
    assert deliveries == []


def test_partial_history_keeps_its_limits_without_automatically_saving(delivery_tools):
    save, slack, deliveries, output = delivery_tools
    manager = _partial_manager(save, slack)

    result = manager.run_agent_flow("공식 문서와 첨부 코드를 비교하고 저장해줘")
    assert result["status"] == "partial"
    partial = AnswerResponse.model_validate(result["response"])
    assert export_answer_text(partial) in str(manager.messages[-1].content)
    assert result["message"] in str(manager.messages[-1].content)
    assert manager._ensure_session().previous_response == partial
    assert manager._ensure_session().pending_action is not None
    assert not manager._ensure_session().pending_action.body_prepared

    acknowledgement = manager.run_agent_flow("알겠어")

    assert acknowledgement["status"] == "completed"
    assert manager._ensure_session().previous_response == partial
    assert not manager._ensure_session().pending_action.body_prepared
    assert not output.exists()
    assert deliveries == []
