import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.runtime.agent_runtime.debug_collector import DebugCollector
from src.runtime.agent_runtime.response_assembler import ResponseAssembler
from src.runtime.nodes.actions import build_action_only_answer, make_action_postprocess_node, should_short_circuit_action_only

from .helpers import _ToolWrapper, build_legacy_state


class ActionsNodeTest(unittest.TestCase):
    def test_action_postprocess_save_adds_tool_message_without_touching_answer(self) -> None:
        recorded = {}

        def _save_fn(content: str, filename_prefix: str = "response"):
            recorded["content"] = content
            recorded["prefix"] = filename_prefix
            return {"message": "Saved output to response_20260101_010101.txt", "file_path": "output/save.txt"}

        action_node = make_action_postprocess_node(
            save_text_tool=_ToolWrapper(_save_fn),
            slack_notify_tool=_ToolWrapper(lambda **kwargs: {"status": "ok"}),
            verbose=False,
        )

        updates = action_node(
            build_legacy_state(
                {
                    "user_input": "save this answer to txt",
                    "final_answer": "final answer text",
                    "messages": [],
                }
            )
        )

        self.assertEqual(recorded["content"], "final answer text")
        self.assertNotIn("final_answer", updates)
        tool_messages = updates.get("messages", [])
        self.assertEqual(len(tool_messages), 2)
        self.assertIsInstance(tool_messages[0], AIMessage)
        self.assertEqual(tool_messages[1].name, "save_text")
        payload = json.loads(tool_messages[1].content)
        self.assertIn("file_path", payload)

    def test_short_circuit_only_blocks_when_slack_target_is_missing(self) -> None:
        self.assertFalse(
            should_short_circuit_action_only(
                user_input="방금 답변을 txt로 저장해줘",
                messages=[HumanMessage(content="방금 답변을 txt로 저장해줘")],
                slack_target_available=True,
            )
        )
        self.assertFalse(
            should_short_circuit_action_only(
                user_input="방금 답변을 txt로 저장해줘",
                messages=[
                    HumanMessage(content="이전 질문"),
                    AIMessage(content="이전 답변"),
                    HumanMessage(content="방금 답변을 txt로 저장해줘"),
                ],
                slack_target_available=True,
            )
        )
        self.assertTrue(
            should_short_circuit_action_only(
                user_input="send this to slack",
                messages=[HumanMessage(content="send this to slack")],
                slack_target_available=False,
            )
        )

    def test_build_action_only_answer_reuses_previous_answer_when_available(self) -> None:
        answer = build_action_only_answer(
            user_input="방금 답변을 txt로 저장해줘",
            messages=[
                HumanMessage(content="이전 질문"),
                AIMessage(content="저장할 실제 본문"),
                HumanMessage(content="방금 답변을 txt로 저장해줘"),
            ],
            slack_target_available=True,
        )

        self.assertEqual(answer, "저장할 실제 본문")

    def test_action_postprocess_rewrites_meta_answer_into_minimal_delivery_body(self) -> None:
        recorded = {}

        def _save_fn(content: str, filename_prefix: str = "response"):
            recorded["content"] = content
            recorded["prefix"] = filename_prefix
            return {"status": "ok", "file_path": "output/save_text/response.txt"}

        action_node = make_action_postprocess_node(
            save_text_tool=_ToolWrapper(_save_fn),
            slack_notify_tool=_ToolWrapper(lambda **kwargs: {"status": "ok"}),
            verbose=False,
        )

        updates = action_node(
            build_legacy_state(
                {
                    "user_input": "최종 답변을 텍스트 파일로 저장해줘",
                    "final_answer": "현재 대화에는 재사용할 이전 답변 본문이 없습니다.",
                    "messages": [],
                }
            )
        )

        self.assertIn("저장", recorded["content"])
        self.assertNotIn("현재 대화에는", recorded["content"])
        self.assertNotIn("저장용 본문", recorded["content"])
        self.assertIsInstance(updates["messages"][0], AIMessage)
        self.assertIn("저장 완료:", updates["messages"][0].content)
        self.assertEqual(updates["messages"][1].name, "save_text")

    def test_action_postprocess_reuses_previous_answer_as_delivery_body_before_appending_receipt(self) -> None:
        recorded = {}

        def _save_fn(content: str, filename_prefix: str = "response"):
            recorded["content"] = content
            recorded["prefix"] = filename_prefix
            return {"status": "ok", "file_path": "output/save_text/response.txt"}

        action_node = make_action_postprocess_node(
            save_text_tool=_ToolWrapper(_save_fn),
            slack_notify_tool=_ToolWrapper(lambda **kwargs: {"status": "ok"}),
            verbose=False,
        )

        updates = action_node(
            build_legacy_state(
                {
                    "user_input": "방금 답변을 txt로 저장해줘",
                    "final_answer": "요청하신 내용을 저장하겠습니다.",
                    "messages": [
                        HumanMessage(content="이전 질문"),
                        AIMessage(content="저장할 실제 본문"),
                        HumanMessage(content="방금 답변을 txt로 저장해줘"),
                    ],
                }
            )
        )

        self.assertEqual(recorded["content"], "저장할 실제 본문")
        self.assertIn("저장할 실제 본문", updates["messages"][0].content)
        self.assertIn("저장 완료:", updates["messages"][0].content)

    def test_response_assembler_appends_save_receipt_to_final_answer(self) -> None:
        assembler = ResponseAssembler()
        result = assembler.assemble(
            response=build_legacy_state(
                {
                    "final_answer": "공유할 본문",
                    "response_payload": {
                        "answer": "공유할 본문",
                        "claims": [],
                        "evidence": [],
                        "confidence": None,
                    },
                }
            ),
            updated_messages=[
                AIMessage(content="공유할 본문"),
                ToolMessage(
                    content=json.dumps({"status": "ok", "file_path": "output/save_text/response.txt"}),
                    name="save_text",
                    tool_call_id="tool-1",
                ),
            ],
            debug_info={},
        )

        self.assertIn("저장 완료:", result["message"])
        self.assertTrue(result["filepath"].endswith("output\\save_text\\response.txt"))
        self.assertEqual(result["response_payload"]["answer"], result["message"])

    def test_debug_collector_extracts_action_results_from_tool_messages(self) -> None:
        collector = DebugCollector()
        response = build_legacy_state(
            {
                "final_answer": "shared body",
                "response_payload": {
                    "answer": "shared body",
                    "claims": [],
                    "evidence": [],
                    "confidence": None,
                },
            }
        )
        updated_messages = [
            HumanMessage(content="send this to slack"),
            AIMessage(content="shared body"),
            ToolMessage(
                content=json.dumps({"status": "ok", "channel_id": "C123LIVE", "target_type": "Public Channel"}),
                name="slack_notify",
                tool_call_id="tool-1",
            ),
            ToolMessage(
                content=json.dumps({"status": "ok", "file_path": "output/save_text/response.txt"}),
                name="save_text",
                tool_call_id="tool-2",
            ),
        ]

        debug_info = collector.build(
            response=response,
            updated_messages=updated_messages,
            graph_total_ms=10,
            upload_retriever_build_ms=None,
        )

        self.assertIn("action_results", debug_info)
        self.assertEqual(debug_info["action_results"]["slack_notify"]["status"], "ok")
        self.assertEqual(debug_info["action_results"]["slack_notify"]["channel_id"], "C123LIVE")
        self.assertEqual(
            debug_info["action_results"]["save_text"]["file_path"],
            "output/save_text/response.txt",
        )


if __name__ == "__main__":
    unittest.main()
