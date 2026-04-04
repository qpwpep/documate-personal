import json
import unittest

from langchain_core.messages import AIMessage, ToolMessage

from src.agent_runtime.response_assembler import ResponseAssembler
from src.nodes.actions import make_action_postprocess_node

from .helpers import _ToolWrapper, build_legacy_state


class ActionsNodeTest(unittest.TestCase):
    def test_action_postprocess_save_adds_tool_message_without_touching_answer(self) -> None:
        def _save_fn(content: str, filename_prefix: str = "response"):
            _ = (content, filename_prefix)
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

        self.assertNotIn("final_answer", updates)
        tool_messages = updates.get("messages", [])
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].name, "save_text")
        payload = json.loads(tool_messages[0].content)
        self.assertIn("file_path", payload)

    def test_action_postprocess_slack_skips_without_destination(self) -> None:
        calls = {"count": 0}

        def _slack_fn(**kwargs):
            calls["count"] += 1
            return kwargs

        action_node = make_action_postprocess_node(
            save_text_tool=_ToolWrapper(lambda **kwargs: {"status": "ok"}),
            slack_notify_tool=_ToolWrapper(_slack_fn),
            verbose=False,
        )

        updates = action_node(
            build_legacy_state(
                {
                    "user_input": "send this to slack",
                    "final_answer": "final answer text",
                    "messages": [],
                    "session_metadata": {"slack_destination": None},
                }
            )
        )

        self.assertEqual(calls["count"], 0)
        self.assertEqual(updates, {})

    def test_action_postprocess_slack_uses_explicit_destination(self) -> None:
        recorded = {}

        def _slack_fn(**kwargs):
            recorded.update(kwargs)
            return {"status": "ok", "channel_id": kwargs.get("channel_id")}

        action_node = make_action_postprocess_node(
            save_text_tool=_ToolWrapper(lambda **kwargs: {"status": "ok"}),
            slack_notify_tool=_ToolWrapper(_slack_fn),
            verbose=False,
        )

        updates = action_node(
            build_legacy_state(
                {
                    "user_input": "send this to slack",
                    "final_answer": "final answer text",
                    "messages": [],
                    "session_metadata": {
                        "slack_destination": {
                            "channel_id": "C123BENCH",
                            "user_id": None,
                            "email": None,
                        }
                    },
                }
            )
        )

        self.assertEqual(recorded["channel_id"], "C123BENCH")
        tool_messages = updates.get("messages", [])
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].name, "slack_notify")

    def test_action_postprocess_rewrites_save_followup_into_delivery_body(self) -> None:
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
                    "user_input": "최종 답변을 텍스트 파일로 저장해줘.",
                    "final_answer": (
                        "지금은 저장할 “최종 답변” 내용이 제공되지 않았습니다. "
                        "저장할 내용을 알려주시면 그 내용을 그대로 .txt 파일로 저장해 드리겠습니다."
                    ),
                    "messages": [],
                }
            )
        )

        self.assertIn("저장 내용", recorded["content"])
        self.assertIn("이 메시지 자체를 이번 요청의 최종 전달본으로 사용합니다.", recorded["content"])
        self.assertNotIn("알려주시면", recorded["content"])
        self.assertIsInstance(updates["messages"][0], AIMessage)
        self.assertEqual(updates["messages"][1].name, "save_text")

    def test_action_postprocess_rewrites_meta_only_slack_checklist(self) -> None:
        recorded = {}

        def _slack_fn(**kwargs):
            recorded.update(kwargs)
            return {"status": "ok"}

        action_node = make_action_postprocess_node(
            save_text_tool=_ToolWrapper(lambda **kwargs: {"status": "ok"}),
            slack_notify_tool=_ToolWrapper(_slack_fn),
            verbose=False,
        )

        updates = action_node(
            build_legacy_state(
                {
                    "user_input": (
                        "이 답변을 슬랙으로 보내줘.\n"
                        "추가로 최종 답변은 저장/공유 요청 조건을 충족해줘. 요약 + 체크리스트 형태로 답변해줘."
                    ),
                    "final_answer": (
                        "요약\n"
                        "- 슬랙으로 보낼 최종 답변을 요약 + 체크리스트 형태로 작성했고, 저장/공유 요청 조건을 충족하도록 "
                        "“바로 붙여넣을 수 있는 최종 메시지 본문”을 제공했습니다.\n\n"
                        "체크리스트\n"
                        "- [ ] 슬랙 전송용: 요약 + 체크리스트 형식으로 작성됨\n"
                        "- [ ] 최종 답변: 저장/공유 요청 조건을 충족하기 위해 ‘메시지 본문’ 형태로 제공됨"
                    ),
                    "messages": [],
                    "session_metadata": {
                        "slack_destination": {
                            "channel_id": "C123BENCH",
                            "user_id": None,
                            "email": None,
                        }
                    },
                }
            )
        )

        self.assertIn("공유 내용", recorded["text"])
        self.assertIn("이 메시지를 그대로 Slack으로 전달합니다.", recorded["text"])
        self.assertNotIn("저장/공유 요청 조건", recorded["text"])
        self.assertIsInstance(updates["messages"][0], AIMessage)
        self.assertIn(
            "slack_notify",
            [message.name for message in updates["messages"] if isinstance(message, ToolMessage)],
        )

    def test_response_assembler_prefers_latest_ai_override_and_tool_path(self) -> None:
        assembler = ResponseAssembler()
        result = assembler.assemble(
            response=build_legacy_state(
                {
                    "final_answer": "질문형 초안",
                    "response_payload": {
                        "answer": "질문형 초안",
                        "claims": [],
                        "evidence": [],
                        "confidence": None,
                    },
                }
            ),
            updated_messages=[
                AIMessage(content="질문형 초안"),
                AIMessage(content="공유 내용\n- 이 메시지 자체를 이번 요청의 최종 전달본으로 사용합니다."),
                ToolMessage(
                    content=json.dumps({"status": "ok", "file_path": "output/save_text/response.txt"}),
                    name="save_text",
                    tool_call_id="tool-1",
                ),
            ],
            debug_info={},
        )

        self.assertEqual(result["message"], "공유 내용\n- 이 메시지 자체를 이번 요청의 최종 전달본으로 사용합니다.")
        self.assertTrue(result["filepath"].endswith("output\\save_text\\response.txt"))
        self.assertEqual(result["response_payload"]["answer"], result["message"])
