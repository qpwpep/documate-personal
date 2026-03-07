import json
import unittest

from langchain_core.messages import SystemMessage

from src.nodes.actions import make_action_postprocess_node

from .helpers import _ToolWrapper


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
            {
                "user_input": "save this answer to txt",
                "final_answer": "final answer text",
                "messages": [],
            }
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
            {
                "user_input": "send this to slack",
                "final_answer": "final answer text",
                "messages": [],
            }
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
            {
                "user_input": "send this to slack",
                "final_answer": "final answer text",
                "messages": [
                    SystemMessage(
                        content="[Slack Destinations]\nchannel_id=C123BENCH\n(When the user asks to send to Slack, call slack_notify with these values.)"
                    )
                ],
            }
        )

        self.assertEqual(recorded["channel_id"], "C123BENCH")
        tool_messages = updates.get("messages", [])
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].name, "slack_notify")
