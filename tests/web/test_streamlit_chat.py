from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from src.web.streamlit_api_client import AgentCallResult
from src.web.streamlit_chat import process_chat_prompt, render_chat_history


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeStreamlit:
    def __init__(self) -> None:
        self.chat_roles: list[str] = []
        self.expander_labels: list[str] = []
        self.markdowns: list[tuple[str, bool]] = []
        self.infos: list[str] = []
        self.spinner_messages: list[str] = []
        self.rerun_calls = 0

    def chat_message(self, role: str) -> _NullContext:
        self.chat_roles.append(role)
        return _NullContext()

    def expander(self, label: str) -> _NullContext:
        self.expander_labels.append(label)
        return _NullContext()

    def markdown(self, body: str, unsafe_allow_html: bool = False) -> None:
        self.markdowns.append((body, unsafe_allow_html))

    def info(self, body: str) -> None:
        self.infos.append(body)

    def spinner(self, body: str) -> _NullContext:
        self.spinner_messages.append(body)
        return _NullContext()

    def rerun(self) -> None:
        self.rerun_calls += 1


class StreamlitChatTest(unittest.TestCase):
    def test_render_chat_history_renders_evidence_and_existing_download_only(self) -> None:
        fake_st = _FakeStreamlit()
        with TemporaryDirectory() as temp_dir:
            saved_file = Path(temp_dir) / "answer.txt"
            saved_file.write_text("saved", encoding="utf-8")

            messages = [
                {
                    "role": "assistant",
                    "content": "answer",
                    "file_path": str(saved_file),
                    "evidence": [
                        {"kind": "official", "title": "Docs", "url_or_path": "https://docs.example.com"},
                        "skip-me",
                    ],
                },
                {
                    "role": "assistant",
                    "content": "missing file",
                    "file_path": str(saved_file.parent / "missing.txt"),
                    "evidence": [],
                },
            ]

            with patch("src.web.streamlit_chat.st", fake_st):
                render_chat_history(messages, "http://localhost:8000")

        self.assertEqual(fake_st.expander_labels, ["근거 보기"])
        self.assertTrue(any("Docs" in body for body, _ in fake_st.markdowns))
        self.assertTrue(any("파일 저장 완료" in body for body in fake_st.infos))
        self.assertEqual(sum("download/answer.txt" in body for body, _ in fake_st.markdowns), 1)

    def test_process_chat_prompt_appends_messages_in_order_and_reruns(self) -> None:
        fake_st = _FakeStreamlit()
        appended_messages: list[dict[str, object]] = []

        def append_message(message):
            appended_messages.append(message)

        def call_agent(user_input: str) -> AgentCallResult:
            self.assertEqual(user_input, "질문")
            return AgentCallResult(
                answer="응답",
                file_path="output/result.txt",
                evidence_items=[{"kind": "official"}],
            )

        with patch("src.web.streamlit_chat.st", fake_st):
            process_chat_prompt(
                call_agent=call_agent,
                prompt="질문",
                append_user_message=append_message,
                append_assistant_message=append_message,
            )

        self.assertEqual([message["role"] for message in appended_messages], ["user", "assistant"])
        self.assertEqual(appended_messages[0]["content"], "질문")
        self.assertEqual(appended_messages[1]["content"], "응답")
        self.assertEqual(fake_st.chat_roles, ["user"])
        self.assertEqual(fake_st.spinner_messages, ["Agent가 생각 중입니다..."])
        self.assertEqual(fake_st.rerun_calls, 1)


if __name__ == "__main__":
    unittest.main()
