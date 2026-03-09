from __future__ import annotations

import unittest
from unittest.mock import patch

import requests

from src.web.streamlit_api_client import AgentRequestContext, get_agent_response


class _Response:
    def __init__(self, status_code: int, payload: object, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        return self._payload


class StreamlitApiClientTest(unittest.TestCase):
    @patch("src.web.streamlit_api_client.requests.post")
    def test_get_agent_response_sends_expected_payload(self, mock_post) -> None:
        mock_post.return_value = _Response(
            200,
            {
                "response": {
                    "answer": "답변",
                    "evidence": [{"kind": "official", "url_or_path": "https://docs.example.com"}],
                },
                "file_path": "output/result.txt",
            },
        )

        result = get_agent_response(
            "질문",
            AgentRequestContext(
                fastapi_url="http://localhost:8000",
                session_id="session-1",
                slack_user_id="U123",
                slack_email="user@example.com",
                slack_channel_id="C123",
                upload_file_path="uploads/session-1/sample.py",
            ),
        )

        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["timeout"], 60)
        self.assertEqual(
            kwargs["json"],
            {
                "query": "질문",
                "session_id": "session-1",
                "slack_user_id": "U123",
                "slack_email": "user@example.com",
                "slack_channel_id": "C123",
                "upload_file_path": "uploads/session-1/sample.py",
            },
        )
        self.assertEqual(result.answer, "답변")
        self.assertEqual(result.file_path, "output/result.txt")
        self.assertEqual(len(result.evidence_items), 1)

    @patch("src.web.streamlit_api_client.requests.post")
    def test_get_agent_response_handles_error_status(self, mock_post) -> None:
        mock_post.return_value = _Response(500, {}, text="server exploded")

        result = get_agent_response(
            "질문",
            AgentRequestContext(
                fastapi_url="http://localhost:8000",
                session_id="session-1",
            ),
        )

        self.assertEqual(
            result.answer,
            "Agent 호출 실패: 상태 코드 500\n응답: server exploded",
        )
        self.assertIsNone(result.file_path)
        self.assertEqual(result.evidence_items, [])

    @patch(
        "src.web.streamlit_api_client.requests.post",
        side_effect=requests.exceptions.Timeout,
    )
    def test_get_agent_response_handles_timeout(self, _mock_post) -> None:
        result = get_agent_response(
            "질문",
            AgentRequestContext(
                fastapi_url="http://localhost:8000",
                session_id="session-1",
            ),
        )

        self.assertEqual(result.answer, "요청이 타임아웃되었습니다. 서버 상태를 확인해 주세요.")

    @patch(
        "src.web.streamlit_api_client.requests.post",
        side_effect=requests.exceptions.ConnectionError,
    )
    def test_get_agent_response_handles_connection_error(self, _mock_post) -> None:
        result = get_agent_response(
            "질문",
            AgentRequestContext(
                fastapi_url="http://localhost:8000",
                session_id="session-1",
            ),
        )

        self.assertEqual(
            result.answer,
            "FastAPI 서버에 연결할 수 없습니다. 서버(8000번 포트) 실행 여부를 확인해 주세요.",
        )

    @patch("src.web.streamlit_api_client.requests.post", side_effect=RuntimeError("boom"))
    def test_get_agent_response_handles_unexpected_error(self, _mock_post) -> None:
        result = get_agent_response(
            "질문",
            AgentRequestContext(
                fastapi_url="http://localhost:8000",
                session_id="session-1",
            ),
        )

        self.assertEqual(result.answer, "요청 중 예기치 않은 오류가 발생했습니다: boom")


if __name__ == "__main__":
    unittest.main()
