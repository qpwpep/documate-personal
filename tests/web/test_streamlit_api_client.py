from __future__ import annotations

import unittest
from unittest.mock import patch

import requests

from src.app.web.streamlit_api_client import AgentCallResult, AgentRequestContext, _iter_sse_events, get_agent_response, stream_agent_response


class _Response:
    def __init__(self, status_code: int, payload: object, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        return self._payload


class _StreamResponse:
    def __init__(self, status_code: int, chunks: list[str], text: str = "") -> None:
        self.status_code = status_code
        self._chunks = chunks
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_content(self, chunk_size=None, decode_unicode: bool = False):
        _ = chunk_size, decode_unicode
        for chunk in self._chunks:
            yield chunk


class _BrokenStreamResponse(_StreamResponse):
    def iter_content(self, chunk_size=None, decode_unicode: bool = False):
        yielded = False
        for chunk in super().iter_content(chunk_size=chunk_size, decode_unicode=decode_unicode):
            yield chunk
            yielded = True
        if yielded:
            raise RuntimeError("stream broke")


class StreamlitApiClientTest(unittest.TestCase):
    @patch("src.app.web.streamlit_api_client.requests.post")
    def test_get_agent_response_sends_expected_payload(self, mock_post) -> None:
        mock_post.return_value = _Response(
            200,
            {
                "response": {
                    "answer": "응답",
                    "evidence": [{"kind": "official", "url_or_path": "https://docs.example.com"}],
                },
                "file_path": "output/result.txt",
            },
        )

        result = get_agent_response(
            "질문",
            AgentRequestContext(
                fastapi_url="http://127.0.0.1:8000",
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
        self.assertEqual(result.answer, "응답")
        self.assertEqual(result.file_path, "output/result.txt")
        self.assertEqual(len(result.evidence_items), 1)

    @patch("src.app.web.streamlit_api_client.requests.post")
    def test_get_agent_response_handles_error_status(self, mock_post) -> None:
        mock_post.return_value = _Response(500, {}, text="server exploded")

        result = get_agent_response(
            "질문",
            AgentRequestContext(
                fastapi_url="http://127.0.0.1:8000",
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
        "src.app.web.streamlit_api_client.requests.post",
        side_effect=requests.exceptions.Timeout,
    )
    def test_get_agent_response_handles_timeout(self, _mock_post) -> None:
        result = get_agent_response(
            "질문",
            AgentRequestContext(
                fastapi_url="http://127.0.0.1:8000",
                session_id="session-1",
            ),
        )

        self.assertEqual(result.answer, "요청이 타임아웃되었습니다. 서버 상태를 확인해 주세요.")

    @patch(
        "src.app.web.streamlit_api_client.requests.post",
        side_effect=requests.exceptions.ConnectionError,
    )
    def test_get_agent_response_handles_connection_error(self, _mock_post) -> None:
        result = get_agent_response(
            "질문",
            AgentRequestContext(
                fastapi_url="http://127.0.0.1:8000",
                session_id="session-1",
            ),
        )

        self.assertEqual(
            result.answer,
            "FastAPI 서버에 연결할 수 없습니다. 서버(8000번 포트) 실행 여부를 확인해 주세요.",
        )

    @patch("src.app.web.streamlit_api_client.requests.post", side_effect=RuntimeError("boom"))
    def test_get_agent_response_handles_unexpected_error(self, _mock_post) -> None:
        result = get_agent_response(
            "질문",
            AgentRequestContext(
                fastapi_url="http://127.0.0.1:8000",
                session_id="session-1",
            ),
        )

        self.assertEqual(result.answer, "요청 중 예기치 않은 오류가 발생했습니다: boom")

    def test_iter_sse_events_parses_chunked_frames(self) -> None:
        chunks = [
            'event: request_started\ndata: {"request_id":"r',
            'eq-1"}\n\n',
            'event: progress_snapshot\ndata: {"summary":"docs ready"}\n\n',
            'event: final_response\ndata: {"response":{"answer":"응',
            '답","evidence":[]},"file_path":"output/result.txt"}\n\n',
        ]

        events = list(_iter_sse_events(chunks))

        self.assertEqual([event.event for event in events], ["request_started", "progress_snapshot", "final_response"])
        self.assertEqual(events[0].data["request_id"], "req-1")
        self.assertEqual(events[1].data["summary"], "docs ready")
        self.assertIsNotNone(events[2].result)
        self.assertEqual(events[2].result.answer, "응답")

    @patch("src.app.web.streamlit_api_client.get_agent_response")
    @patch("src.app.web.streamlit_api_client.requests.post")
    def test_stream_agent_response_falls_back_before_first_event(
        self,
        mock_post,
        mock_get_agent_response,
    ) -> None:
        mock_post.side_effect = requests.exceptions.ConnectionError()
        mock_get_agent_response.return_value = AgentCallResult(answer="fallback")

        events = list(
            stream_agent_response(
                "질문",
                AgentRequestContext(
                    fastapi_url="http://127.0.0.1:8000",
                    session_id="session-1",
                ),
            )
        )

        self.assertEqual([event.event for event in events], ["final_response"])
        self.assertIsNotNone(events[0].result)
        self.assertEqual(events[0].result.answer, "fallback")

    @patch("src.app.web.streamlit_api_client.requests.post")
    def test_stream_agent_response_emits_error_after_stream_break(self, mock_post) -> None:
        mock_post.return_value = _BrokenStreamResponse(
            200,
            [
                'event: request_started\ndata: {"request_id":"req-1"}\n\n',
            ],
        )

        events = list(
            stream_agent_response(
                "질문",
                AgentRequestContext(
                    fastapi_url="http://127.0.0.1:8000",
                    session_id="session-1",
                ),
            )
        )

        self.assertEqual([event.event for event in events], ["request_started", "error"])


if __name__ == "__main__":
    unittest.main()
