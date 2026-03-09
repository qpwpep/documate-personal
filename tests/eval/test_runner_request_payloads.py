import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.eval.runner_online import _run_single_case
from src.eval.schemas import BenchmarkCase, BenchmarkConfig


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self) -> dict:
        return self._payload


class _DummyJudge:
    def score_case(self, case, response_text, tool_calls):
        _ = (case, response_text, tool_calls)
        return (None, None, None)


class RunnerRequestPayloadTest(unittest.TestCase):
    @patch("src.eval.runner_online.requests.post")
    def test_slack_destination_fields_are_forwarded(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            200,
            {
                "response": {"answer": "shared", "evidence": []},
                "trace": "trace-id",
                "file_path": "",
                "debug": {"tool_calls": ["slack_notify"], "token_usage": {}, "observed_evidence": []},
            },
        )
        case = BenchmarkCase(
            case_id="tool_seed_999",
            category="tool_action",
            query="share this to slack",
            expected_tools=["slack_notify"],
            slack_channel_id="C123BENCH",
            slack_user_id="U123BENCH",
            slack_email="bench@example.com",
        )

        _run_single_case(
            run_id="run-slack-payload",
            endpoint="http://localhost:8000",
            fixtures_path=Path("data/benchmarks/fixtures/cases.generated.jsonl"),
            case=case,
            timeout_seconds=5,
            judge=_DummyJudge(),
            config=BenchmarkConfig(),
        )

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        self.assertEqual(payload["slack_channel_id"], "C123BENCH")
        self.assertEqual(payload["slack_user_id"], "U123BENCH")
        self.assertEqual(payload["slack_email"], "bench@example.com")


if __name__ == "__main__":
    unittest.main()
