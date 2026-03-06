import json
import unittest

from src.eval.schemas import BenchmarkCase, CaseResult


class BackwardCompatInputsTest(unittest.TestCase):
    def test_seed_without_scenario_defaults_to_seed_mutation(self) -> None:
        raw = {"case_id": "legacy_seed", "category": "docs_only", "query": "legacy query"}
        case = BenchmarkCase.model_validate(raw)
        self.assertEqual(case.scenario, "seed_mutation")

    def test_legacy_raw_result_with_errors_field_is_still_parseable(self) -> None:
        legacy_result = {
            "run_id": "legacy_run",
            "case_id": "legacy_case",
            "category": "docs_only",
            "query": "legacy query",
            "session_id": "legacy_session",
            "endpoint": "http://localhost:8000/agent",
            "request_payload": {"query": "legacy query", "session_id": "legacy_session"},
            "http_status": 0,
            "response_text": "",
            "errors": ["request timeout"],
            "created_at_utc": "2026-01-01T00:00:00+00:00",
        }
        parsed = CaseResult.model_validate_json(json.dumps(legacy_result, ensure_ascii=False))
        self.assertEqual(parsed.runtime_errors, [])
        self.assertEqual(parsed.response_errors, [])
        self.assertEqual(parsed.judge_errors, [])

    def test_case_result_parses_extended_planner_diagnostics(self) -> None:
        raw = {
            "run_id": "run-1",
            "case_id": "case-1",
            "category": "docs_only",
            "query": "docs query",
            "session_id": "session-1",
            "endpoint": "http://localhost:8000/agent",
            "request_payload": {"query": "docs query", "session_id": "session-1"},
            "http_status": 200,
            "response_text": "현재 확인 가능한 근거를 찾지 못했습니다.",
            "planner_diagnostics": {
                "status": "llm",
                "reason": "missing_required_retrieval",
                "fallback_routes": [],
                "intent_required": True,
                "required_routes": ["docs"],
                "override_applied": True,
                "override_reason": "missing_required_retrieval",
            },
            "created_at_utc": "2026-01-01T00:00:00+00:00",
        }
        parsed = CaseResult.model_validate_json(json.dumps(raw, ensure_ascii=False))
        self.assertIsNotNone(parsed.planner_diagnostics)
        self.assertTrue(parsed.planner_diagnostics.intent_required)
        self.assertEqual(parsed.planner_diagnostics.required_routes, ["docs"])
        self.assertEqual(parsed.planner_diagnostics.override_reason, "missing_required_retrieval")


if __name__ == "__main__":
    unittest.main()
