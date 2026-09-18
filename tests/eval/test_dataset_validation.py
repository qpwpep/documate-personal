import json
from pathlib import Path

import pytest

from src.eval.config_models import BenchmarkCase
from src.eval.io import dump_jsonl, load_cases_jsonl
from src.eval.judge_llm import LLMJudge


def grounded_case(**updates):
    payload = {
        "case_id": "limit_lookup", "category": "rag_only", "scenario": "standard",
        "difficulty": "easy", "evaluation_role": "new_evaluation", "capability": "parameter_lookup",
        "query": "What limit is configured?", "upload_fixtures": ["limits.py"],
        "expected_tools": ["upload_search"], "forbidden_tools": ["save_text"],
        "must_include": ["17"], "oracle": {
            "required_facts": ["The configured limit is 17."],
            "expected_behaviors": ["Report the configured limit with its source."],
            "forbidden_behaviors": ["Do not save or invent another limit."],
            "evidence": [{"source": "limits.py", "locator": "line 1", "excerpt": "LIMIT = 17"}],
            "ambiguity_resolution": "The query names the only configuration file.",
        }, "provenance": {"generator": "test fixture"},
    }
    payload.update(updates)
    return BenchmarkCase.model_validate(payload)


@pytest.fixture
def fixtures_path(tmp_path):
    (tmp_path / "uploads").mkdir()
    (tmp_path / "uploads" / "limits.py").write_text("LIMIT = 17\n", encoding="utf-8")
    return tmp_path / "cases.jsonl"


def test_oracle_round_trip_and_judge_payload_preserve_actual_facts(fixtures_path):
    original = grounded_case()
    dump_jsonl(fixtures_path, [original])
    restored = load_cases_jsonl(fixtures_path)[0]
    payload = LLMJudge.build_case_payload(case=restored, tool_calls=[], response=None)
    assert payload["case"]["oracle"]["required_facts"] == ["The configured limit is 17."]
    assert payload["case"]["oracle"]["evidence"][0]["excerpt"] == "LIMIT = 17"
    assert restored.difficulty == "easy"
    assert restored.provenance == {"generator": "test fixture"}






























