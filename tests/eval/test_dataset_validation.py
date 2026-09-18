import json
from pathlib import Path

import pytest

from src.eval.config_models import BenchmarkCase
from src.eval.dataset_validation import validate_dataset
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


def test_grounded_dataset_checks_attachments_and_distribution(fixtures_path):
    assert validate_dataset([grounded_case()], fixtures_path=fixtures_path, expected_count=1,
                            expected_distributions={"category": {"rag_only": 1}}) == []


def test_duplicate_records_and_ids_are_rejected(fixtures_path):
    case = grounded_case()
    other = grounded_case(case_id="another_id")
    errors = validate_dataset([case, other, case], fixtures_path=fixtures_path, expected_count=3)
    assert any("duplicate case_id" in error for error in errors)
    assert any("duplicate content" in error for error in errors)


def test_oracle_quotes_must_be_supported_and_referenced_upload_declared(fixtures_path):
    case = grounded_case()
    case.oracle.evidence[0].excerpt = "LIMIT = 99"
    errors = validate_dataset([case], fixtures_path=fixtures_path, expected_count=1)
    assert any("excerpt not found" in error for error in errors)
    case.oracle.evidence[0].source = "other.py"
    errors = validate_dataset([case], fixtures_path=fixtures_path, expected_count=1)
    assert any("undeclared upload evidence" in error for error in errors)


def test_traversal_and_missing_attachments_are_rejected(fixtures_path):
    case = grounded_case(upload_fixtures=["../outside.py", "missing.py"])
    errors = validate_dataset([case], fixtures_path=fixtures_path, expected_count=1)
    assert any("outside uploads" in error for error in errors)
    assert any("missing upload" in error for error in errors)


def test_conflicting_expectations_and_wrong_plan_counts_are_reported(fixtures_path):
    case = grounded_case(expected_tools=["upload_search", "save_text"],
                         must_not_include=["17"])
    errors = validate_dataset([case], fixtures_path=fixtures_path, expected_count=120,
                              expected_distributions={"difficulty": {"hard": 1}})
    assert any("expected 120 cases" in error for error in errors)
    assert any("expected/forbidden tools overlap" in error for error in errors)
    assert any("include/exclude overlap" in error for error in errors)
    assert any("save expectation is missing" in error for error in errors)
    assert any("difficulty distribution" in error for error in errors)


def test_official_snapshot_and_user_evidence_use_explicit_sources(fixtures_path):
    case = grounded_case()
    case.oracle.evidence[0].source = "https://example.org/docs/limit"
    assert any("source text unavailable" in e for e in validate_dataset(
        [case], fixtures_path=fixtures_path, expected_count=1))
    assert validate_dataset([case], fixtures_path=fixtures_path, expected_count=1,
                            source_manifest={"https://example.org/docs/limit": "LIMIT = 17"}) == []
    case.oracle.evidence[0].source = "user:query"
    case.oracle.evidence[0].excerpt = "What limit is configured?"
    assert validate_dataset([case], fixtures_path=fixtures_path, expected_count=1) == []


def test_notebook_source_is_read_as_cell_text(fixtures_path):
    notebook = {"cells": [{"cell_type": "code", "source": ["LIMIT = 17\n", "print(LIMIT)\n"]}]}
    (fixtures_path.parent / "uploads" / "limits.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    case = grounded_case(upload_fixtures=["limits.ipynb"])
    case.oracle.evidence[0].source = "limits.ipynb"
    case.oracle.evidence[0].excerpt = "LIMIT = 17\nprint(LIMIT)"
    assert validate_dataset([case], fixtures_path=fixtures_path, expected_count=1) == []


def test_legitimate_prohibition_needs_no_setup_or_search(fixtures_path):
    case = grounded_case(category="tool_action", scenario="correction", query="Cancel saving.",
                         upload_fixtures=[], expected_tools=[], forbidden_tools=["save_text"],
                         save_expectation={"outcome": "must_not_execute"})
    case.oracle.evidence[0].source = "user:query"
    case.oracle.evidence[0].excerpt = "Cancel saving."
    assert validate_dataset([case], fixtures_path=fixtures_path, expected_count=1) == []


@pytest.mark.parametrize("setup_turns", [[], ["Remember that the requested file is not attached."]])
def test_required_upload_search_needs_a_declared_attachment(fixtures_path, setup_turns):
    case = grounded_case(upload_fixtures=[], setup_turns=setup_turns)
    case.oracle.evidence[0].source = "user:query"
    case.oracle.evidence[0].excerpt = case.query
    errors = validate_dataset([case], fixtures_path=fixtures_path, expected_count=1)
    assert any("upload_search requires a declared attachment" in error for error in errors)


def test_legacy_attachment_declaration_satisfies_upload_search(fixtures_path):
    case = grounded_case(upload_fixtures=[], upload_fixture="limits.py")
    assert validate_dataset([case], fixtures_path=fixtures_path, expected_count=1) == []


def test_authored_design_evidence_is_distinct_from_runtime_uploads(fixtures_path):
    case = grounded_case()
    case.oracle.evidence[0].source = "design:limit_lookup"
    case.oracle.evidence[0].excerpt = "Author fixed LIMIT = 17."
    assert validate_dataset([case], fixtures_path=fixtures_path, expected_count=1,
                            source_manifest={"design:limit_lookup": "Author fixed LIMIT = 17."}) == []
    assert any("source text unavailable" in e for e in validate_dataset(
        [case], fixtures_path=fixtures_path, expected_count=1))


@pytest.mark.parametrize("filename,data", [
    ("limits.py", b"LIMIT = 17\n"),
    ("limits.ipynb", json.dumps({"cells": [{"source": ["LIMIT = 17\n"]}]}).encode("utf-8")),
])
def test_attachment_snapshot_supplies_evidence_without_reopening_files(tmp_path, filename, data):
    case = grounded_case(upload_fixtures=[filename])
    case.oracle.evidence[0].source = filename
    assert validate_dataset(
        [case], fixtures_path=tmp_path / "not-created" / "cases.jsonl", expected_count=1,
        attachment_bytes={filename: data},
    ) == []


def test_missing_snapshot_attachment_cannot_fall_back_to_an_existing_file(fixtures_path):
    errors = validate_dataset([grounded_case()], fixtures_path=fixtures_path, expected_count=1,
                              attachment_bytes={})
    assert any("missing upload: limits.py" in error for error in errors)


def test_snapshot_content_is_checked_instead_of_current_disk_content(fixtures_path):
    errors = validate_dataset([grounded_case()], fixtures_path=fixtures_path, expected_count=1,
                              attachment_bytes={"limits.py": b"LIMIT = 99\n"})
    assert any("excerpt not found" in error for error in errors)


@pytest.mark.parametrize("filename", ["../outside.py", "/outside.py", "C:/outside.py", "C:outside.py"])
def test_snapshot_keys_cannot_authorize_unsafe_attachment_paths(tmp_path, filename):
    case = grounded_case(upload_fixtures=[filename])
    case.oracle.evidence[0].source = filename
    errors = validate_dataset([case], fixtures_path=tmp_path / "cases.jsonl", expected_count=1,
                              attachment_bytes={filename: b"LIMIT = 17\n"})
    assert any("outside uploads" in error for error in errors)
