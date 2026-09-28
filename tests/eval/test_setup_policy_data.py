"""Preparation policy revisions preserve the generation evidence they extend."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from src.eval.config_models import BenchmarkCase
from src.eval.dataset_validation import validate_dataset
from src.eval.nemo_generate import _sha256
from src.eval.release_dataset import _lineage_errors
from tests.eval.test_dataset_validation import grounded_case, fixtures_path


def test_release_validation_requires_an_explicit_policy_for_every_preparation_turn(fixtures_path):
    case = grounded_case(setup_turns=["Remember this context."])
    errors = validate_dataset([case], fixtures_path=fixtures_path, expected_count=1)
    assert any("setup_forbidden_tools" in error for error in errors)


def policy_revision():
    root = Path(__file__).resolve().parents[2] / "data/benchmarks"
    def find(path):
        return next(row for line in path.read_text(encoding="utf-8").splitlines()
                    for row in [json.loads(line)] if row["case_id"] == "release_action_003")
    authored = find(root / "design/action_specs.jsonl")
    row = find(root / "fixtures/cases.generated.jsonl")
    authored.pop("setup_forbidden_tools", None)
    row.pop("setup_forbidden_tools", None)
    row["provenance"].pop("setup_policy_revision", None)
    original_generation = deepcopy(row["provenance"]["nemo_generation"])
    revised = deepcopy(authored)
    revised["setup_forbidden_tools"] = [["save_text", "slack_notify"]]
    row["setup_forbidden_tools"] = revised["setup_forbidden_tools"]
    row["provenance"]["setup_policy_revision"] = {
        "schema": "setup-forbidden-tools-v1",
        "source_spec_sha256": _sha256(authored),
        "authored_spec_sha256": _sha256(revised),
    }
    return row, revised, original_generation


def test_reviewed_policy_addition_keeps_the_authentic_generation_hash():
    row, authored, generation = policy_revision()
    assert _lineage_errors(BenchmarkCase.model_validate(row), authored) == []
    assert row["provenance"]["nemo_generation"] == generation
    assert generation["source_spec_sha256"] != _sha256(authored)


@pytest.mark.parametrize("change", ["final_policy", "query", "source_hash", "current_hash"])
def test_policy_revision_cannot_hide_changes_to_original_generation_facts(change):
    row, authored, _ = policy_revision()
    revision = row["provenance"]["setup_policy_revision"]
    if change == "final_policy":
        row["forbidden_tools"] = authored["forbidden_tools"] = []
        revision["authored_spec_sha256"] = _sha256(authored)
    elif change == "query":
        row["query"] = "An unrelated new user request."
    elif change == "source_hash":
        revision["source_spec_sha256"] = "0" * 64
    else:
        revision["authored_spec_sha256"] = "0" * 64
    assert _lineage_errors(BenchmarkCase.model_validate(row), authored)


ALL_TOOLS = {"tavily_search", "upload_search", "save_text", "slack_notify"}
UPLOAD_ALLOWED = {("release_action_003", 0), ("release_action_004", 0),
                  ("release_action_005", 0), ("release_action_005", 1)}


def test_approved_corpus_has_exactly_the_reviewed_preparation_policies():
    root = Path(__file__).resolve().parents[2] / "data/benchmarks"
    cases = [json.loads(line) for line in (root / "fixtures/cases.generated.jsonl").read_text(encoding="utf-8").splitlines()]
    prepared = [case for case in cases if case.get("setup_turns")]
    assert len(prepared) == 23
    assert sum(len(case["setup_turns"]) for case in prepared) == 27
    for case in prepared:
        assert len(case["setup_forbidden_tools"]) == len(case["setup_turns"])
        for index, forbidden in enumerate(case["setup_forbidden_tools"]):
            expected = ALL_TOOLS - {"upload_search"} if (case["case_id"], index) in UPLOAD_ALLOWED else ALL_TOOLS
            assert set(forbidden) == expected, (case["case_id"], index)


def test_original_package_hashes_are_recoverable_without_rewriting_generation_history():
    root = Path(__file__).resolve().parents[2] / "data/benchmarks"
    review = json.loads((root / "design/release_review.json").read_text(encoding="utf-8"))
    policy_review = review["execution_policy_review"]
    predecessor = policy_review["predecessor"]
    assert policy_review["deterministic_checks"]["sha256"] == review["candidate_sha256"]
    expected_hashes = {"cases.generated.jsonl": predecessor["candidate_sha256"],
                       **predecessor["artifact_hashes"]}
    for name, digest in expected_hashes.items():
        path = root / name if name.startswith("design/") else root / "fixtures" / name
        content = path.read_bytes()
        if name in {"cases.generated.jsonl", "design/action_specs.jsonl", "design/retrieval_specs.jsonl"}:
            records = [json.loads(line) for line in content.decode("utf-8").splitlines()]
            for record in records:
                record.pop("setup_forbidden_tools", None)
                record.get("provenance", {}).pop("setup_policy_revision", None)
            kwargs = {"separators": (",", ":")} if name == "cases.generated.jsonl" else {}
            content = "".join(json.dumps(record, ensure_ascii=False, **kwargs) + "\n" for record in records).encode("utf-8")
        assert hashlib.sha256(content).hexdigest() == digest, name


def test_source_builder_requires_explicit_preparation_policies(monkeypatch):
    from script import build_release_sources as builder
    monkeypatch.setattr(builder, "CASES", [])
    with pytest.raises(ValueError, match="setup_forbidden_tools"):
        builder.case("docs_only", "synthetic", "Question", [], [], setup=["Prepare"])
    builder.case("docs_only", "synthetic", "Question", [], [], setup=["Prepare"],
                 setup_forbidden_tools=[sorted(ALL_TOOLS)])
    assert builder.CASES[0]["setup_forbidden_tools"] == [sorted(ALL_TOOLS)]
