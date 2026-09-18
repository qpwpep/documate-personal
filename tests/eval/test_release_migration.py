"""The active release changes one authored task without rewriting its history."""

import hashlib
import json
from pathlib import Path

from src.eval.nemo_generate import _sha256, assemble_candidates


ROOT = Path(__file__).resolve().parents[2] / "data/benchmarks"
HISTORY = ROOT / "history/release-nemo-v1"
CHANGED_CASE = "release_action_028"


def rows(path):
    return {row["case_id"]: row for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() for row in [json.loads(line)]}


def test_original_approval_still_identifies_the_archived_original_bytes():
    review = json.loads((HISTORY / "design/release_review.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((HISTORY / "fixtures/cases.generated.jsonl").read_bytes()).hexdigest() == review["candidate_sha256"]
    for name, digest in review["artifact_hashes"].items():
        source = HISTORY / name if name.startswith("design/") else HISTORY / "fixtures" / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, name


def test_migration_preserves_119_cases_and_their_authentic_generation_lineage():
    before = rows(HISTORY / "fixtures/cases.generated.jsonl")
    after = rows(ROOT / "fixtures/cases.generated.jsonl")
    assert len(before) == len(after) == 120
    assert before.keys() == after.keys()
    assert {key for key in before if before[key] != after[key]} == {CHANGED_CASE}
    assert after[CHANGED_CASE]["expected_tools"] == []
    assert "upload_search" in after[CHANGED_CASE]["forbidden_tools"]
    assert after[CHANGED_CASE]["save_expectation"]["outcome"] == "must_not_execute"
    new_spec = rows(ROOT / "design/action_specs.jsonl")[CHANGED_CASE]
    lineage = after[CHANGED_CASE]["provenance"]["nemo_generation"]
    assert lineage["source_spec_sha256"] == _sha256(new_spec)
    assert lineage["source_spec_sha256"] != before[CHANGED_CASE]["provenance"]["nemo_generation"]["source_spec_sha256"]


def test_active_approval_matches_the_new_format_and_new_package():
    plan = json.loads((ROOT / "design/plan.json").read_text(encoding="utf-8"))
    review = json.loads((ROOT / "design/release_review.json").read_text(encoding="utf-8"))
    assert plan["dataset_version"] == review["dataset_version"] == "release-nemo-v2"
    assert plan["text_format"] == "utf8-lf-v1"
    candidate = (ROOT / "fixtures/cases.generated.jsonl").read_bytes()
    assert b"\r" not in candidate
    assert hashlib.sha256(candidate).hexdigest() == review["candidate_sha256"]
    assert review["unresolved_issues"] == []
    assert len(review["approved_case_ids"]) == 120
    for name, digest in review["artifact_hashes"].items():
        path = ROOT / name if name.startswith("design/") else ROOT / "fixtures" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


def test_new_generation_is_replayable_without_relabelling_historical_runs():
    design = ROOT / "design"
    manifest = json.loads((design / "generation_manifest.json").read_text(encoding="utf-8"))
    previous = json.loads((HISTORY / "design/generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_runs"][:-1] == previous["generation_runs"]
    assert manifest["migration"]["retained_case_count"] == 119
    assert manifest["migration"]["changed_case_ids"] == [CHANGED_CASE]
    run = design / "runs/missing-upload-v2"
    source = list(rows(run / "source_specs.jsonl").values())
    generated = list(rows(run / "generated_rows.jsonl").values())
    run_manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    accepted, rejected = assemble_candidates(source, generated, run_manifest)
    assert rejected == []
    assert accepted == list(rows(run / "candidates.jsonl").values())
    assert accepted == [rows(ROOT / "fixtures/cases.generated.jsonl")[CHANGED_CASE]]
    assert run_manifest["status"] == "complete"
    assert run_manifest["accepted_count"] == 1
    assert run_manifest["rejected_count"] == 0
    assert hashlib.sha256((run / "candidates.jsonl").read_bytes()).hexdigest() == run_manifest["candidate_output_sha256"]
    assert hashlib.sha256((run / "input_specs.jsonl").read_bytes()).hexdigest() == run_manifest["source_files"][0]["sha256"]
    for name, digest in manifest["generation_runs"][-1]["archived_files"].items():
        assert hashlib.sha256((design / name).read_bytes()).hexdigest() == digest, name
    # Retained authored contracts are also unchanged: LF migration is storage only.
    before = rows(HISTORY / "design/action_specs.jsonl")
    after = rows(design / "action_specs.jsonl")
    assert {key for key in before if before[key] != after[key]} == {CHANGED_CASE}
    assert rows(HISTORY / "design/retrieval_specs.jsonl") == rows(design / "retrieval_specs.jsonl")
