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






