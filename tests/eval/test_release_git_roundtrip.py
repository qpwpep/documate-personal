"""The shipped approval must survive checkout and relocation as a complete package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from src.eval.release_dataset import load_release_input, load_reviewed_release, promote


REPOSITORY = Path(__file__).resolve().parents[2]
BENCHMARKS = Path("data/benchmarks")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _copy_release_package(repo: Path) -> dict[Path, bytes]:
    paths = [REPOSITORY / ".gitattributes", REPOSITORY / ".gitignore",
             REPOSITORY / BENCHMARKS / "fixtures/cases.generated.jsonl"]
    for directory in ("design", "fixtures/uploads", "history"):
        paths.extend(path for path in (REPOSITORY / BENCHMARKS / directory).rglob("*") if path.is_file())
    originals = {}
    for source in paths:
        relative = source.relative_to(REPOSITORY)
        content = source.read_bytes()
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        if relative not in {Path(".gitattributes"), Path(".gitignore")}:
            originals[relative] = content
    return originals


@pytest.mark.parametrize("autocrlf", ["true", "false", "input"])
def test_shipped_approval_survives_git_checkout_and_custom_promotion(tmp_path, autocrlf):
    if shutil.which("git") is None:
        pytest.skip("Git is required to verify approval portability")
    repo = tmp_path / "repository"
    repo.mkdir()
    originals = _copy_release_package(repo)
    design = repo / BENCHMARKS / "design"
    review = design / "release_review.json"
    candidates = repo / BENCHMARKS / "fixtures/cases.generated.jsonl"
    promoted = candidates.with_name("custom-release.jsonl")
    result = promote(candidates, review=review, design=design, out=promoted)
    assert result["errors"] == []
    store = promoted.with_name(promoted.name + ".approvals")
    originals.update({
        path.relative_to(repo): path.read_bytes()
        for path in [promoted, *(path for path in store.rglob("*")
                                if path.is_file() and path.name != ".save.lock")]
    })
    _git(repo, "init", "--quiet")
    # These settings and every Git mutation belong only to the temporary repo.
    _git(repo, "config", "core.autocrlf", autocrlf)
    _git(repo, "config", "core.safecrlf", "false")
    _git(repo, "config", "core.eol", "native")
    _git(repo, "config", "core.longpaths", "true")
    lock = store / ".save.lock"
    assert lock.is_file()
    _git(repo, "check-ignore", "--quiet", lock.relative_to(repo).as_posix())
    _git(repo, "add", ".")
    for relative in originals:
        (repo / relative).unlink()
    lock.unlink()
    _git(repo, "checkout-index", "--all", "--force")
    assert not lock.exists()

    approved = load_reviewed_release(candidates, review=review, design=design)

    assert len(approved.cases) == 120
    assert approved.inspection["errors"] == []
    assert len(approved.uploads) == 35
    assert approved.inspection["audit_artifact_hashes"]
    assert {relative: (repo / relative).read_bytes() for relative in originals} == originals

    # Move the checked-out deployment as a unit, then remove external review and
    # design sources. Automatic approval resolution must be fully self-contained.
    deployment = tmp_path / "custom-deployment"
    promoted.parent.rename(deployment)
    shutil.rmtree(design)
    cases, deployed = load_release_input(deployment / promoted.name)

    assert deployed is not None
    assert cases == approved.cases
    assert deployed.candidate_bytes == approved.candidate_bytes
    assert deployed.uploads == approved.uploads
    assert deployed.inspection["artifact_hashes"] == approved.inspection["artifact_hashes"]
    assert deployed.inspection["audit_artifact_hashes"] == approved.inspection["audit_artifact_hashes"]
    assert deployed.review_sha256 == approved.review_sha256


def test_archived_v1_preserves_the_original_candidate_and_evidence_approval():
    history = REPOSITORY / BENCHMARKS / "history/release-nemo-v1"
    review = json.loads((history / "design/release_review.json").read_bytes())
    candidate = (history / "fixtures/cases.generated.jsonl").read_bytes()

    assert review["candidate_sha256"] == hashlib.sha256(candidate).hexdigest()
    assert b"\r\n" in candidate
    for logical_name, expected in review["artifact_hashes"].items():
        path = history / logical_name if logical_name.startswith("design/") else history / "fixtures" / logical_name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, logical_name
    assert hashlib.sha256((history / "design/generation_manifest.json").read_bytes()).hexdigest() == review[
        "generation_manifest_sha256"
    ]
    assert hashlib.sha256((history / "design/similarity_report.json").read_bytes()).hexdigest() == review[
        "semantic_similarity_review"
    ]["report_sha256"]

    current_review = json.loads((REPOSITORY / BENCHMARKS / "design/release_review.json").read_bytes())
    assert current_review["candidate_sha256"] != review["candidate_sha256"]
