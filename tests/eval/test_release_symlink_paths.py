"""Release paths keep their existing symlink and upload-directory semantics."""

from pathlib import Path
import shutil

import pytest

from src.eval.release_dataset import load_release_input, promote
from tests.eval.test_release_promotion import reviewed_candidates


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"File symlinks are unavailable: {exc}")


@pytest.mark.parametrize("same_bytes", [False, True], ids=["different-bytes", "same-bytes"])
def test_promotion_replaces_output_symlink_without_changing_its_target(tmp_path, same_bytes):
    source = tmp_path / "source"
    source.mkdir()
    candidates, review, design, _ = reviewed_candidates(source)
    target = tmp_path / "referenced-release.jsonl"
    previous_bytes = candidates.read_bytes() if same_bytes else b"Keep the referenced file unchanged.\n"
    target.write_bytes(previous_bytes)
    execution = tmp_path / "deployed-release.jsonl"
    _symlink_or_skip(execution, target)

    result = promote(candidates, review=review, design=design, out=execution)

    assert result["errors"] == []
    assert not execution.is_symlink()
    assert execution.read_bytes() == candidates.read_bytes()
    assert target.read_bytes() == previous_bytes
    assert not target.with_name(target.name + ".approvals").exists()
    _, approved = load_release_input(execution)
    assert approved is not None
    assert approved.candidate_bytes == candidates.read_bytes()


def test_explicit_review_uses_uploads_next_to_the_requested_symlink_path(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    candidates, review, design, rows = reviewed_candidates(source, with_attachment=True)
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    execution = deployment / "release.jsonl"
    _symlink_or_skip(execution, candidates)
    shutil.copytree(source / "uploads", deployment / "uploads")
    upload_name = rows[0]["upload_fixtures"][0]
    deployed_bytes = (deployment / "uploads" / upload_name).read_bytes()
    (source / "uploads" / upload_name).write_bytes(b"Changed outside the execution directory.\n")

    cases, approved = load_release_input(execution, review=review, design=design)

    assert len(cases) == 120
    assert approved is not None
    assert approved.uploads[upload_name].logical_ref == upload_name
    assert approved.uploads[upload_name].getbuffer() == deployed_bytes


@pytest.mark.parametrize("explicit_review", [False, True], ids=["automatic", "explicit"])
def test_loading_keeps_captured_candidate_when_its_symlink_is_replaced(
    tmp_path, monkeypatch, explicit_review,
):
    source = tmp_path / "source"
    source.mkdir()
    candidates, review, design, rows = reviewed_candidates(source)
    execution = tmp_path / "release.jsonl"
    promote(candidates, review=review, design=design, out=execution)
    execution.unlink()
    _symlink_or_skip(execution, candidates)
    original_bytes = candidates.read_bytes()
    lines = original_bytes.splitlines(keepends=True)
    lines[0], lines[1] = lines[1], lines[0]
    replacement_bytes = b"".join(lines)
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(replacement_bytes)
    read_bytes = Path.read_bytes

    def replace_after_capture(path):
        content = read_bytes(path)
        if path == candidates and execution.is_symlink():
            replacement.replace(execution)
        return content

    monkeypatch.setattr(Path, "read_bytes", replace_after_capture)
    options = {"review": review, "design": design} if explicit_review else {}

    cases, approved = load_release_input(execution, **options)

    assert approved is not None
    assert approved.candidate_bytes == original_bytes
    assert cases[0].case_id == rows[0]["case_id"]
    assert not execution.is_symlink()
    assert execution.read_bytes() == replacement_bytes


def test_default_approval_is_not_downgraded_when_symlink_changes_after_capture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    candidates, review, design, rows = reviewed_candidates(source)
    (design / "release_review.json").write_bytes(review.read_bytes())
    monkeypatch.setattr("src.eval.release_dataset.DEFAULT_RELEASE", candidates)
    monkeypatch.setattr("src.eval.release_dataset.DEFAULT_DESIGN", design)
    execution = tmp_path / "release.jsonl"
    _symlink_or_skip(execution, candidates)
    original_bytes = candidates.read_bytes()
    lines = original_bytes.splitlines(keepends=True)
    lines[0], lines[1] = lines[1], lines[0]
    replacement_bytes = b"".join(lines)
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(replacement_bytes)
    read_bytes = Path.read_bytes

    def replace_after_capture(path):
        content = read_bytes(path)
        if path == candidates and execution.is_symlink():
            replacement.replace(execution)
        return content

    monkeypatch.setattr(Path, "read_bytes", replace_after_capture)

    cases, approved = load_release_input(execution)

    assert approved is not None
    assert approved.candidate_bytes == original_bytes
    assert cases[0].case_id == rows[0]["case_id"]
    assert not execution.is_symlink()
    assert execution.read_bytes() == replacement_bytes
