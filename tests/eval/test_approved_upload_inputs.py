"""Approved attachments retain their declared names and captured contents."""

import hashlib
import json
from pathlib import Path

import pytest

from src.app.uploads import stage_uploaded_files
from src.eval.config_models import BenchmarkCase
from src.eval.nemo_generate import _sha256
from src.eval.online_runner.scenario_inputs import resolve_fixture_uploads
from src.eval.release_dataset import load_reviewed_release, release_artifact_hashes, sha256
from tests.eval.test_reviewed_release_execution import approved_package


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"File symlinks are unavailable: {exc}")


def _load_package(tmp_path):
    execution, review, design, _, upload = approved_package(tmp_path)
    approved = load_reviewed_release(execution, review=review, design=design)
    return execution, approved, upload


def _approve_nested_uploads(tmp_path, *, same_content, names):
    execution, review, design, rows, original = approved_package(tmp_path)
    first = original.read_bytes()
    contents = [first, first if same_content else b"LIMIT = 29\n"]
    for name, content in zip(names, contents, strict=True):
        path = original.parent / name
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
    specs_path = design / "retrieval_specs.jsonl"
    specs = [json.loads(line) for line in specs_path.read_text(encoding="utf-8").splitlines()]
    specs[0]["upload_fixtures"] = names
    specs[0]["oracle"]["evidence"][-1]["source"] = names[0]
    rows[0]["upload_fixtures"] = names
    rows[0]["oracle"] = specs[0]["oracle"]
    rows[0]["provenance"]["nemo_generation"]["source_spec_sha256"] = _sha256(specs[0])
    for path, records in [(specs_path, specs), (execution, rows)]:
        path.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode("utf-8"))
    payload = json.loads(review.read_bytes())
    payload["candidate_sha256"] = sha256(execution)
    payload["artifact_hashes"] = release_artifact_hashes(
        [BenchmarkCase.model_validate(row) for row in rows], design=design, fixtures_path=execution,
    )
    review.write_bytes(json.dumps(payload).encode("utf-8"))
    return execution, load_reviewed_release(execution, review=review, design=design)


def test_approved_upload_mapping_cannot_be_replaced_after_validation(tmp_path):
    execution, approved, original = _load_package(tmp_path)
    expected = original.read_bytes()

    with pytest.raises(TypeError):
        approved.uploads[original.name] = b"LIMIT = 999\n"

    uploads = resolve_fixture_uploads(execution, approved.cases[0], verified_uploads=approved.uploads)
    assert uploads[0].getbuffer() == expected
    assert uploads[0].fingerprint == hashlib.sha256(expected).hexdigest()


def test_approved_upload_value_cannot_change_its_name_or_content(tmp_path):
    execution, approved, original = _load_package(tmp_path)
    expected = original.read_bytes()
    upload = resolve_fixture_uploads(execution, approved.cases[0], verified_uploads=approved.uploads)[0]

    with pytest.raises((AttributeError, TypeError)):
        upload.content = b"LIMIT = 999\n"
    with pytest.raises((AttributeError, TypeError)):
        upload.name = "other.py"

    assert upload.name == original.name
    assert upload.getbuffer() == expected


def test_approved_symlink_uses_declared_name_instead_of_physical_target_name(tmp_path):
    execution, review, design, _, original = approved_package(tmp_path)
    expected = original.read_bytes()
    physical = original.with_name("physical-source.py")
    original.rename(physical)
    _symlink_or_skip(original, physical)
    approved = load_reviewed_release(execution, review=review, design=design)

    upload = resolve_fixture_uploads(execution, approved.cases[0], verified_uploads=approved.uploads)[0]

    assert upload.name == "settings.py"
    assert upload.getbuffer() == expected
    assert approved.inspection["artifact_hashes"]["uploads/settings.py"] == hashlib.sha256(expected).hexdigest()


def test_capture_rejects_symlink_outside_upload_root_even_with_matching_bytes(tmp_path):
    execution, review, design, _, original = approved_package(tmp_path)
    outside = tmp_path / "outside.py"
    original.rename(outside)
    _symlink_or_skip(original, outside)

    with pytest.raises(ValueError, match="outside root|missing upload"):
        load_reviewed_release(execution, review=review, design=design)


@pytest.mark.parametrize("name", [
    "", ".", "folder/", "../settings.py", "folder/../settings.py", r"folder\..\settings.py",
    "/settings.py", r"\settings.py", "C:/settings.py", "C:settings.py", r"\\server\share\settings.py",
])
def test_approved_uploads_reject_unsafe_logical_names(tmp_path, name):
    execution, approved, original = _load_package(tmp_path)
    case = BenchmarkCase(case_id="unsafe-name", category="rag_only", query="Explain", upload_fixtures=[name])
    supplied = {name: approved.uploads[original.name]}

    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_fixture_uploads(execution, case, verified_uploads=supplied)


def test_missing_approved_entry_does_not_fall_back_to_an_existing_source(tmp_path):
    execution, approved, original = _load_package(tmp_path)
    assert original.is_file()

    with pytest.raises(ValueError, match="missing"):
        resolve_fixture_uploads(execution, approved.cases[0], verified_uploads={})


def test_unverified_uploads_keep_physical_name_and_deferred_read(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    physical = root / "physical-source.py"
    physical.write_bytes(b"LIMIT = 17\n")
    alias = root / "settings.py"
    _symlink_or_skip(alias, physical)
    case = BenchmarkCase(case_id="legacy-smoke", category="rag_only", query="Explain", upload_fixtures=[alias.name])

    upload = resolve_fixture_uploads(tmp_path / "cases.jsonl", case)[0]
    physical.write_bytes(b"LIMIT = 29\n")

    assert upload.name == physical.name
    assert upload.getbuffer() == b"LIMIT = 29\n"


@pytest.mark.parametrize("same_content", [False, True])
@pytest.mark.parametrize("names", [
    ["first/settings.py", "second/settings.py"],
    ["first/Settings.py", "second/settings.py"],
    ["first/caf\u00e9.py", "second/cafe\u0301.py"],
], ids=["identical", "casefold", "unicode-normalization"])
def test_approved_logical_basename_collisions_are_rejected(tmp_path, same_content, names):
    with pytest.raises(ValueError, match="duplicate|collision"):
        execution, approved = _approve_nested_uploads(tmp_path, same_content=same_content, names=names)
        resolve_fixture_uploads(execution, approved.cases[0], verified_uploads=approved.uploads)


@pytest.mark.parametrize("same_content", [False, True])
def test_unverified_basename_collisions_keep_the_existing_staging_policy(tmp_path, same_content):
    names = ["first/settings.py", "second/settings.py"]
    contents = [b"LIMIT = 17\n", b"LIMIT = 17\n" if same_content else b"LIMIT = 29\n"]
    for name, content in zip(names, contents, strict=True):
        path = tmp_path / "uploads" / name
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
    case = BenchmarkCase(case_id="legacy-collision", category="rag_only", query="Explain", upload_fixtures=names)
    uploads = resolve_fixture_uploads(tmp_path / "cases.jsonl", case)

    staged = stage_uploaded_files(
        uploads, tmp_path / "session", existing_files=[], max_files=8, max_file_mib=1, max_total_mib=8,
    )

    if same_content:
        assert staged.errors == []
        assert len(staged.files) == 1
        assert staged.files[0].name == "settings.py"
        assert Path(staged.files[0].path).read_bytes() == contents[0]
        assert staged.unchanged_names == ["settings.py"]
    else:
        assert staged.errors
        assert staged.files == []
        assert not (tmp_path / "session").exists()
