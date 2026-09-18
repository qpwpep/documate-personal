"""Filesystem failures must preserve the previously runnable release input."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from src.eval.release_dataset import load_release_input, promote, sha256
from tests.eval.test_promoted_release_binding import promotion_package, release_http, run_release


def _next_candidate(package):
    lines = package.candidates.read_bytes().splitlines(keepends=True)
    lines[2], lines[3] = lines[3], lines[2]
    package.candidates.write_bytes(b"".join(lines))
    review = json.loads(package.review.read_bytes())
    review["candidate_sha256"] = sha256(package.candidates)
    package.review.write_bytes(json.dumps(review).encode("utf-8"))


@pytest.mark.parametrize("previous", ["legacy-default", "legacy-custom", "approved-custom"])
@pytest.mark.parametrize("failure", ["object", "registry", "fixture"])
def test_interrupted_promotion_keeps_previous_execution_and_can_retry(
    tmp_path, monkeypatch, release_http, previous, failure,
):
    package = promotion_package(tmp_path, monkeypatch, default=previous == "legacy-default")
    if previous == "legacy-custom":
        package.execution.write_bytes(package.candidates.read_bytes() + b"\n")
    if previous == "approved-custom":
        promote(package.candidates, review=package.review, out=package.execution, design=package.design)
        _next_candidate(package)
    old_bytes = package.execution.read_bytes()
    _, old_approval = load_release_input(package.execution)
    expected_review = old_approval.review_sha256 if old_approval else None
    registry_writes = 0
    original_replace, original_rename = Path.replace, Path.rename

    def replace(source, destination):
        nonlocal registry_writes
        destination = Path(destination)
        if destination.name == "bindings.json":
            registry_writes += 1
            # The first write for a legacy input records its baseline. Fail at
            # the later publication of the new candidate's selected approval.
            target_write = 1 if previous == "approved-custom" else 2
            if failure == "registry" and registry_writes == target_write:
                raise OSError("injected registry publication failure")
        if failure == "fixture" and destination == package.execution:
            raise OSError("injected fixture replacement failure")
        return original_replace(source, destination)

    def rename(source, destination):
        if failure == "object" and Path(destination).parent.name == "objects":
            raise OSError("injected approval publication failure")
        return original_rename(source, destination)

    with monkeypatch.context() as failing:
        failing.setattr(Path, "replace", replace)
        failing.setattr(Path, "rename", rename)
        with pytest.raises(OSError, match="injected"):
            promote(package.candidates, review=package.review, out=package.execution, design=package.design)
    assert package.execution.read_bytes() == old_bytes
    _, _, summary = run_release(package, tmp_path)
    approval = summary.audit_metrics["dataset_approval"]
    assert approval["status"] == ("verified" if expected_review else "not_verified")
    if expected_review:
        assert approval["review_sha256"] == expected_review

    promote(package.candidates, review=package.review, out=package.execution, design=package.design)
    _, approved = load_release_input(package.execution)
    assert approved.review_sha256 == sha256(package.review)
    assert approved.candidate_bytes == package.candidates.read_bytes()


@pytest.mark.parametrize("default", [True, False])
def test_first_registry_write_failure_does_not_manage_the_old_input(tmp_path, monkeypatch, default):
    package = promotion_package(tmp_path, monkeypatch, default=default)
    if not default:
        package.execution.write_bytes(package.candidates.read_bytes() + b"\n")
    old_bytes = package.execution.read_bytes()
    original_replace = Path.replace

    def replace(source, destination):
        if Path(destination).name == "bindings.json":
            raise OSError("injected initial registry failure")
        return original_replace(source, destination)

    with monkeypatch.context() as failing:
        failing.setattr(Path, "replace", replace)
        with pytest.raises(OSError, match="injected"):
            promote(package.candidates, review=package.review, out=package.execution, design=package.design)
    cases, approved = load_release_input(package.execution)
    assert len(cases) == 120
    assert package.execution.read_bytes() == old_bytes
    assert (approved is not None) == default


def test_same_candidate_reapproval_preserves_history_without_replacing_fixture(tmp_path, monkeypatch):
    package = promotion_package(tmp_path, monkeypatch, default=False)
    first_review = package.review.read_bytes()
    promote(package.candidates, review=package.review, out=package.execution, design=package.design)
    first = load_release_input(package.execution)[1]
    updated = json.loads(first_review)
    updated["oracle_review"] = "Second completed review of the same candidate"
    package.review.write_bytes(json.dumps(updated).encode("utf-8"))
    original_replace = Path.replace

    def replace(source, destination):
        if Path(destination) == package.execution:
            pytest.fail("Reapproval must not replace unchanged candidate bytes")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", replace)
    promote(package.candidates, review=package.review, out=package.execution, design=package.design)
    second = load_release_input(package.execution)[1]
    assert first.candidate_bytes == second.candidate_bytes
    assert first.review_sha256 != second.review_sha256 == sha256(package.review)
    store = package.execution.with_name(package.execution.name + ".approvals")
    assert (store / "objects" / first.review_sha256 / "review.json").read_bytes() == first_review
    archived = {str(path.relative_to(store)): path.read_bytes() for path in store.rglob("*") if path.is_file()}
    promote(package.candidates, review=package.review, out=package.execution, design=package.design)
    assert {str(path.relative_to(store)): path.read_bytes() for path in store.rglob("*") if path.is_file()} == archived


@pytest.mark.parametrize("changed", ["registry", "missing-registry", "review", "source", "audit"])
def test_damaged_approval_store_never_falls_back(tmp_path, monkeypatch, changed):
    package = promotion_package(tmp_path, monkeypatch, default=True)
    promote(package.candidates, review=package.review, out=package.execution, design=package.design)
    store = package.execution.with_name(package.execution.name + ".approvals")
    archived = store / "objects" / package.review_sha256
    target = {
        "registry": store / "bindings.json", "missing-registry": store / "bindings.json",
        "review": archived / "review.json", "source": archived / "design" / "sources.json",
        "audit": archived / "design" / "similarity_report.json",
    }[changed]
    if changed == "missing-registry":
        target.unlink()
    elif changed in {"registry", "review"}:
        payload = json.loads(target.read_bytes())
        payload["version" if changed == "registry" else "oracle_review"] = "tampered"
        target.write_bytes(json.dumps(payload).encode("utf-8"))
    else:
        target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="registry|approval|review|approved"):
        load_release_input(package.execution)


@pytest.mark.parametrize("phase", ["promotion", "execution"])
def test_snapshot_survives_candidate_and_review_replacement_after_read(tmp_path, monkeypatch, phase):
    package = promotion_package(tmp_path, monkeypatch, default=False)
    if phase == "execution":
        promote(package.candidates, review=package.review, out=package.execution, design=package.design)
    candidate = package.candidates if phase == "promotion" else package.execution
    expected = candidate.read_bytes()
    original_read = Path.read_bytes

    def read_then_change(path):
        content = original_read(path)
        if path == candidate:
            path.write_bytes(b"changed after the validated snapshot read\n")
        if phase == "promotion" and path == package.review:
            path.write_bytes(b"changed review after read\n")
        return content

    with monkeypatch.context() as changing:
        changing.setattr(Path, "read_bytes", read_then_change)
        if phase == "promotion":
            promote(package.candidates, review=package.review, out=package.execution, design=package.design)
        else:
            _, approved = load_release_input(package.execution)
    if phase == "promotion":
        _, approved = load_release_input(package.execution)
    assert approved.candidate_bytes == expected
    assert approved.review_sha256 == package.review_sha256


def test_explicit_default_design_still_selects_its_own_review(tmp_path, monkeypatch):
    package = promotion_package(tmp_path, monkeypatch, default=True)
    promote(package.candidates, review=package.review, out=package.execution, design=package.design)
    # This exact legacy usage selects <design>/release_review.json even though
    # the automatic archive is valid and has a different review hash.
    review = json.loads(package.review.read_bytes())
    review["oracle_review"] = "Explicit design review"
    explicit = package.design / "release_review.json"
    explicit.write_bytes(json.dumps(review).encode("utf-8"))
    _, approved = load_release_input(package.execution, design=package.design)
    assert approved.review_sha256 == sha256(explicit)


def test_unknown_bytes_cannot_use_a_pre_promotion_legacy_entry(tmp_path, monkeypatch):
    package = promotion_package(tmp_path, monkeypatch, default=False)
    package.execution.write_bytes(package.candidates.read_bytes() + b"\n")
    legacy = package.execution.read_bytes()
    promote(package.candidates, review=package.review, out=package.execution, design=package.design)
    package.execution.write_bytes(legacy)
    assert load_release_input(package.execution)[1] is None
    package.execution.write_bytes(legacy + b"\n")
    with pytest.raises(ValueError, match="no binding"):
        load_release_input(package.execution)


def test_concurrent_promotions_preserve_both_approved_versions(tmp_path, monkeypatch):
    package = promotion_package(tmp_path, monkeypatch, default=False)
    first_candidate = package.source / "first.jsonl"
    first_review = package.source / "first-review.json"
    first_candidate.write_bytes(package.candidates.read_bytes())
    first_review.write_bytes(package.review.read_bytes())
    _next_candidate(package)
    inputs = [(first_candidate, first_review), (package.candidates, package.review)]
    barrier = Barrier(2)

    def publish(candidate, review):
        barrier.wait(timeout=10)
        return promote(candidate, review=review, out=package.execution, design=package.design)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(publish, *arguments) for arguments in inputs]
        assert all(future.result(timeout=20)["errors"] == [] for future in futures)
    active = load_release_input(package.execution)[1]
    assert active.review_sha256 in {sha256(review) for _, review in inputs}
    # Either publication may win, and both exact approved inputs remain usable.
    for candidate, review in inputs:
        package.execution.write_bytes(candidate.read_bytes())
        _, approved = load_release_input(package.execution)
        assert approved.review_sha256 == sha256(review)
