from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from threading import Barrier

import pytest

from src.infra.tools import save_text as save_module
from src.infra import saved_artifacts
from src.core.save_contract import SaveOperation
from src.infra.saved_artifacts import ArtifactError, artifact_filename, manifest_path_for, read_saved_artifact


@pytest.mark.parametrize("concurrent", [False, True])
def test_same_time_saves_preserve_every_answer(tmp_path, monkeypatch, concurrent):
    monkeypatch.setattr(saved_artifacts, "time", lambda: 1_000_000.0)
    monkeypatch.setattr(save_module, "get_save_text_output_dir", lambda: tmp_path)
    save = save_module.build_save_text_tool()
    answers = [f"서로 다른 답변 {index}" for index in range(8)]
    barrier = Barrier(len(answers)) if concurrent else None

    def store(answer):
        if barrier is not None:
            barrier.wait(timeout=10)
        return save(answer)

    if concurrent:
        with ThreadPoolExecutor(max_workers=len(answers)) as pool:
            results = list(pool.map(store, answers))
    else:
        results = [store(answer) for answer in answers]

    assert len({result["file_path"] for result in results}) == len(answers)
    assert len(list(tmp_path.glob("*.txt"))) == len(answers)
    for answer, result in zip(answers, results, strict=True):
        assert result["status"] == "success"
        assert Path(result["file_path"]).read_text(encoding="utf-8-sig") == answer


def test_partial_write_never_publishes_a_final_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(save_module, "get_save_text_output_dir", lambda: tmp_path)
    real_write = saved_artifacts.os.write

    def fail_after_partial_write(descriptor, payload):
        real_write(descriptor, payload[:3])
        raise OSError("simulated full disk")

    monkeypatch.setattr(saved_artifacts.os, "write", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="full disk"):
        save_module.build_save_text_tool()("모든 내용이 저장되어야 하는 답변")

    assert list(tmp_path.glob("*.txt")) == []
    assert list(tmp_path.glob("*.json")) == []
    assert list(tmp_path.glob("*.part")) == []


def _operation(content="원본 답변", *, session_id="session-a", operation_id="operation-a"):
    answer_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return SaveOperation.for_text(
        content, operation_id=operation_id, session_id=session_id, request_id="request-a",
        contract_revision=1, target_kind="compose", source_hash=answer_hash, answer_hash=answer_hash,
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(save_module, "get_save_text_output_dir", lambda: tmp_path)
    return save_module.build_save_text_tool(ttl_seconds=60), tmp_path


def test_saved_artifact_binds_exact_encoded_bytes_and_retention(store, monkeypatch):
    save, root = store
    monkeypatch.setattr(saved_artifacts, "time", lambda: 100.0)
    text = "한글 답변\r\n두 번째 줄\n"
    operation = _operation(text)
    result = save(text, operation=operation)
    manifest, payload = read_saved_artifact(root, artifact_filename(operation), now_epoch=101)

    assert payload == text.encode("utf-8-sig")
    assert manifest.operation == operation
    assert manifest.artifact.created_at == 100.0
    assert manifest.artifact.expires_at == 160.0
    assert result["bytes"] == len(payload)
    assert result["operation"] == operation.model_dump(mode="json")
    assert result["artifact"] == manifest.artifact.model_dump(mode="json")
    assert result["verification"] == "verified"


@pytest.mark.parametrize("concurrent", [False, True])
def test_retries_return_one_preserved_artifact(store, concurrent):
    save, root = store
    operation = _operation()
    first = save("원본 답변", operation=operation)
    before = Path(first["file_path"]).stat().st_mtime_ns
    barrier = Barrier(8) if concurrent else None

    def retry(index):
        if barrier is not None:
            barrier.wait(timeout=10)
        return save("원본 답변", filename_prefix=f"different-hint-{index}", operation=operation)

    if concurrent:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(retry, range(8)))
    else:
        results = [retry(index) for index in range(8)]

    assert all(result == first for result in results)
    assert Path(first["file_path"]).stat().st_mtime_ns == before
    assert Path(first["file_path"]).read_bytes() == "원본 답변".encode("utf-8-sig")
    assert len(list(root.glob("*.txt"))) == len(list(root.glob("*.json"))) == 1


def test_direct_call_receipt_can_be_retried_with_its_operation(store):
    save, root = store
    first = save("원본 답변", filename_prefix="custom-name")
    operation = SaveOperation.model_validate(first["operation"])
    second = save("원본 답변", filename_prefix="another-hint", operation=operation)
    assert first == second
    assert len(list(root.glob("*.txt"))) == 1


def test_concurrent_first_attempts_for_one_operation_share_one_artifact(store):
    save, root = store
    operation = _operation()
    barrier = Barrier(8)

    def first_attempt(_):
        barrier.wait(timeout=10)
        return save("원본 답변", operation=operation)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(first_attempt, range(8)))

    assert all(result == results[0] for result in results)
    assert len(list(root.glob("*.txt"))) == 1
    assert list(root.glob("*.part")) == []


def test_same_operation_id_in_different_sessions_preserves_both_files(store):
    save, root = store
    first = save("첫 세션", operation=_operation("첫 세션", session_id="first"))
    second = save("둘째 세션", operation=_operation("둘째 세션", session_id="second"))
    assert first["file_path"] != second["file_path"]
    assert Path(first["file_path"]).read_text(encoding="utf-8-sig") == "첫 세션"
    assert Path(second["file_path"]).read_text(encoding="utf-8-sig") == "둘째 세션"
    assert len(list(root.glob("*.txt"))) == 2


@pytest.mark.parametrize("concurrent", [False, True])
def test_changed_payload_conflicts_without_overwriting_the_winner(store, concurrent):
    save, root = store
    answers = ["첫 답변", "다른 답변"]
    barrier = Barrier(2) if concurrent else None

    def attempt(text):
        if barrier is not None:
            barrier.wait(timeout=10)
        try:
            return save(text, operation=_operation(text))
        except ArtifactError as exc:
            return exc.code

    if concurrent:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, answers))
    else:
        results = [attempt(answer) for answer in answers]
    assert results.count("idempotency_conflict") == 1
    winner = next(index for index, result in enumerate(results) if isinstance(result, dict))
    manifest, payload = read_saved_artifact(root, artifact_filename(_operation()))
    assert payload == answers[winner].encode("utf-8-sig")
    assert manifest.operation == _operation(answers[winner])
    assert len(list(root.glob("*.txt"))) == 1


def test_operation_bytes_must_match_before_any_file_is_published(store):
    save, root = store
    with pytest.raises(ArtifactError) as error:
        save("틀린 내용", operation=_operation())
    assert error.value.code == "idempotency_conflict"
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("mutation", ["missing", "changed", "corrupt_manifest"])
def test_reader_rejects_missing_or_changed_artifacts(store, mutation):
    save, root = store
    result = save("원본 답변", operation=_operation())
    path = Path(result["file_path"])
    if mutation == "missing":
        path.unlink()
    elif mutation == "changed":
        path.write_bytes(b"different data")
    else:
        manifest_path_for(path).write_bytes(b"{}")
    with pytest.raises(ArtifactError) as error:
        read_saved_artifact(root, path.name)
    assert error.value.code == ("artifact_missing" if mutation == "missing" else "artifact_mismatch")


def test_retry_never_overwrites_a_changed_committed_artifact(store):
    save, root = store
    result = save("원본 답변", operation=_operation())
    path = Path(result["file_path"])
    path.write_bytes(b"external replacement")
    with pytest.raises(ArtifactError) as error:
        save("원본 답변", operation=_operation())
    assert error.value.code == "artifact_mismatch"
    assert path.read_bytes() == b"external replacement"


def test_retry_completes_manifest_reservation_after_publish_failure(store, monkeypatch):
    save, root = store
    real_link = saved_artifacts.os.link

    def fail_payload_link(source, destination):
        if Path(destination).suffix == ".txt":
            raise OSError("publication interrupted")
        return real_link(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(saved_artifacts.os, "link", fail_payload_link)
        with pytest.raises(ArtifactError, match="publication interrupted"):
            save("원본 답변", operation=_operation())
    assert len(list(root.glob("*.json"))) == 1
    assert list(root.glob("*.txt")) == []
    with pytest.raises(ArtifactError) as error:
        read_saved_artifact(root, artifact_filename(_operation()))
    assert error.value.code == "artifact_missing"

    result = save("원본 답변", operation=_operation())
    assert Path(result["file_path"]).read_bytes() == "원본 답변".encode("utf-8-sig")
    assert len(list(root.glob("*.txt"))) == 1


def test_expired_operation_cannot_be_recreated_by_retry(store, monkeypatch):
    save, root = store
    monkeypatch.setattr(saved_artifacts, "time", lambda: 100.0)
    result = save("원본 답변", operation=_operation())
    path = Path(result["file_path"])
    path.unlink()
    monkeypatch.setattr(saved_artifacts, "time", lambda: 160.0)
    with pytest.raises(ArtifactError) as error:
        save("원본 답변", operation=_operation())
    assert error.value.code == "artifact_expired"
    assert not path.exists()
    assert manifest_path_for(path).exists()


def test_fsync_failure_does_not_publish_any_artifact(store, monkeypatch):
    save, root = store

    def fail_sync(_descriptor):
        raise OSError("sync failed")

    monkeypatch.setattr(saved_artifacts.os, "fsync", fail_sync)
    with pytest.raises(ArtifactError, match="sync failed"):
        save("원본 답변", operation=_operation())
    assert list(root.glob("*.txt")) == list(root.glob("*.json")) == list(root.glob("*.part")) == []


def test_unsupported_no_replace_publication_fails_closed(store, monkeypatch):
    save, root = store

    def unsupported_link(_source, _destination):
        raise OSError("hard links are unavailable")

    monkeypatch.setattr(saved_artifacts.os, "link", unsupported_link)
    with pytest.raises(ArtifactError, match="hard links are unavailable"):
        save("원본 답변", operation=_operation())
    assert list(root.glob("*.txt")) == list(root.glob("*.json")) == list(root.glob("*.part")) == []


def test_short_writes_are_completed_before_publication(store, monkeypatch):
    save, root = store
    real_write = saved_artifacts.os.write

    def short_write(descriptor, payload):
        return real_write(descriptor, payload[:7])

    monkeypatch.setattr(saved_artifacts.os, "write", short_write)
    result = save("원본 답변", operation=_operation())
    _, payload = read_saved_artifact(root, Path(result["file_path"]).name)
    assert payload == "원본 답변".encode("utf-8-sig")


def test_readback_error_after_publication_never_returns_success_and_can_retry(store, monkeypatch):
    save, root = store
    real_read = Path.read_bytes

    def fail_final_read(path):
        if path.suffix == ".txt":
            raise PermissionError("readback unavailable")
        return real_read(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", fail_final_read)
        with pytest.raises(ArtifactError) as error:
            save("원본 답변", operation=_operation())
    assert error.value.code == "artifact_unverifiable"
    assert len(list(root.glob("*.txt"))) == 1
    result = save_module.build_save_text_tool(ttl_seconds=60)("원본 답변", operation=_operation())
    assert result["verification"] == "verified"
    assert len(list(root.glob("*.txt"))) == 1
