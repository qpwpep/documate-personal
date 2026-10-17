from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from threading import Barrier

import pytest

from src.infra.tools import save_text as save_module
from src.infra import saved_artifacts
from src.core.save_contract import SaveOperation
from src.infra.saved_artifacts import ArtifactError, artifact_filename, read_saved_artifact


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


def _operation(content="원본 답변"):
    answer_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return SaveOperation.for_text(
        content, operation_id="operation-a", session_id="session-a", request_id="request-a",
        contract_revision=1, target_kind="compose", source_hash=answer_hash, answer_hash=answer_hash,
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(save_module, "get_save_text_output_dir", lambda: tmp_path)
    return save_module.build_save_text_tool(ttl_seconds=60), tmp_path


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


def test_changed_payload_conflicts_without_overwriting_the_winner(store):
    save, root = store
    answers = ["첫 답변", "다른 답변"]

    def attempt(text):
        try:
            return save(text, operation=_operation(text))
        except ArtifactError as exc:
            return exc.code

    results = [attempt(answer) for answer in answers]
    assert results.count("idempotency_conflict") == 1
    winner = next(index for index, result in enumerate(results) if isinstance(result, dict))
    manifest, payload = read_saved_artifact(root, artifact_filename(_operation()))
    assert payload == answers[winner].encode("utf-8-sig")
    assert manifest.operation == _operation(answers[winner])
    assert len(list(root.glob("*.txt"))) == 1


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
