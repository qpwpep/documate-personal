"""Private conversion process entry point; no sessions, credentials or vector store."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from threading import Event, Thread
from typing import NoReturn

import psutil

from src.core.uploads import UploadRecord
from src.infra.document_cache import canonical_json
from src.infra.document_ingestion import ConversionPolicy, IngestionError


def _terminate_orphan() -> NoReturn:
    """The owning API is gone; stop subordinate work before exiting immediately."""
    try:
        descendants = psutil.Process(os.getpid()).children(recursive=True)
        for child in reversed(descendants):
            try:
                child.kill()
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    finally:
        os._exit(1)


def _start_parent_watchdog() -> Event:
    stopped = Event()
    parent_pid = os.getppid()
    try:
        expected_owner = None
        expected_pid = os.environ.get("DOCUMATE_CONVERSION_OWNER_PID")
        expected_created = os.environ.get("DOCUMATE_CONVERSION_OWNER_CREATED")
        if expected_pid is not None or expected_created is not None:
            if expected_pid is None or expected_created is None:
                _terminate_orphan()
            owner_pid, owner_created = int(expected_pid), float(expected_created)
            if owner_pid < 1 or not math.isfinite(owner_created) or owner_created <= 0:
                _terminate_orphan()
            expected_owner = psutil.Process(owner_pid)
            if (expected_owner.create_time() != owner_created or not expected_owner.is_running()
                    or expected_owner.status() == psutil.STATUS_ZOMBIE):
                _terminate_orphan()
        if parent_pid <= 1 and (expected_owner is None or parent_pid != expected_owner.pid):
            # A container's PID 1 API is legitimate only when the runner
            # supplied that exact owner identity; otherwise this is an orphan.
            _terminate_orphan()
        worker = psutil.Process(os.getpid())
        parent = psutil.Process(parent_pid)
        parents = [parent]
        if expected_owner is not None and expected_owner.pid != parent_pid:
            parents.append(expected_owner)
        # Windows venv Python is a redirector which waits for this interpreter.
        # Its lifetime alone cannot prove the owning API is still running.
        if (expected_owner is None and os.name == "nt" and os.path.normcase(parent.exe()) == os.path.normcase(sys.executable)
                and os.path.normcase(worker.exe()) != os.path.normcase(sys.executable)):
            owner = parent.parent()
            if owner is None:
                _terminate_orphan()
            parents.append(owner)
        identities = [(item.pid, item.create_time()) for item in parents]
        if any(created > worker.create_time() for _pid, created in identities):
            _terminate_orphan()
    except (psutil.Error, ValueError, TypeError, OverflowError):
        _terminate_orphan()

    def watch() -> None:
        while not stopped.wait(0.25):
            try:
                alive = True
                for pid, created in identities:
                    observed = psutil.Process(pid)
                    if (observed.create_time() != created or not observed.is_running()
                            or observed.status() == psutil.STATUS_ZOMBIE):
                        alive = False
                        break
            except psutil.Error:
                alive = False
            if not alive and not stopped.is_set():
                _terminate_orphan()

    Thread(target=watch, name="docling-parent-watchdog", daemon=True).start()
    return stopped


def _run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    result_path = Path(args.result)
    try:
        request = json.loads(Path(args.request).read_bytes())
        policy = ConversionPolicy.model_validate(request["policy"])
        files = [UploadRecord.model_validate(value) for value in request["files"]]
        config = request["parser_config"]
        if (not files or not isinstance(config, dict)
                or any(config.get(key) != value for key, value in policy.extraction_options().items())):
            raise IngestionError("DOCUMENT_ADAPTER_ERROR", "문서 변환 요청과 추출 설정이 일치하지 않습니다.")
        try:
            from src.infra.docling_adapter import convert_file
        except ImportError as exc:
            raise IngestionError("DOCUMENT_CONVERTER_UNAVAILABLE", "Docling 변환 의존성을 불러올 수 없습니다.", retryable=True) from exc
        documents = [convert_file(file, policy, parser_config=config) for file in files]
        payload = {"documents": [document.model_dump(mode="json") for document in documents]}
        code = 0
    except IngestionError as exc:
        payload = {"error": {"code": exc.code, "message": exc.message,
                    "file_name": exc.file_name, "retryable": exc.retryable}}
        code = 1
    except (ValueError, TypeError, KeyError):
        payload = {"error": {"code": "DOCUMENT_ADAPTER_ERROR", "message": "문서 변환 요청이나 결과를 검증하지 못했습니다."}}
        code = 1
    except Exception:
        payload = {"error": {"code": "DOCUMENT_CONVERSION_FAILED", "message": "문서 변환을 완료하지 못했습니다.", "retryable": True}}
        code = 1
    try:
        encoded = canonical_json(payload)
    except (ValueError, TypeError):
        encoded = canonical_json({"error": {"code": "DOCUMENT_ADAPTER_ERROR", "message": "문서 변환 결과를 직렬화하지 못했습니다."}})
        code = 1
    limit = locals().get("policy", ConversionPolicy()).max_output_mib * 1024 * 1024
    if len(encoded) > limit:
        encoded = canonical_json({"error": {"code": "DOCUMENT_LIMIT_EXCEEDED", "message": "변환 결과 크기 한도를 초과했습니다."}})
        code = 1
    temporary = result_path.with_name("result.part")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(result_path)
    return code


def main() -> int:
    # Start before request I/O and model imports, which may themselves block.
    stopped = _start_parent_watchdog()
    try:
        return _run()
    finally:
        stopped.set()


if __name__ == "__main__":
    raise SystemExit(main())
