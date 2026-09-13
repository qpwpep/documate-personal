"""Request-owned conversion processes with session-local, identity-free cached results."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from threading import BoundedSemaphore, Event, Lock

import psutil

from src.core.documents import ParsedDocument
from src.core.uploads import UploadRecord
from src.infra.document_cache import ConversionCache, canonical_json, checked_directory
from src.infra.document_ingestion import ConversionPolicy, IngestionError, read_verified_upload, validate_document_input
from src.infra.runtime_paths import get_project_root_path


logger = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def _model_digest(path: str, size: int, modified: int, changed: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conversion_identity(policy: ConversionPolicy) -> dict:
    versions = {}
    for name in ("docling", "docling-core", "docling-parse", "docling-ibm-models", "torch",
                 "easyocr", "rapidocr", "onnxruntime"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    if versions["docling"] == "unavailable":
        raise IngestionError("DOCUMENT_CONVERTER_UNAVAILABLE", "Docling 선택 의존성이 설치되지 않았습니다.", retryable=True)
    root = Path(policy.artifacts_path).expanduser()
    if not policy.artifacts_path or not root.is_dir():
        raise IngestionError("DOCUMENT_CONVERTER_UNAVAILABLE", "문서 변환 모델을 먼저 준비해 주세요.", retryable=True)
    assets = []
    # Exclude download locks and caches; hash model contents and configuration, not local paths.
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or path.suffix in {".lock", ".tmp", ".part"} or ".cache" in relative.parts:
            continue
        stat = path.stat()
        assets.append((relative.as_posix(), _model_digest(str(path), stat.st_size,
                                                                      stat.st_mtime_ns, stat.st_ctime_ns)))
    if not assets:
        raise IngestionError("DOCUMENT_CONVERTER_UNAVAILABLE", "문서 변환 모델 디렉터리가 비어 있습니다.", retryable=True)
    return {**policy.extraction_options(), "runtime_versions": versions,
            "models_digest": hashlib.sha256(canonical_json(assets)).hexdigest()}


def _stop_process(process: subprocess.Popen) -> None:
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.Error:
        descendants = []
    for child in reversed(descendants):
        try:
            child.kill()
        except psutil.Error:
            pass
    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    process.wait(timeout=10)
    if descendants:
        _, alive = psutil.wait_procs(descendants, timeout=5)
        if alive:
            raise subprocess.TimeoutExpired("document worker descendants", 5)


class DoclingRunner:
    def __init__(self, *, cache_max_bytes: int = 64 * 1024 * 1024, cache_ttl_seconds: int = 1800,
                 command: Sequence[str] | None = None,
                 identity_provider: Callable[[ConversionPolicy], dict] = conversion_identity):
        self.cache_max_bytes = cache_max_bytes
        self.cache_ttl_seconds = cache_ttl_seconds
        self._command = list(command) if command is not None else [sys.executable, "-m", "src.infra.docling_worker"]
        self._identity_provider = identity_provider
        self._slot = BoundedSemaphore(1)
        self._lock = Lock()
        self._closed = Event()
        self._active: dict[int, subprocess.Popen] = {}

    @property
    def active_process_count(self) -> int:
        with self._lock:
            return len(self._active)

    def close(self) -> None:
        with self._lock:
            self._closed.set()
            processes = list(self._active.values())
        for process in processes:
            try:
                _stop_process(process)
            except (OSError, subprocess.SubprocessError, psutil.Error):
                logger.warning("document_worker_shutdown_failed")

    def _check_request(self, expires: float) -> None:
        if self._closed.is_set():
            raise IngestionError("DOCUMENT_CONVERTER_UNAVAILABLE", "문서 변환기가 종료되었습니다.", retryable=True)
        if time.monotonic() >= expires:
            raise IngestionError("DOCUMENT_PROCESSING_TIMEOUT", "문서 처리 시간이 제한을 초과했습니다.", retryable=True)

    def convert_many(self, files: Sequence[UploadRecord], *, policy: ConversionPolicy,
                     workspace: Path, cache_dir: Path | None = None,
                     deadline: float | None = None) -> tuple[ParsedDocument, ...]:
        if not files:
            return ()
        if self._closed.is_set() or not self._slot.acquire(blocking=False):
            raise IngestionError("DOCUMENT_CONVERTER_BUSY", "다른 문서를 처리 중입니다. 잠시 후 다시 시도해 주세요.", retryable=True)
        process = None
        root = None
        started = time.monotonic()
        expires = min(deadline if deadline is not None else float("inf"), started + policy.timeout_seconds)
        try:
            self._check_request(expires)
            config = self._identity_provider(policy)
            self._check_request(expires)
            originals = {file.file_id: read_verified_upload(file) for file in files}
            for file in files:
                self._check_request(expires)
                validate_document_input(file, originals[file.file_id], policy)
            self._check_request(expires)
            cache = (ConversionCache(cache_dir, max_bytes=self.cache_max_bytes, ttl_seconds=self.cache_ttl_seconds)
                     if cache_dir is not None else None)
            documents: dict[str, ParsedDocument] = {}
            output_bytes = len(canonical_json({"documents": []}))

            def retain(file: UploadRecord, document: ParsedDocument) -> None:
                nonlocal output_bytes
                self._check_request(expires)
                self._validate(document, file, config, policy)
                output_bytes += len(canonical_json(document.model_dump(mode="json"))) + int(bool(documents))
                if output_bytes > policy.max_output_mib * 1024 * 1024:
                    raise IngestionError("DOCUMENT_LIMIT_EXCEEDED", "변환된 문서의 합계 출력 한도를 초과했습니다.")
                documents[file.file_id] = document

            missing = []
            for file in files:
                self._check_request(expires)
                cached = cache.get(file, originals[file.file_id], config) if cache is not None else None
                if cached is None:
                    missing.append(file)
                else:
                    retain(file, cached)
            if missing:
                root = checked_directory(workspace)
                if list(root.iterdir()):
                    raise IngestionError("DOCUMENT_ADAPTER_ERROR", "변환 작업 공간이 비어 있지 않습니다.")
                request_path, result_path = root / "request.json", root / "result.json"
                request_path.write_bytes(canonical_json({"files": [file.model_dump(mode="json") for file in missing],
                    "policy": policy.model_dump(mode="json"), "parser_config": config}))
                env = dict(os.environ)
                for name in ("OPENAI_API_KEY", "TAVILY_API_KEY", "SLACK_BOT_TOKEN"):
                    env.pop(name, None)
                env.update(PYTHONPATH=str(get_project_root_path()), PYTHONUTF8="1", HF_HUB_OFFLINE="1",
                           HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                           DOCUMATE_CONVERSION_OWNER_PID=str(os.getpid()),
                           DOCUMATE_CONVERSION_OWNER_CREATED=str(psutil.Process(os.getpid()).create_time()))
                with (root / "worker.log").open("wb") as output:
                    with self._lock:
                        # Close and process registration share this lock: close
                        # cannot miss a worker which starts immediately after it.
                        self._check_request(expires)
                        process = subprocess.Popen([*self._command, "--request", str(request_path), "--result", str(result_path)],
                            cwd=get_project_root_path(), env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        self._active[process.pid] = process
                    while process.poll() is None:
                        self._check_request(expires)
                        try:
                            parent = psutil.Process(process.pid)
                            rss = parent.memory_info().rss + sum(child.memory_info().rss for child in parent.children(recursive=True))
                            if rss > policy.max_worker_mib * 1024 * 1024:
                                raise IngestionError("DOCUMENT_LIMIT_EXCEEDED", "문서 변환 메모리 한도를 초과했습니다.")
                        except psutil.Error:
                            pass
                        if (root / "worker.log").stat().st_size > policy.max_output_mib * 1024 * 1024:
                            raise IngestionError("DOCUMENT_LIMIT_EXCEEDED", "문서 변환 출력 한도를 초과했습니다.")
                        try:
                            process.wait(timeout=0.1)
                        except subprocess.TimeoutExpired:
                            pass
                self._check_request(expires)
                if (root / "worker.log").stat().st_size > policy.max_output_mib * 1024 * 1024:
                    raise IngestionError("DOCUMENT_LIMIT_EXCEEDED", "문서 변환 출력 한도를 초과했습니다.")
                if not result_path.is_file() or result_path.is_symlink():
                    raise IngestionError("DOCUMENT_CONVERSION_FAILED", "문서 변환 프로세스가 결과 없이 종료되었습니다.", retryable=True)
                if result_path.stat().st_size > policy.max_output_mib * 1024 * 1024:
                    raise IngestionError("DOCUMENT_LIMIT_EXCEEDED", "변환된 문서가 출력 한도를 초과했습니다.")
                result = json.loads(result_path.read_bytes())
                if not isinstance(result, dict):
                    raise ValueError("Document worker result must be an object")
                if "error" in result:
                    failure = result["error"]
                    if (not isinstance(failure, dict) or not isinstance(failure.get("code"), str)
                            or not isinstance(failure.get("message"), str)
                            or not isinstance(failure.get("file_name", ""), str)
                            or type(failure.get("retryable", False)) is not bool):
                        raise ValueError("Document worker error is malformed")
                    raise IngestionError(failure["code"], failure["message"],
                        file_name=failure.get("file_name", ""), retryable=bool(failure.get("retryable")))
                if (process.returncode or not isinstance(result.get("documents"), list)
                        or len(result["documents"]) != len(missing)):
                    raise IngestionError("DOCUMENT_ADAPTER_ERROR", "변환 결과의 파일 목록이 일치하지 않습니다.")
                parsed = [ParsedDocument.model_validate(value) for value in result["documents"]]
                for file, document in zip(missing, parsed, strict=True):
                    retain(file, document)
                self._check_request(expires)
                for file, document in zip(missing, parsed, strict=True):
                    if cache is not None:
                        cache.put(file, document)
            self._check_request(expires)
            logger.info("document_conversion_complete files=%s cache_hits=%s elapsed_ms=%s", len(files),
                        len(files) - len(missing), int((time.monotonic() - started) * 1000))
            return tuple(documents[file.file_id] for file in files)
        except IngestionError:
            raise
        except (ValueError, TypeError, KeyError) as exc:
            raise IngestionError("DOCUMENT_ADAPTER_ERROR", "문서 변환 결과를 검증하지 못했습니다.") from exc
        except OSError as exc:
            raise IngestionError("DOCUMENT_CONVERSION_FAILED", "문서 변환 자원을 준비하지 못했습니다.", retryable=True) from exc
        finally:
            try:
                if process is not None:
                    try:
                        _stop_process(process)
                    except (OSError, subprocess.SubprocessError, psutil.Error) as exc:
                        # Preserve the conversion result/error when reaping an
                        # already stopped process fails. If it may still run,
                        # close admission before releasing the concurrency slot.
                        if process.poll() is None or isinstance(exc, subprocess.TimeoutExpired):
                            self._closed.set()
                        logger.warning("document_worker_cleanup_failed")
                    finally:
                        with self._lock:
                            if process.poll() is not None:
                                self._active.pop(process.pid, None)
                if root is not None:
                    try:
                        if checked_directory(root) == root:
                            for name in ("request.json", "result.json", "worker.log", "result.part"):
                                path = root / name
                                if not path.is_symlink() and not path.is_junction():
                                    path.unlink(missing_ok=True)
                            root.rmdir()
                    except (OSError, ValueError):
                        logger.warning("document_workspace_cleanup_failed")
            finally:
                self._slot.release()

    @staticmethod
    def _validate(document: ParsedDocument, file: UploadRecord, config: dict, policy: ConversionPolicy) -> None:
        snapshot = document.snapshot
        if (snapshot.content_hash != file.content_hash or snapshot.source_uri != file.source_uri
                or snapshot.title != file.name or snapshot.parser_config != config
                or snapshot.capture_scope != "full_document"):
            raise IngestionError("DOCUMENT_ADAPTER_ERROR", "변환 결과가 원본 식별자와 일치하지 않습니다.", file_name=file.name)
        pages = {anchor.page_no for element in document.elements for anchor in element.anchors if anchor.page_no is not None}
        page_count = document.elements[0].metadata.get("document_page_count") if document.elements else None
        if page_count is not None and (type(page_count) is not int or page_count < 1):
            raise IngestionError("DOCUMENT_ADAPTER_ERROR", "변환 결과의 페이지 수가 올바르지 않습니다.", file_name=file.name)
        if max([page_count or 0, *pages]) > policy.max_pdf_pages:
            raise IngestionError("DOCUMENT_LIMIT_EXCEEDED", "문서 페이지 한도를 초과했습니다.", file_name=file.name)
