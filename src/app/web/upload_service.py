"""Commit a complete session attachment set only after its index is ready."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import HTTPException

from src.core.uploads import (
    UploadAddition, UploadContext, UploadManifest, UploadRecord, UploadSyncRequest,
    UploadSyncResponse, normalized_upload_name, validate_session_id,
)
from src.infra.runtime_paths import get_project_root_path, get_upload_session_dir, get_uploads_dir
from src.infra.settings import AppSettings
from src.core.upload_formats import CODE_UPLOAD_SUFFIXES, enabled_upload_suffixes
from src.infra.document_ingestion import ConversionPolicy, DocumentConverterPort, DocumentIngestionContext, IngestionError
from src.infra.tools.local_rag import build_upload_retriever
from src.infra.tools.local_rag.uploads import retry_pending_upload_index_cleanup
from src.infra.upload_storage import UploadStorage, reconcile_managed_upload_files, remove_managed_upload_files, clear_auxiliary_upload_files

if TYPE_CHECKING:
    from src.app.agent_manager import AgentFlowManager
    from src.app.web.session_store import InMemorySessionStore
    from src.runtime.agent_runtime.session_context import SessionContext


logger = logging.getLogger(__name__)
_MIB = 1024 * 1024


def _error(status: int, code: str, message: str, *, files: list[dict] | None = None) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message, "files": files or []})


class UploadService:
    def __init__(self, *, settings: AppSettings, session_store: InMemorySessionStore,
                 converter: DocumentConverterPort | None = None):
        self.settings = settings
        self.session_store = session_store
        self.converter = converter

    def get_manifest(self, session_id: str) -> UploadManifest:
        session_id = validate_session_id(session_id)
        with self.session_store.locked_session(session_id) as (entry, _wait_ms):
            self._cleanup_staging(session_id)
            session = entry.agent._ensure_session()
            self._cleanup_resources(session_id, session)
            return session.upload_manifest()

    @staticmethod
    def check_context(session: SessionContext, context: UploadContext) -> None:
        if context.epoch != session.upload_epoch or context.revision != session.upload_revision:
            raise _error(409, "UPLOAD_REVISION_CONFLICT", "첨부 목록이 변경되었거나 세션이 만료되었습니다. 목록을 새로고침해 주세요.")

    def sync(self, session_id: str, request: UploadSyncRequest) -> UploadSyncResponse:
        session_id = validate_session_id(session_id)
        with self.session_store.locked_session(session_id) as (entry, _wait_ms):
            session = entry.agent._ensure_session()
            self._cleanup_staging(session_id, protected_paths=tuple(item.path for item in request.add))
            self._cleanup_resources(session_id, session, protected_paths=tuple(item.path for item in request.add))
            fingerprint = hashlib.sha256(request.model_dump_json(exclude={"operation_id"}).encode()).hexdigest()
            previous = session.upload_operations.get(request.operation_id)
            if previous is not None:
                if previous[0] != fingerprint:
                    raise _error(409, "UPLOAD_OPERATION_CONFLICT", "이미 사용한 작업 ID의 요청 내용을 변경할 수 없습니다.")
                return previous[1].model_copy(deep=True)
            self.check_context(session, UploadContext(epoch=request.epoch, revision=request.expected_revision))
            result = self._sync_locked(session_id, session, request)
            session.upload_operations[request.operation_id] = (fingerprint, result.model_copy(deep=True))
            while len(session.upload_operations) > 64:
                session.upload_operations.popitem(last=False)
            return result

    @staticmethod
    def _cleanup_resources(
        session_id: str, session: SessionContext, *, protected_paths: tuple[str, ...] = (),
    ) -> None:
        """The caller already holds the pinned session lock; never reacquire it here."""
        try:
            storage = session.bind_upload_storage(session_id)
        except (OSError, RuntimeError, ValueError) as exc:
            raise _error(422, "UPLOAD_PATH_INVALID", "세션 저장 경로가 업로드 영역을 벗어났습니다.") from exc
        protected = []
        for value in protected_paths:
            path = Path(value).expanduser()
            protected.append(str(path if path.is_absolute() else get_project_root_path() / path))
        retry_pending_upload_index_cleanup()
        reconcile_managed_upload_files(storage, retained_paths=(item.path for item in session.upload_records),
                                       protected_paths=protected)
        clear_auxiliary_upload_files(storage, include_cache=False)

    def _cleanup_staging(self, session_id: str, *, protected_paths: tuple[str, ...] = ()) -> None:
        """Reclaim expired, unreferenced batches while holding the pinned session lock.

        The same lock covers indexing, so another request cannot clean a running
        build's inputs. Streamlit writers use fresh directories; the newest child
        or parent mtime protects recent writes, including unfinished .upload-part.
        """
        try:
            root = get_upload_session_dir(session_id).resolve()
            staging = root / "staging"
            if (not root.is_relative_to(get_uploads_dir().resolve()) or staging.is_symlink()
                    or staging.is_junction() or not staging.resolve().is_relative_to(root)
                    or not staging.is_dir()):
                return
            staging = staging.resolve()
            protected: set[Path] = set()
            for value in protected_paths:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = get_project_root_path() / path
                protected.add(path.resolve())
            cutoff = time.time() - self.settings.session_ttl_seconds
            folders = list(staging.iterdir())
        except (OSError, RuntimeError, ValueError):
            logger.warning("upload_staging_cleanup_scan_failed", exc_info=True)
            return

        for folder in folders:
            try:
                if folder.is_symlink() or folder.is_junction() or not folder.is_dir():
                    continue
                resolved = folder.resolve()
                if not resolved.is_relative_to(staging) or any(path.is_relative_to(resolved) for path in protected):
                    continue
                files = list(folder.iterdir())
                # Staged batches are flat; unknown nested structures and all
                # redirects are left untouched rather than recursively followed.
                if any(path.is_symlink() or path.is_junction() or not path.is_file()
                       or not path.resolve().is_relative_to(resolved) for path in files):
                    continue
                if max(path.stat().st_mtime for path in [folder, *files]) >= cutoff:
                    continue
                for path in files:
                    if (folder.resolve() != resolved or path.is_symlink() or path.is_junction()
                            or not path.resolve().is_relative_to(resolved) or path.stat().st_mtime >= cutoff):
                        break
                    path.unlink()
                else:
                    if folder.resolve() == resolved:
                        folder.rmdir()
            except (OSError, RuntimeError, ValueError):
                # Housekeeping failure must not change an attachment operation's
                # result; a later request can retry the remaining expired files.
                logger.warning("upload_staging_cleanup_failed", exc_info=True)

    def _read_addition(self, addition: UploadAddition, session_id: str) -> tuple[bytes, str]:
        from src.app.web.cleanup import validate_upload_file_path

        name = addition.name
        if (name in {".", ".."} or name != name.strip() or name.endswith(".")
                or re.search(r'[\\/:*?"<>|\x00-\x1f]', name)
                or re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", name, re.I)):
            raise _error(422, "UPLOAD_NAME_INVALID", "사용할 수 없는 파일 이름입니다.")
        if Path(name).suffix.casefold() not in enabled_upload_suffixes(docling_enabled=self.settings.docling_enabled):
            raise _error(422, "UPLOAD_TYPE_INVALID", "지원하지 않거나 활성화되지 않은 첨부 형식입니다.")
        try:
            validated = validate_upload_file_path(addition.path, session_id)
        except HTTPException as exc:
            if "Upload file not found" in str(exc.detail):
                raise _error(422, "UPLOAD_PATH_INVALID", "임시 첨부 파일을 찾을 수 없습니다. 파일을 다시 첨부해 주세요.") from exc
            raise
        if validated is None:
            raise _error(422, "UPLOAD_PATH_INVALID", "파일 경로가 필요합니다.")
        limit = self.settings.upload_max_file_mib * _MIB
        path = Path(validated)
        if path.stat().st_size > limit:
            raise _error(413, "UPLOAD_FILE_TOO_LARGE", f"파일당 {self.settings.upload_max_file_mib} MiB까지 첨부할 수 있습니다.")
        # Bound the read as well as stat: a file can grow between those operations.
        with path.open("rb") as stream:
            content = stream.read(limit + 1)
        if len(content) > limit:
            raise _error(413, "UPLOAD_FILE_TOO_LARGE", f"파일당 {self.settings.upload_max_file_mib} MiB까지 첨부할 수 있습니다.")
        if not content:
            raise _error(422, "UPLOAD_EMPTY", "빈 파일은 검색할 수 없습니다.")
        return content, "sha256:" + hashlib.sha256(content).hexdigest()

    def _check_limits(self, sizes: list[int]) -> None:
        if len(sizes) > self.settings.upload_max_files:
            raise _error(413, "UPLOAD_TOO_MANY_FILES", f"한 세션에 {self.settings.upload_max_files}개까지 첨부할 수 있습니다.")
        if sum(sizes) > self.settings.upload_max_total_mib * _MIB:
            raise _error(413, "UPLOAD_TOTAL_TOO_LARGE", f"첨부 합계는 {self.settings.upload_max_total_mib} MiB를 넘을 수 없습니다.")

    def _sync_locked(self, session_id: str, session: SessionContext, request: UploadSyncRequest) -> UploadSyncResponse:
        existing = {item.file_id: item for item in session.upload_records}
        unknown = set(request.remove).difference(existing)
        if unknown:
            raise _error(409, "UPLOAD_FILE_NOT_FOUND", "삭제할 파일이 현재 첨부 목록에 없습니다.")
        if request.clear:
            changed = bool(existing or session.upload_retriever_handle is not None)
            if changed:
                self._commit(session_id, session, [], None)
            else:
                # A failed first attachment may have populated caches without
                # ever publishing a manifest. Explicit clear releases those too.
                clear_auxiliary_upload_files(session.bind_upload_storage(session_id))
            return UploadSyncResponse(manifest=session.upload_manifest(), changed=changed)

        records = [item for item in session.upload_records if item.file_id not in request.remove]
        pending: dict[str, tuple[UploadAddition, bytes, str, str]] = {}
        pending_bytes = 0
        batch_hashes: dict[str, str] = {}
        unchanged: list[str] = []
        failures: list[dict] = []
        failure_status = 422
        name_to_id = {normalized_upload_name(item.name): item.file_id for item in records}
        hashes = {item.file_id: item.content_hash for item in records}
        for addition in request.add:
            try:
                content, digest = self._read_addition(addition, session_id)
                key = normalized_upload_name(addition.name)
                if key in batch_hashes and batch_hashes[key] != digest:
                    raise _error(409, "UPLOAD_NAME_CONFLICT", "한 번에 같은 이름의 서로 다른 파일을 추가할 수 없습니다.")
                batch_hashes[key] = digest
                target = addition.replace_file_id
                if target is not None:
                    if target not in existing or target in request.remove:
                        raise _error(409, "UPLOAD_FILE_NOT_FOUND", "교체할 파일이 현재 첨부 목록에 없습니다.")
                    if key != normalized_upload_name(existing[target].name):
                        raise _error(422, "UPLOAD_REPLACEMENT_NAME_MISMATCH", "교체 파일의 이름은 기존 파일과 같아야 합니다.")
                owner = name_to_id.get(key)
                if owner is not None and hashes[owner] == digest:
                    unchanged.append(addition.name)
                    continue
                if owner is not None and target != owner:
                    raise _error(409, "UPLOAD_NAME_CONFLICT", "같은 이름의 파일이 있습니다. 기존 파일 교체를 선택해 주세요.")
                file_id = target or uuid4().hex
                source_uri = existing[target].source_uri if target else f"upload:///{session_id}/{file_id}"
                pending_bytes += len(content) - (len(pending[file_id][1]) if file_id in pending else 0)
                pending[file_id] = (addition, content, digest, source_uri)
                name_to_id[key] = file_id
                hashes[file_id] = digest
            except (HTTPException, OSError, ValueError) as exc:
                detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
                info = detail if isinstance(detail, dict) else {"code": "UPLOAD_PATH_INVALID", "message": str(detail)}
                failures.append({"name": addition.name, "code": info["code"], "message": info["message"]})
                if isinstance(exc, HTTPException):
                    failure_status = exc.status_code
            # Do not catch this batch-wide stop and continue reading more files.
            if pending_bytes > self.settings.upload_max_total_mib * _MIB:
                raise _error(413, "UPLOAD_TOTAL_TOO_LARGE", f"첨부 합계는 {self.settings.upload_max_total_mib} MiB를 넘을 수 없습니다.")
        if failures:
            code = failures[0]["code"] if len({item["code"] for item in failures}) == 1 else "UPLOAD_VALIDATION_FAILED"
            raise _error(failure_status, code, "첨부 변경을 반영하지 않았습니다. 파일별 오류를 확인해 주세요.", files=failures)

        sizes = [len(pending[item.file_id][1]) if item.file_id in pending else item.size_bytes for item in records]
        sizes.extend(len(value[1]) for file_id, value in pending.items() if file_id not in existing)
        self._check_limits(sizes)
        if not pending and not request.remove:
            return UploadSyncResponse(manifest=session.upload_manifest(), changed=False, unchanged_names=unchanged)

        created: list[UploadRecord] = []
        handle = None
        try:
            if pending:
                self._check_storage_capacity(session_id, pending_bytes)
            for file_id, (addition, content, digest, source_uri) in pending.items():
                created.append(self._store_content(session_id, file_id, addition.name, content, digest, source_uri))
            created_by_id = {item.file_id: item for item in created}
            candidate = [created_by_id.get(item.file_id, item) for item in records]
            candidate.extend(item for item in created if item.file_id not in existing)
            handle = self._build_candidate(candidate, session_id)
            self._commit(session_id, session, candidate, handle)
        except BaseException:
            if handle is not None and handle is not session.upload_retriever_handle:
                session._release_upload_handle(handle)
            self._remove_managed_files(session_id, created)
            raise
        return UploadSyncResponse(manifest=session.upload_manifest(), changed=True, unchanged_names=unchanged)

    def _build_candidate(self, records: list[UploadRecord], session_id: str):
        if not records:
            return None
        try:
            generation = uuid4().hex
            options = {}
            if self.settings.docling_enabled:
                storage = UploadStorage.bind(session_id)
                artifacts = Path(self.settings.docling_artifacts_path).expanduser()
                if not artifacts.is_absolute():
                    artifacts = get_project_root_path() / artifacts
                policy = ConversionPolicy(artifacts_path=str(artifacts),
                    ocr_engine=self.settings.docling_ocr_engine, do_ocr=self.settings.docling_ocr_enabled,
                    max_pdf_pages=self.settings.docling_max_pages, timeout_seconds=self.settings.docling_timeout_seconds,
                    max_worker_mib=self.settings.docling_max_worker_mib, max_output_mib=self.settings.docling_max_output_mib,
                    max_image_pixels=self.settings.docling_max_image_pixels)
                options["ingestion"] = DocumentIngestionContext(converter=self.converter, policy=policy,
                    workspace=storage.root / "conversions" / generation,
                    cache_dir=storage.root / "cache" if self.settings.document_cache_enabled else None,
                    cache_max_bytes=self.settings.document_cache_max_mib * _MIB // 2,
                    cache_ttl_seconds=self.settings.document_cache_ttl_seconds,
                    max_chunks=self.settings.document_max_chunks,
                    deadline=time.monotonic() + self.settings.document_upload_timeout_seconds)
            handle = build_upload_retriever(records, session_id=session_id, generation=generation,
                                          api_key=self.settings.openai_api_key, **options)
            try:
                if "ingestion" in options:
                    options["ingestion"].check_deadline()
            except BaseException:
                try:
                    handle.cleanup()
                except Exception:
                    logger.warning("expired_candidate_cleanup_failed", exc_info=True)
                raise
            return handle
        except IngestionError as exc:
            status = (504 if exc.code == "DOCUMENT_PROCESSING_TIMEOUT" else
                      413 if exc.code == "DOCUMENT_LIMIT_EXCEEDED" else
                      503 if exc.code in {"DOCUMENT_CONVERTER_BUSY", "DOCUMENT_CONVERTER_UNAVAILABLE", "DOCUMENT_CONVERSION_FAILED"} else
                      500 if exc.code == "DOCUMENT_ADAPTER_ERROR" else 422)
            raise _error(status, exc.code, f"{exc.message} 기존 첨부는 유지됩니다.",
                         files=[{"name": exc.file_name, "code": exc.code, "message": exc.message}] if exc.file_name else []) from exc
        except ValueError as exc:
            raise _error(422, "UPLOAD_VALIDATION_FAILED", f"파일 내용을 확인해 주세요. 첨부는 변경되지 않았습니다: {exc}") from exc
        except Exception as exc:
            logger.warning("upload_index_build_failed", exc_info=True)
            raise _error(503, "UPLOAD_INDEX_FAILED", "검색 인덱스를 만들지 못했습니다. 기존 첨부는 유지됩니다. 잠시 후 다시 시도해 주세요.") from exc

    def _check_storage_capacity(self, session_id: str, additional: int) -> None:
        root = get_upload_session_dir(session_id).resolve()
        if not root.is_relative_to(get_uploads_dir().resolve()):
            raise _error(422, "UPLOAD_PATH_INVALID", "세션 저장 경로가 업로드 영역을 벗어났습니다.")
        used = 0
        for area in (root / "objects", root / "staging", root / "conversions"):
            if not area.resolve().is_relative_to(root):
                raise _error(422, "UPLOAD_PATH_INVALID", "첨부 저장 경로가 세션 영역을 벗어났습니다.")
            if area.exists():
                used += sum(path.stat().st_size for path in area.rglob("*") if path.is_file())
        if used + additional > self.settings.upload_max_total_mib * _MIB * 3:
            raise _error(413, "UPLOAD_STORAGE_FULL", "임시 업로드 공간이 가득 찼습니다. 사용하지 않는 첨부를 정리하거나 새 채팅을 시작해 주세요.")

    @staticmethod
    def _store_content(session_id: str, file_id: str, name: str, content: bytes, digest: str, source_uri: str) -> UploadRecord:
        root = get_upload_session_dir(session_id).resolve()
        folder = root / "objects" / file_id / uuid4().hex
        if (not root.is_relative_to(get_uploads_dir().resolve())
                or not folder.resolve().is_relative_to(root)):
            raise _error(422, "UPLOAD_PATH_INVALID", "첨부 저장 경로가 세션 영역을 벗어났습니다.")
        folder.mkdir(parents=True, exist_ok=False)
        target = folder / name
        temporary = folder / ".upload-part"
        if not all(path.resolve().is_relative_to(root) for path in (folder, target, temporary)):
            raise _error(422, "UPLOAD_PATH_INVALID", "첨부 저장 경로가 세션 영역을 벗어났습니다.")
        try:
            with temporary.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            folder.rmdir()
            raise
        return UploadRecord(file_id=file_id, name=name, size_bytes=len(content), content_hash=digest,
                            source_uri=source_uri, path=str(target))

    @staticmethod
    def _remove_managed_files(session_id: str, records: list[UploadRecord]) -> None:
        try:
            storage = UploadStorage.bind(session_id)
        except (OSError, RuntimeError, ValueError):
            logger.warning("upload_file_cleanup_escaped_session", exc_info=True)
            return
        remove_managed_upload_files(storage, (item.path for item in records))

    def _commit(self, session_id: str, session: SessionContext, records: list[UploadRecord], handle) -> None:
        session.replace_upload_resources(session_id, records, handle)

    def sync_legacy_locked(self, session_id: str, agent: AgentFlowManager, path: str | None) -> UploadContext:
        """The old path replaces the set; absent/null still clears it."""
        session_id = validate_session_id(session_id)
        session = agent._ensure_session()
        self._cleanup_resources(session_id, session, protected_paths=(path,) if path else ())
        if not path:
            if session.upload_records or session.upload_retriever_handle is not None:
                self._commit(session_id, session, [], None)
            return session.upload_manifest().context()
        from src.app.web.cleanup import validate_upload_file_path

        validated = validate_upload_file_path(path, session_id)
        if Path(validated).suffix.casefold() not in CODE_UPLOAD_SUFFIXES:
            raise _error(422, "UPLOAD_TYPE_INVALID", "PDF·DOCX·이미지는 첨부 목록 API로 추가해 주세요.")
        addition = UploadAddition(path=validated, name=Path(validated).name)
        content, digest = self._read_addition(addition, session_id)
        previous = next((item for item in session.upload_records if item.source_uri == validated), None)
        if previous is not None and len(session.upload_records) == 1 and previous.content_hash == digest:
            return session.upload_manifest().context()
        self._check_limits([len(content)])
        self._check_storage_capacity(session_id, len(content))
        record = self._store_content(session_id, previous.file_id if previous else uuid4().hex,
                                     addition.name, content, digest, validated)
        handle = None
        try:
            handle = self._build_candidate([record], session_id)
            self._commit(session_id, session, [record], handle)
        except BaseException:
            if handle is not None and handle is not session.upload_retriever_handle:
                session._release_upload_handle(handle)
            self._remove_managed_files(session_id, [record])
            raise
        return session.upload_manifest().context()
