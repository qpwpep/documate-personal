"""Bounded session-local conversion cache; cached content never owns source identities."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from src.core.documents import ParsedDocument, build_snapshot
from src.core.uploads import UploadRecord


logger = logging.getLogger(__name__)


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def checked_directory(path: Path) -> Path:
    """Reject filesystem redirects before creating or reading an owned cache area."""
    path = path.absolute()
    if any(parent.is_symlink() or parent.is_junction() for parent in (path, *path.parents)):
        raise ValueError("Document cache directory is redirected")
    if path.resolve() != path:
        raise ValueError("Document cache directory is not canonical")
    path.mkdir(parents=True, exist_ok=True)
    return path


class ConversionCache:
    def __init__(self, cache_dir: Path, *, max_bytes: int, ttl_seconds: int,
                 clock: Callable[[], float] = time.time):
        self.directory = cache_dir.absolute()
        self.max_bytes = max(0, max_bytes)
        self.ttl_seconds = max(0, ttl_seconds)
        self.clock = clock

    @staticmethod
    def _key(file: UploadRecord, parser_config: dict) -> str:
        return hashlib.sha256(canonical_json({"schema": 1, "content_hash": file.content_hash,
            "format": Path(file.name).suffix.casefold(), "parser_config": parser_config})).hexdigest()

    def _path(self, key: str) -> Path:
        return self._checked_entry(self.directory / (key + ".json"))

    def _checked_entry(self, target: Path) -> Path:
        root = checked_directory(self.directory)
        if (target.parent != root or target.is_symlink() or target.is_junction()
                or target.resolve().parent != root):
            raise ValueError("Document cache entry is redirected")
        return target

    def _read_entry(self, target: Path) -> bytes:
        target = self._checked_entry(target)
        with target.open("rb") as stream:
            # Opening is a filesystem boundary: recheck ancestors and the opened
            # inode before reading bytes, even if the earlier path was canonical.
            self._checked_entry(target)
            if not os.path.samestat(os.fstat(stream.fileno()), target.stat(follow_symlinks=False)):
                raise ValueError("Document cache entry changed during open")
            content = stream.read(self.max_bytes + 1)
        if len(content) > self.max_bytes:
            raise ValueError("Document cache entry exceeds its size budget")
        return content

    def _remove_entry(self, target: Path) -> None:
        try:
            self._checked_entry(target).unlink(missing_ok=True)
        except (OSError, ValueError):
            pass

    def get(self, file: UploadRecord, raw: bytes, parser_config: dict) -> ParsedDocument | None:
        if not self.max_bytes or not self.ttl_seconds:
            return None
        if "sha256:" + hashlib.sha256(raw).hexdigest() != file.content_hash:
            return None
        target = None
        try:
            key = self._key(file, parser_config)
            target = self._path(key)
            if not target.is_file():
                return None
            envelope = json.loads(self._read_entry(target))
            payload = envelope["payload"]
            if (envelope["key"] != key or envelope["expires_at"] <= self.clock()
                    or envelope["checksum"] != hashlib.sha256(canonical_json(payload)).hexdigest()
                    or payload["content_hash"] != file.content_hash
                    or payload["parser_config"] != parser_config):
                raise ValueError("Invalid or expired document cache entry")
            snapshot = build_snapshot(source_uri=file.source_uri, title=file.name,
                media_type=payload["media_type"], source_type="upload", content=raw,
                parser=payload["parser"], parser_version=payload["parser_version"],
                parser_config=parser_config, quality_issues=payload["quality_issues"])
            document = ParsedDocument(snapshot=snapshot, elements=payload["elements"])
            # A lookup changes only eviction recency, never the captured document revision.
            os.utime(self._checked_entry(target), (self.clock(), self.clock()))
            return document
        except (OSError, ValueError, TypeError, KeyError):
            if target is not None:
                self._remove_entry(target)
            return None

    def put(self, file: UploadRecord, document: ParsedDocument) -> None:
        if not self.max_bytes or not self.ttl_seconds:
            return
        snapshot = document.snapshot
        if snapshot.content_hash != file.content_hash or snapshot.capture_scope != "full_document":
            raise ValueError("Only complete matching source conversions can be cached")
        elements = [element.model_dump(mode="json") for element in document.elements]
        for element in elements:
            element["metadata"].pop("file_id", None)
        payload = {"content_hash": file.content_hash, "parser": snapshot.parser,
            "parser_version": snapshot.parser_version, "parser_config": snapshot.parser_config,
            "media_type": snapshot.media_type, "quality_issues": snapshot.quality_issues,
            "elements": elements}
        key = self._key(file, snapshot.parser_config)
        encoded = canonical_json({"key": key, "expires_at": self.clock() + self.ttl_seconds,
            "checksum": hashlib.sha256(canonical_json(payload)).hexdigest(), "payload": payload})
        if len(encoded) > self.max_bytes:
            return
        temporary = None
        try:
            target = self._path(key)
            temporary = target.with_name(f".{uuid4().hex}.part")
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self._checked_entry(temporary).replace(self._checked_entry(target))
            os.utime(self._checked_entry(target), (self.clock(), self.clock()))
            self.prune()
        except (OSError, ValueError):
            logger.warning("document_conversion_cache_write_failed")
        finally:
            if temporary is not None:
                self._remove_entry(temporary)

    def prune(self) -> None:
        try:
            root = checked_directory(self.directory)
            entries = []
            for path in root.glob("*.json"):
                if path.is_symlink() or path.is_junction() or not path.is_file():
                    continue
                stat = path.stat()
                try:
                    envelope = json.loads(self._read_entry(path))
                    if envelope["expires_at"] <= self.clock():
                        raise ValueError("Expired cache entry")
                except (OSError, ValueError, KeyError, TypeError):
                    self._remove_entry(path)
                    continue
                entries.append((stat.st_mtime, stat.st_size, path))
            total = sum(size for _, size, _ in entries)
            for _, size, path in sorted(entries):
                if total <= self.max_bytes:
                    break
                self._checked_entry(path).unlink(missing_ok=True)
                total -= size
        except (OSError, ValueError):
            logger.warning("document_conversion_cache_prune_failed")
