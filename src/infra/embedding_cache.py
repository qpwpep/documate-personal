"""Optional session-owned document vector cache, independent of source citations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path
from threading import RLock
from time import time
from typing import Any

from langchain_core.embeddings import Embeddings


_ENTRY_NAME = re.compile(r"[0-9a-f]{64}\.json")
_IO_LOCK = RLock()
_CACHE_ERRORS = (OSError, ValueError, RuntimeError, OverflowError)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _vector(value: Any, dimensions: int | None = None) -> list[float]:
    if (not isinstance(value, (list, tuple)) or not value
            or (dimensions is not None and len(value) != dimensions)
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)):
        raise ValueError("Embedding vectors must have consistent dimensions and finite numeric values")
    try:
        vector = [float(item) for item in value]
    except (OverflowError, ValueError) as exc:
        raise ValueError("Embedding vector contains an invalid number") from exc
    if not all(math.isfinite(item) for item in vector):
        raise ValueError("Embedding vector contains a nonfinite number")
    return vector


class CachedEmbeddings(Embeddings):
    """Cache exact document text under an immutable parsing/embedding namespace.

    The caller supplies a trusted session directory and a complete namespace
    containing source content, parser/model versions and options, chunking
    settings, and embedding model/dimensions. Only hashes and vectors are
    persisted, never source text, file identities or namespace configuration.
    ``for_namespace`` creates a separate wrapper, so concurrent files cannot
    change one another's active namespace. Query embedding remains uncached.
    """

    def __init__(self, underlying: Embeddings, cache_dir: Path, namespace: dict,
                 max_bytes: int, ttl_seconds: int):
        self._underlying = underlying
        self._cache_dir = Path(cache_dir).absolute()
        self._namespace = _digest(_json_bytes(namespace))
        embedding = namespace.get("embedding")
        dimensions = (embedding.get("dimensions") if isinstance(embedding, dict) else None)
        if dimensions is None:
            dimensions = namespace.get("embedding_dimensions", namespace.get("dimensions"))
        if dimensions is not None and (type(dimensions) is not int or dimensions <= 0):
            raise ValueError("Embedding dimensions must be a positive integer")
        self._dimensions = dimensions
        self._max_bytes = max(0, int(max_bytes))
        self._ttl_seconds = max(0, int(ttl_seconds))

    def for_namespace(self, namespace: dict) -> CachedEmbeddings:
        """Reuse the same service and session cache with a new fixed identity."""
        return CachedEmbeddings(self._underlying, self._cache_dir, namespace,
                                self._max_bytes, self._ttl_seconds)

    def embed_query(self, text: str) -> list[float]:
        return self._underlying.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        unique = list(dict.fromkeys(texts))
        hashes = {text: _digest(text.encode("utf-8")) for text in unique}
        keys = {text: _digest(f"{self._namespace}:{hashes[text]}".encode("ascii")) for text in unique}
        vectors: dict[str, list[float]] = {}
        dimensions = self._dimensions
        with _IO_LOCK:
            if self._available():
                entries = self._prune()
                for text in unique:
                    vector = self._load(keys[text], dimensions=dimensions, entries=entries)
                    if vector is not None:
                        dimensions = len(vector)
                        vectors[text] = vector
        missing = [text for text in unique if text not in vectors]
        if missing:
            # Validate the complete upstream batch before publishing any result.
            computed = self._underlying.embed_documents(missing)
            if not isinstance(computed, (list, tuple)) or len(computed) != len(missing):
                raise ValueError("Embedding service returned an incomplete document batch")
            validated = []
            for value in computed:
                vector = _vector(value, dimensions)
                dimensions = len(vector)
                validated.append(vector)
            vectors.update(zip(missing, validated))
            with _IO_LOCK:
                if self._available():
                    entries = self._prune()
                    for text in missing:
                        self._save(keys[text], hashes[text], vectors[text], entries=entries)
        return [list(vectors[text]) for text in texts]

    def _safe_root(self) -> bool:
        # Inspect the lexical path before mkdir/open: resolve alone would accept
        # a redirect and turn its outside target into the new trust boundary.
        try:
            return all(not path.is_symlink() and not path.is_junction()
                       for path in (self._cache_dir, *self._cache_dir.parents))
        except _CACHE_ERRORS:
            return False

    def _available(self) -> bool:
        if not self._max_bytes or not self._ttl_seconds or not self._safe_root():
            return False
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            return self._safe_root() and self._cache_dir.is_dir()
        except _CACHE_ERRORS:
            return False

    def _safe_entry(self, path: Path) -> bool:
        return (path.parent == self._cache_dir and self._safe_root()
                and not path.is_symlink() and not path.is_junction())

    def _read(self, path: Path) -> dict:
        if not self._safe_entry(path):
            raise ValueError("Cache entry was redirected")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(path, flags), "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > self._max_bytes:
                raise ValueError("Cache entry is not a bounded regular file")
            payload = json.loads(stream.read(self._max_bytes + 1))
        if not isinstance(payload, dict):
            raise ValueError("Invalid embedding cache record")
        checksum = payload.pop("checksum", None)
        if checksum != _digest(_json_bytes(payload)):
            raise ValueError("Embedding cache checksum mismatch")
        required = {"version", "key", "namespace", "text_hash", "created_at", "accessed_at", "dimensions", "vector"}
        if set(payload) != required or payload["version"] != 1 or payload["key"] != path.stem:
            raise ValueError("Embedding cache identity mismatch")
        namespace, text_hash = payload["namespace"], payload["text_hash"]
        if (not isinstance(namespace, str) or not isinstance(text_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", namespace)
                or not re.fullmatch(r"[0-9a-f]{64}", text_hash)
                or _digest(f"{namespace}:{text_hash}".encode("ascii")) != path.stem):
            raise ValueError("Embedding cache identity mismatch")
        created_at = payload["created_at"]
        if (type(created_at) not in (int, float) or not math.isfinite(created_at)
                or not 0 <= time() - created_at < self._ttl_seconds):
            raise ValueError("Embedding cache entry expired")
        accessed_at = payload["accessed_at"]
        if (type(accessed_at) not in (int, float) or not math.isfinite(accessed_at)
                or not created_at <= accessed_at <= time()):
            raise ValueError("Invalid embedding cache access time")
        dimensions = payload["dimensions"]
        if type(dimensions) is not int or dimensions <= 0:
            raise ValueError("Invalid cached embedding dimensions")
        payload["vector"] = _vector(payload["vector"], dimensions)
        return payload

    def _remove(self, path: Path) -> None:
        try:
            if self._safe_entry(path) and _ENTRY_NAME.fullmatch(path.name) and path.is_file():
                path.unlink(missing_ok=True)
        except _CACHE_ERRORS:
            pass

    def _load(self, key: str, *, dimensions: int | None,
              entries: dict[Path, tuple[float, int]]) -> list[float] | None:
        path = self._cache_dir / f"{key}.json"
        try:
            payload = self._read(path)
            if payload["namespace"] != self._namespace:
                raise ValueError("Embedding cache namespace mismatch")
            vector = _vector(payload["vector"], dimensions)
        except _CACHE_ERRORS:
            self._remove(path)
            return None
        # Windows cannot update mtime without following a link. Persist access
        # time by atomic replacement instead, retaining the original TTL origin.
        payload["accessed_at"] = time()
        self._write(key, payload, entries=entries)
        return vector

    def _save(self, key: str, text_hash: str, vector: list[float], *,
              entries: dict[Path, tuple[float, int]]) -> None:
        now = time()
        payload = dict(version=1, key=key, namespace=self._namespace, text_hash=text_hash,
                       created_at=now, accessed_at=now, dimensions=len(vector), vector=vector)
        self._write(key, payload, entries=entries)

    def _write(self, key: str, payload: dict, *, entries: dict[Path, tuple[float, int]]) -> None:
        payload["checksum"] = _digest(_json_bytes(payload))
        encoded = _json_bytes(payload)
        if len(encoded) > self._max_bytes:
            return
        temporary: Path | None = None
        try:
            target = self._cache_dir / f"{key}.json"
            if not self._safe_entry(target):
                return
            # Reserve space before writing each vector without rescanning the
            # whole directory for every chunk in a large document batch.
            total = sum(size for path, (_, size) in entries.items() if path != target)
            for path, (_, size) in sorted(entries.items(), key=lambda entry: entry[1][0]):
                if total + len(encoded) <= self._max_bytes:
                    break
                if path != target:
                    self._remove(path)
                    if path.exists():
                        continue
                    total -= size
                    entries.pop(path, None)
            if total + len(encoded) > self._max_bytes:
                return
            descriptor, name = tempfile.mkstemp(prefix=".embedding-", suffix=".tmp", dir=self._cache_dir)
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            if not self._safe_entry(target) or not self._safe_entry(temporary):
                return
            os.replace(temporary, target)
            temporary = None
            entries[target] = (payload["accessed_at"], len(encoded))
        except _CACHE_ERRORS:
            pass
        finally:
            if temporary is not None:
                try:
                    if self._safe_entry(temporary):
                        temporary.unlink(missing_ok=True)
                except _CACHE_ERRORS:
                    pass

    def _prune(self) -> dict[Path, tuple[float, int]]:
        entries: dict[Path, tuple[float, int]] = {}
        try:
            if not self._safe_root():
                return entries
            for path in self._cache_dir.iterdir():
                if not _ENTRY_NAME.fullmatch(path.name) or not self._safe_entry(path):
                    continue
                try:
                    payload = self._read(path)
                    info = path.stat(follow_symlinks=False)
                    entries[path] = (payload["accessed_at"], info.st_size)
                except _CACHE_ERRORS:
                    self._remove(path)
            total = sum(size for _, size in entries.values())
            for path, (_, size) in sorted(entries.items(), key=lambda entry: entry[1][0]):
                if total <= self._max_bytes:
                    break
                self._remove(path)
                if not path.exists():
                    total -= size
                    entries.pop(path, None)
        except _CACHE_ERRORS:
            pass
        return entries
