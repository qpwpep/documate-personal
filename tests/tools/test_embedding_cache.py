from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from src.infra.embedding_cache import CachedEmbeddings
import src.infra.embedding_cache as cache_module


class EmbeddingService(Embeddings):
    """An external embedding service boundary with deterministic vectors."""

    def __init__(self, *, offset: float = 0):
        self.offset = offset
        self.requests: list[list[str]] = []
        self.unavailable = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.requests.append(list(texts))
        if self.unavailable:
            raise ConnectionError("Embedding service unavailable")
        return [[float(len(text)) + self.offset, float(sum(map(ord, text)))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [self.offset, float(len(text))]


def cache(tmp_path, service, **overrides):
    values = dict(underlying=service, cache_dir=tmp_path / "uploads" / "session-a" / "cache" / "embeddings",
                  namespace={"content_hash": "source-v1", "parser": "docling-v1",
                             "embedding": {"model": "fake-v1", "dimensions": 2}},
                  max_bytes=100_000, ttl_seconds=60)
    values.update(overrides)
    return CachedEmbeddings(**values)


def test_repeated_documents_reuse_persisted_vectors_without_reusing_mutable_results(tmp_path):
    """A new wrapper reuses vectors from disk and callers cannot mutate retained data."""
    service = EmbeddingService()
    first = cache(tmp_path, service).embed_documents(["alpha", "beta", "alpha"])
    assert first == [[5.0, 518.0], [4.0, 412.0], [5.0, 518.0]]
    assert service.requests == [["alpha", "beta"]]
    first[0][0] = -1
    assert first[2] == [5.0, 518.0]
    service.unavailable = True
    assert cache(tmp_path, service).embed_documents(["beta", "alpha"]) == [[4.0, 412.0], [5.0, 518.0]]


@pytest.mark.parametrize("namespace", [
    {"content_hash": "source-v2"}, {"parser": "docling-v2"}, {"parser_config": {"ocr": True}},
    {"chunking": {"size": 200}}, {"embedding": {"model": "fake-v2", "dimensions": 2}},
])
def test_namespace_revisions_do_not_reuse_incompatible_vectors(tmp_path, namespace):
    """Changing any parser, source, chunking or embedding identity recomputes vectors."""
    service = EmbeddingService()
    original = cache(tmp_path, service)
    assert original.embed_documents(["alpha"]) == [[5.0, 518.0]]
    service.offset = 100
    assert original.for_namespace(namespace).embed_documents(["alpha"]) == [[105.0, 518.0]]
    assert original.embed_documents(["alpha"]) == [[5.0, 518.0]]


def test_namespace_is_frozen_and_canonical_without_persisting_source_text_or_secrets(tmp_path):
    """Mutating input configuration cannot change a wrapper's identity or disclose its values."""
    namespace = {"embedding": {"dimensions": 2}, "private_option": "secret-value"}
    service = EmbeddingService()
    original = cache(tmp_path, service, namespace=namespace)
    namespace["embedding"]["dimensions"] = 3
    assert original.embed_documents(["private-document"]) == [[16.0, 1671.0]]
    service.unavailable = True
    restored = cache(tmp_path, service, namespace={"private_option": "secret-value", "embedding": {"dimensions": 2}})
    assert restored.embed_documents(["private-document"]) == [[16.0, 1671.0]]
    saved = "".join(path.read_text() for path in tmp_path.rglob("*.json"))
    assert "secret-value" not in saved and "private-document" not in saved


def test_identical_documents_in_other_sessions_require_their_own_embedding(tmp_path):
    """A session never obtains cached vectors from another session directory."""
    service = EmbeddingService()
    assert cache(tmp_path, service).embed_documents(["alpha"]) == [[5.0, 518.0]]
    service.offset = 100
    other = cache(tmp_path, service, cache_dir=tmp_path / "uploads" / "session-b" / "cache" / "embeddings")
    assert other.embed_documents(["alpha"]) == [[105.0, 518.0]]


def test_expiration_recomputes_vectors_even_after_recent_access(tmp_path, monkeypatch):
    """TTL expires from creation while successful access only updates LRU recency."""
    clock = [1000.0]
    monkeypatch.setattr(cache_module, "time", lambda: clock[0])
    service = EmbeddingService()
    wrapper = cache(tmp_path, service)
    assert wrapper.embed_documents(["alpha"]) == [[5.0, 518.0]]
    clock[0] = 1050
    service.offset = 100
    assert wrapper.embed_documents(["alpha"]) == [[5.0, 518.0]]
    clock[0] = 1061
    assert wrapper.embed_documents(["alpha"]) == [[105.0, 518.0]]


def test_lru_eviction_keeps_recently_used_vectors_within_byte_limit(tmp_path, monkeypatch):
    """A bounded cache evicts the least recently used entry across document namespaces."""
    clock = [1000.0]
    monkeypatch.setattr(cache_module, "time", lambda: clock[0])
    service = EmbeddingService()
    directory = tmp_path / "cache"
    assert cache(tmp_path, service, cache_dir=directory).embed_documents(["a"]) == [[1.0, 97.0]]
    entry_bytes = sum(path.stat().st_size for path in directory.glob("*.json"))
    budget = entry_bytes * 2 + 10
    wrapper = cache(tmp_path, service, cache_dir=directory, max_bytes=budget)
    another_document = wrapper.for_namespace({"content_hash": "source-v2", "embedding": {"dimensions": 2}})
    clock[0] = 1001
    assert another_document.embed_documents(["b"]) == [[1.0, 98.0]]
    clock[0] = 1002
    assert wrapper.embed_documents(["a"]) == [[1.0, 97.0]]
    clock[0] = 1003
    assert wrapper.embed_documents(["c"]) == [[1.0, 99.0]]
    assert sum(path.stat().st_size for path in directory.glob("*.json")) <= budget
    service.offset = 100
    assert wrapper.embed_documents(["a", "c"]) == [[1.0, 97.0], [1.0, 99.0]]
    assert another_document.embed_documents(["b"]) == [[101.0, 98.0]]


@pytest.mark.parametrize("damage", ["invalid_json", "swapped_entry", "changed_vector"])
def test_corrupted_or_swapped_cache_records_are_recomputed(tmp_path, damage):
    """Malformed data or a mismatched content checksum never supplies a cached vector."""
    service = EmbeddingService()
    directory = tmp_path / "cache"
    wrapper = cache(tmp_path, service, cache_dir=directory)
    assert wrapper.embed_documents(["alpha"]) == [[5.0, 518.0]]
    first = next(directory.glob("*.json"))
    assert wrapper.embed_documents(["beta"]) == [[4.0, 412.0]]
    if damage == "invalid_json":
        first.write_text("{broken", encoding="utf-8")
    elif damage == "swapped_entry":
        other = next(path for path in directory.glob("*.json") if path != first)
        first.write_bytes(other.read_bytes())
    else:
        payload = json.loads(first.read_text())
        payload["vector"][0] = -100
        first.write_text(json.dumps(payload), encoding="utf-8")
    service.offset = 100
    assert wrapper.embed_documents(["alpha"]) == [[105.0, 518.0]]


def test_failed_embedding_batch_does_not_publish_partial_cache_results(tmp_path):
    """A failing upstream batch leaves previously cached vectors usable and missing vectors absent."""
    service = EmbeddingService()
    wrapper = cache(tmp_path, service)
    assert wrapper.embed_documents(["alpha"]) == [[5.0, 518.0]]
    service.unavailable = True
    with pytest.raises(ConnectionError):
        wrapper.embed_documents(["alpha", "beta", "gamma"])
    service.unavailable = False
    service.offset = 100
    assert wrapper.embed_documents(["alpha", "beta", "gamma"]) == [[5.0, 518.0], [104.0, 412.0], [105.0, 515.0]]


@pytest.mark.parametrize("vectors", [[[1.0, 2.0], [float("nan"), 3.0]], [[1.0, 2.0], [3.0]], [[1.0, 2.0]]])
def test_invalid_provider_vectors_are_rejected_before_any_are_cached(tmp_path, vectors):
    """Nonfinite, inconsistent-dimensional or incomplete provider results cannot poison the cache."""
    class InvalidService(EmbeddingService):
        def embed_documents(self, texts):
            return vectors

    wrapper = cache(tmp_path, InvalidService())
    with pytest.raises(ValueError):
        wrapper.embed_documents(["alpha", "beta"])
    assert not list(tmp_path.rglob("*.json"))
    assert cache(tmp_path, EmbeddingService(offset=100)).embed_documents(["alpha", "beta"]) == [[105.0, 518.0], [104.0, 412.0]]


def test_atomic_cache_write_failure_preserves_embedding_results_and_removes_staging_files(tmp_path, monkeypatch):
    """Optional cache write failures do not fail embedding or leave partial files behind."""
    def fail_replace(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(cache_module.os, "replace", fail_replace)
    service = EmbeddingService()
    wrapper = cache(tmp_path, service)
    assert wrapper.embed_documents(["alpha"]) == [[5.0, 518.0]]
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]
    service.offset = 100
    assert wrapper.embed_documents(["alpha"]) == [[105.0, 518.0]]


def test_unusable_cache_directory_degrades_to_embedding_service(tmp_path):
    """A filesystem obstacle at the cache directory does not make document processing unavailable."""
    obstacle = tmp_path / "cache"
    obstacle.write_text("user-owned file", encoding="utf-8")
    assert cache(tmp_path, EmbeddingService(), cache_dir=obstacle).embed_documents(["alpha"]) == [[5.0, 518.0]]
    assert obstacle.read_text() == "user-owned file"


def test_redirected_cache_directory_never_reads_or_writes_outside_its_boundary(tmp_path):
    """Symlink or junction cache directories are bypassed without changing their targets."""
    outside = tmp_path / "outside"
    outside.mkdir()
    redirect = tmp_path / "redirect"
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(redirect), str(outside)], check=True, capture_output=True)
    else:
        redirect.symlink_to(outside, target_is_directory=True)
    service = EmbeddingService()
    wrapper = cache(tmp_path, service, cache_dir=redirect / "cache")
    assert wrapper.embed_documents(["alpha"]) == [[5.0, 518.0]]
    assert list(outside.iterdir()) == []


def test_queries_always_reach_the_underlying_service_without_creating_cache_files(tmp_path):
    """Only document embeddings are cached, so query behavior remains the provider's behavior."""
    service = EmbeddingService()
    wrapper = cache(tmp_path, service)
    assert wrapper.embed_query("alpha") == [0, 5.0]
    service.offset = 100
    assert wrapper.embed_query("alpha") == [100, 5.0]
    assert not list(tmp_path.rglob("*.json"))
