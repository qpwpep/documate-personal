from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from src.core.documents import DocumentElement, ParsedDocument, build_snapshot
from src.core.uploads import UploadRecord
from src.infra.document_cache import ConversionCache


def source(tmp_path: Path, name: str = "first.pdf", identity: str = "first"):
    raw = b"%PDF-1.7\nfixture"
    path = tmp_path / name
    path.write_bytes(raw)
    return UploadRecord(file_id=identity, name=name, path=str(path), size_bytes=len(raw),
                        content_hash="sha256:" + hashlib.sha256(raw).hexdigest(),
                        source_uri=f"upload:///session/{identity}"), raw


def document(file, raw, config):
    return ParsedDocument(snapshot=build_snapshot(source_uri=file.source_uri, title=file.name,
        media_type="application/pdf", source_type="upload", content=raw,
        parser="docling", parser_version="test", parser_config=config),
        elements=[DocumentElement(element_id="text-0", kind="paragraph", text="검증된 원문 120")])


def test_cached_conversion_rebinds_file_identity_and_returns_independent_sources(tmp_path):
    """Equal bytes reuse conversion while each attachment retains its identity and lifetime."""
    first, raw = source(tmp_path)
    second, _ = source(tmp_path, "second.pdf", "second")
    cache = ConversionCache(tmp_path / "cache", max_bytes=100_000, ttl_seconds=60)
    config = {"model": "v1", "ocr": True}
    cache.put(first, document(first, raw, config))
    one = cache.get(first, raw, config)
    two = cache.get(second, raw, config)
    assert one.snapshot.source_uri == first.source_uri
    assert two.snapshot.source_uri == second.source_uri
    assert two.snapshot.title == second.name
    assert one.snapshot.snapshot_id != two.snapshot.snapshot_id
    one.elements.clear()
    assert two.elements[0].text == "검증된 원문 120"
    assert cache.get(first, raw, config).elements[0].text == "검증된 원문 120"


def test_parser_options_and_model_revision_invalidate_conversion(tmp_path):
    """A cached document never serves a different model or extraction configuration."""
    file, raw = source(tmp_path)
    cache = ConversionCache(tmp_path / "cache", max_bytes=100_000, ttl_seconds=60)
    config = {"model": "v1", "ocr": True}
    cache.put(file, document(file, raw, config))
    assert cache.get(file, raw, {"model": "v2", "ocr": True}) is None
    assert cache.get(file, raw, {"model": "v1", "ocr": False}) is None


def test_corrupt_or_expired_cache_is_a_miss(tmp_path):
    """Corruption and expiration cause recomputation instead of invalid source publication."""
    file, raw = source(tmp_path)
    now = [100.0]
    cache = ConversionCache(tmp_path / "cache", max_bytes=100_000, ttl_seconds=60, clock=lambda: now[0])
    cache.put(file, document(file, raw, {}))
    now[0] = 161.0
    assert cache.get(file, raw, {}) is None
    cache.put(file, document(file, raw, {}))
    next((tmp_path / "cache").glob("*.json")).write_text('{"broken": true}')
    assert cache.get(file, raw, {}) is None


def test_cache_size_limit_does_not_discard_an_already_returned_document(tmp_path):
    """Pruning cached bytes cannot mutate a retained source or previously returned citation."""
    file, raw = source(tmp_path)
    cache = ConversionCache(tmp_path / "cache", max_bytes=100_000, ttl_seconds=60)
    cache.put(file, document(file, raw, {}))
    retained = cache.get(file, raw, {})
    bounded = ConversionCache(tmp_path / "cache", max_bytes=1, ttl_seconds=60)
    bounded.prune()
    assert not list((tmp_path / "cache").glob("*.json"))
    assert retained.elements[0].text == "검증된 원문 120"


def test_cache_expiration_does_not_slide_when_a_hit_updates_lru(tmp_path):
    """Frequently read conversions still expire at their original conversion deadline."""
    file, raw = source(tmp_path)
    now = [100.0]
    cache = ConversionCache(tmp_path / "cache", max_bytes=100_000, ttl_seconds=10, clock=lambda: now[0])
    cache.put(file, document(file, raw, {}))
    now[0] = 109.0
    assert cache.get(file, raw, {}) is not None
    now[0] = 111.0
    cache.prune()
    assert not list((tmp_path / "cache").glob("*.json"))


def test_cache_redirect_during_read_does_not_read_or_delete_outside_entry(tmp_path, monkeypatch):
    """An ancestor replaced between path validation and open cannot escape the cache area."""
    file, raw = source(tmp_path)
    directory = tmp_path / "cache"
    cache = ConversionCache(directory, max_bytes=100_000, ttl_seconds=60)
    cache.put(file, document(file, raw, {}))
    entry = next(directory.glob("*.json"))
    outside = tmp_path / "outside"
    outside.mkdir()
    untouched = outside / entry.name
    untouched.write_bytes(b"outside file")
    original_open = Path.open
    redirected = False

    def redirect_before_open(path, *args, **kwargs):
        nonlocal redirected
        if path == entry and not redirected:
            redirected = True
            directory.rename(tmp_path / "retained-cache")
            if os.name == "nt":
                subprocess.run(["cmd", "/c", "mklink", "/J", str(directory), str(outside)], check=True, capture_output=True)
            else:
                directory.symlink_to(outside, target_is_directory=True)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", redirect_before_open)
    assert cache.get(file, raw, {}) is None
    assert untouched.read_bytes() == b"outside file"
