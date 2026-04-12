from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chroma_store import (
    CHROMA_DISTANCE_METRIC,
    INDEX_SCHEMA_VERSION,
    NORMALIZATION_VERSION,
    NOTEBOOK_COLLECTION_NAME,
    build_openai_embeddings,
    create_chroma_vectorstore,
)
from .chunking import chunk_notebook_path
from .logging_utils import configure_logging, log_event
from .notebook_loader import is_internal_canonical_path
from .settings import AppSettings, ConfigurationError, get_settings


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 256
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexManifest:
    files: dict[str, float]
    is_legacy: bool = False
    requires_full_rebuild: bool = False


@dataclass(frozen=True)
class BuildSummary:
    total_notebooks: int
    unchanged_count: int
    reindexed_count: int
    deleted_count: int
    embedded_chunk_count: int
    index_dir: Path


def _notebook_paths(*, data_dir: Path, uploads_dir: Path) -> dict[str, float]:
    paths: dict[str, float] = {}
    for root in (data_dir, uploads_dir):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.ipynb")):
            if is_internal_canonical_path(path):
                continue
            paths[str(path)] = os.path.getmtime(path)
    return paths


def _load_manifest(manifest_path: Path) -> IndexManifest:
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                isinstance(payload, dict)
                and isinstance(payload.get("files"), dict)
            ):
                files = {
                    str(path): float(mtime)
                    for path, mtime in payload.get("files", {}).items()
                }
                requires_full_rebuild = bool(
                    int(payload.get("index_version", 0) or 0) != INDEX_SCHEMA_VERSION
                    or str(payload.get("metric") or "").strip().lower() != CHROMA_DISTANCE_METRIC
                    or int(payload.get("normalization_version", 0) or 0) != NORMALIZATION_VERSION
                )
                return IndexManifest(
                    files=files,
                    is_legacy=False,
                    requires_full_rebuild=requires_full_rebuild,
                )
            if isinstance(payload, dict):
                return IndexManifest(
                    files={
                        str(path): float(mtime)
                        for path, mtime in payload.items()
                    },
                    is_legacy=True,
                    requires_full_rebuild=True,
                )
        except Exception:
            return IndexManifest(files={}, is_legacy=True, requires_full_rebuild=True)
    return IndexManifest(files={}, is_legacy=False, requires_full_rebuild=False)


def _save_manifest(manifest_path: Path, manifest: dict[str, float]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "index_version": INDEX_SCHEMA_VERSION,
                "metric": CHROMA_DISTANCE_METRIC,
                "normalization_version": NORMALIZATION_VERSION,
                "collection_name": NOTEBOOK_COLLECTION_NAME,
                "files": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_ipynb_docs(file_paths: list[str]) -> list:
    docs = []
    for file_path in file_paths:
        docs.extend(
            chunk_notebook_path(
                path=file_path,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        )
    return docs

def _ensure_chroma(*, embeddings: Any, index_dir: Path):
    return create_chroma_vectorstore(
        embeddings=embeddings,
        persist_directory=index_dir,
        collection_name=NOTEBOOK_COLLECTION_NAME,
    )


def build_rag_index(
    settings: AppSettings,
    *,
    data_dir: Path = Path("data"),
    uploads_dir: Path = Path("uploads"),
    index_dir: Path | None = None,
) -> BuildSummary:
    if not settings.openai_api_key:
        raise ConfigurationError("[rag_build] Missing required environment variable: OPENAI_API_KEY")

    resolved_index_dir = index_dir or (data_dir / "index")
    manifest_path = resolved_index_dir / "manifest.json"

    if not data_dir.exists() and not uploads_dir.exists():
        raise AssertionError("Neither data/ nor uploads/ folder found")

    current = _notebook_paths(data_dir=data_dir, uploads_dir=uploads_dir)
    manifest = _load_manifest(manifest_path)
    manifest_files = dict(manifest.files)
    requires_full_rebuild = manifest.is_legacy or manifest.requires_full_rebuild
    if not manifest_path.exists() and resolved_index_dir.exists() and any(resolved_index_dir.iterdir()):
        requires_full_rebuild = True

    if requires_full_rebuild and resolved_index_dir.exists():
        shutil.rmtree(resolved_index_dir, ignore_errors=True)
        manifest_files = {}

    to_add_or_update = [path for path, mtime in current.items() if manifest_files.get(path) != mtime]
    to_delete = [path for path in manifest_files.keys() if path not in current]

    log_event(
        logger,
        logging.INFO,
        "rag_build_discovered",
        total_notebooks=len(current),
        unchanged_count=len(current) - len(to_add_or_update),
        reindexed_count=len(to_add_or_update),
        deleted_count=len(to_delete),
        index_dir=resolved_index_dir,
        requires_full_rebuild=requires_full_rebuild,
    )

    embeddings = build_openai_embeddings(settings.openai_api_key)
    chroma = _ensure_chroma(embeddings=embeddings, index_dir=resolved_index_dir)

    if to_delete:
        for path in to_delete:
            chroma.delete(where={"source": path})
        log_event(logger, logging.INFO, "rag_build_deleted_old_entries", deleted_count=len(to_delete))

    embedded_chunk_count = 0
    if to_add_or_update:
        for path in to_add_or_update:
            chroma.delete(where={"source": path})

        chunks = _load_ipynb_docs(to_add_or_update)
        embedded_chunk_count = len(chunks)
        log_event(
            logger,
            logging.INFO,
            "rag_build_embedding_started",
            chunk_count=embedded_chunk_count,
            batch_size=BATCH_SIZE,
        )
        for index in range(0, embedded_chunk_count, BATCH_SIZE):
            batch = chunks[index : index + BATCH_SIZE]
            chroma.add_documents(batch)
            log_event(
                logger,
                logging.INFO,
                "rag_build_batch_embedded",
                embedded_count=index + len(batch),
                total_count=embedded_chunk_count,
            )

        for path in to_add_or_update:
            manifest_files[path] = current[path]

    for path in to_delete:
        manifest_files.pop(path, None)
    _save_manifest(manifest_path, manifest_files)
    chroma.get()

    summary = BuildSummary(
        total_notebooks=len(current),
        unchanged_count=len(current) - len(to_add_or_update),
        reindexed_count=len(to_add_or_update),
        deleted_count=len(to_delete),
        embedded_chunk_count=embedded_chunk_count,
        index_dir=resolved_index_dir,
    )
    log_event(
        logger,
        logging.INFO,
        "rag_build_complete",
        total_notebooks=summary.total_notebooks,
        unchanged_count=summary.unchanged_count,
        reindexed_count=summary.reindexed_count,
        deleted_count=summary.deleted_count,
        embedded_chunk_count=summary.embedded_chunk_count,
        index_dir=summary.index_dir,
    )
    return summary


def run_cli() -> int:
    configure_logging()
    try:
        summary = build_rag_index(get_settings())
    except Exception as exc:
        log_event(logger, logging.ERROR, "rag_build_failed", error=exc)
        return 1

    log_event(
        logger,
        logging.INFO,
        "rag_build_summary",
        total_notebooks=summary.total_notebooks,
        unchanged_count=summary.unchanged_count,
        reindexed_count=summary.reindexed_count,
        deleted_count=summary.deleted_count,
        embedded_chunk_count=summary.embedded_chunk_count,
        index_dir=summary.index_dir,
    )
    return 0


def main() -> int:
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
