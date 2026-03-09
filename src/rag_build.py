from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from .chunking import chunk_notebook_path
from .logging_utils import configure_logging, log_event
from .settings import AppSettings, ConfigurationError, get_settings


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 256
logger = logging.getLogger(__name__)


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
            paths[str(path)] = os.path.getmtime(path)
    return paths


def _load_manifest(manifest_path: Path) -> dict[str, float]:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_manifest(manifest_path: Path, manifest: dict[str, float]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
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


def _ensure_chroma(*, embeddings: OpenAIEmbeddings, index_dir: Path) -> Chroma:
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(index_dir),
        collection_name="notebooks",
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

    to_add_or_update = [path for path, mtime in current.items() if manifest.get(path) != mtime]
    to_delete = [path for path in manifest.keys() if path not in current]

    log_event(
        logger,
        logging.INFO,
        "rag_build_discovered",
        total_notebooks=len(current),
        unchanged_count=len(current) - len(to_add_or_update),
        reindexed_count=len(to_add_or_update),
        deleted_count=len(to_delete),
        index_dir=resolved_index_dir,
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key,
    )
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
            manifest[path] = current[path]

    for path in to_delete:
        manifest.pop(path, None)
    _save_manifest(manifest_path, manifest)
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
