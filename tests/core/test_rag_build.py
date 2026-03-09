from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.rag_build import build_rag_index
from src.settings import AppSettings


class _FakeChroma:
    def __init__(self) -> None:
        self.deleted_filters: list[dict] = []
        self.added_batches: list[list] = []
        self.get_called = False

    def delete(self, where: dict) -> None:
        self.deleted_filters.append(where)

    def add_documents(self, batch: list) -> None:
        self.added_batches.append(list(batch))

    def get(self):
        self.get_called = True
        return {}


def _write_notebook(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "id": "cell-1",
                        "metadata": {},
                        "source": [source],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


class RagBuildTest(unittest.TestCase):
    @patch("src.rag_build.OpenAIEmbeddings")
    def test_build_rag_index_creates_manifest_and_batches(self, _mock_embeddings) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            uploads_dir = root / "uploads"
            index_dir = data_dir / "index"
            notebook_path = data_dir / "sample.ipynb"
            _write_notebook(notebook_path, "hello world")
            fake_chroma = _FakeChroma()

            with patch("src.rag_build._ensure_chroma", return_value=fake_chroma):
                summary = build_rag_index(
                    AppSettings(openai_api_key="test-key", tavily_api_key="test"),
                    data_dir=data_dir,
                    uploads_dir=uploads_dir,
                    index_dir=index_dir,
                )

            self.assertEqual(summary.total_notebooks, 1)
            self.assertEqual(summary.reindexed_count, 1)
            self.assertGreater(summary.embedded_chunk_count, 0)
            self.assertTrue((index_dir / "manifest.json").exists())
            self.assertTrue(fake_chroma.get_called)
            self.assertTrue(fake_chroma.added_batches)

    @patch("src.rag_build.OpenAIEmbeddings")
    def test_build_rag_index_deletes_removed_entries(self, _mock_embeddings) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            uploads_dir = root / "uploads"
            index_dir = data_dir / "index"
            manifest_path = index_dir / "manifest.json"
            index_dir.mkdir(parents=True, exist_ok=True)
            stale_path = str(data_dir / "removed.ipynb")
            manifest_path.write_text(json.dumps({stale_path: 1.0}), encoding="utf-8")
            fake_chroma = _FakeChroma()

            with patch("src.rag_build._ensure_chroma", return_value=fake_chroma):
                summary = build_rag_index(
                    AppSettings(openai_api_key="test-key", tavily_api_key="test"),
                    data_dir=data_dir,
                    uploads_dir=uploads_dir,
                    index_dir=index_dir,
                )

            self.assertEqual(summary.deleted_count, 1)
            self.assertIn({"source": stale_path}, fake_chroma.deleted_filters)


if __name__ == "__main__":
    unittest.main()
