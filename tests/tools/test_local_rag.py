import json
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.chroma_store import create_chroma_vectorstore
from src.chunking import chunk_notebook_path
from src.settings import AppSettings
from src.tools.local_rag import (
    _build_query_focused_snippet,
    _rank_retrieval_rows,
    build_local_rag_tools,
    build_temp_retriever,
)


class _FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(text.count("train_test_split"))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(text.count("train_test_split"))]


def _write_notebook(path: Path, *sources: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = []
    for index, source in enumerate(sources):
        cells.append(
            {
                "cell_type": "markdown" if index == 0 else "code",
                "metadata": {},
                "source": [source],
                **({"execution_count": None, "outputs": []} if index > 0 else {}),
            }
        )
    path.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


class LocalRagTest(unittest.TestCase):
    def test_build_query_focused_snippet_centers_on_matching_identifier(self) -> None:
        text = (
            "import pandas as pd\n\n"
            "sales_q1 = pd.DataFrame(...)\n"
            "sales_q2 = pd.DataFrame(...)\n"
            "profiles = pd.DataFrame(...)\n"
            "# concat example\n"
            "all_sales = pd.concat([sales_q1, sales_q2], ignore_index=True)\n"
            "# groupby example\n"
            'grouped = all_sales.groupby("region", as_index=False)["amount"].sum()\n'
            "# merge example\n"
            'sales_with_profile = all_sales.merge(profiles, on="user_id", how="left")\n'
        )

        snippet = _build_query_focused_snippet(
            text,
            query="?낅줈?쒗븳 ?뚯씪?먯꽌 groupby瑜??대뼸寃??곕뒗吏 李얠븘???ㅻ챸?댁쨾.",
            max_length=120,
        )

        self.assertIn("groupby", snippet)
        self.assertNotIn("sales_q1 = pd.DataFrame", snippet)

    def test_rank_retrieval_rows_prefers_parameter_cell_for_parameter_queries(self) -> None:
        import_doc = Document(
            page_content=(
                "from sklearn.model_selection import train_test_split\n"
                "from sklearn.preprocessing import StandardScaler"
            ),
            metadata={"cell_id": 1},
        )
        usage_doc = Document(
            page_content=(
                "X_train, X_test, y_train, y_test = train_test_split("
                "X, y, test_size=0.2, random_state=42)"
            ),
            metadata={"cell_id": 2},
        )

        ranked = _rank_retrieval_rows(
            [(import_doc, 0.3), (usage_doc, 0.28)],
            query="sample_pipeline.ipynb 湲곗??쇰줈 train_test_split ?뚮씪誘명꽣瑜?李얠븘以?",
        )

        self.assertEqual(ranked[0][0].metadata["cell_id"], 2)

    @patch("src.tools.local_rag.build_openai_embeddings", return_value=_FakeEmbeddings())
    def test_local_rag_search_uses_raw_l2_scores_without_userwarning(
        self,
        _mock_local_embeddings,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            notebook_path = data_dir / "sample_pipeline.ipynb"
            _write_notebook(
                notebook_path,
                "# Sample pipeline",
                "from sklearn.model_selection import train_test_split\n"
                "X_train, X_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
            )

            settings = AppSettings(openai_api_key="test-key", tavily_api_key="test")
            docs = chunk_notebook_path(
                path=str(notebook_path),
                chunk_size=800,
                chunk_overlap=120,
            )
            vectorstore = create_chroma_vectorstore(
                embeddings=_FakeEmbeddings(),
                collection_name="local-rag-test",
            )
            vectorstore.add_documents(docs)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rag_tool, _upload_tool = build_local_rag_tools(settings)
                with patch("src.tools.local_rag.INDEX_PATH", data_dir), patch(
                    "src.tools.local_rag.load_chroma",
                    return_value=vectorstore,
                ):
                    payload = rag_tool.func(query="train_test_split parameter", k=2)
            vectorstore.delete_collection()

            self.assertEqual(caught, [])
            self.assertEqual(payload["diagnostics"]["metric"], "l2")
            self.assertEqual(payload["diagnostics"]["score_direction"], "lower_is_better")
            self.assertTrue(all(0.0 <= item["score"] <= 1.0 for item in payload["evidence"]))

    @patch("src.tools.local_rag.build_openai_embeddings", return_value=_FakeEmbeddings())
    def test_upload_rag_search_uses_canonical_copy_and_raw_l2_scores_without_userwarning(
        self,
        _mock_local_embeddings,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            uploads_root = Path(temp_dir) / "uploads" / "session-a"
            notebook_path = uploads_root / "sample_pipeline.ipynb"
            _write_notebook(
                notebook_path,
                "# Sample pipeline",
                "from sklearn.model_selection import train_test_split\n"
                "X_train, X_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
            )

            settings = AppSettings(openai_api_key="test-key", tavily_api_key="test")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                handle = build_temp_retriever(str(notebook_path), api_key="test-key")
                try:
                    _rag_tool, upload_tool = build_local_rag_tools(settings)
                    payload = upload_tool.func(
                        query="train_test_split parameter",
                        k=2,
                        retriever=handle.retriever,
                    )
                finally:
                    handle.cleanup()

            canonical_path = notebook_path.parent / ".canonical" / notebook_path.name
            self.assertEqual(caught, [])
            self.assertTrue(canonical_path.exists())
            self.assertEqual(payload["diagnostics"]["metric"], "l2")
            self.assertEqual(payload["diagnostics"]["score_direction"], "lower_is_better")
            self.assertEqual(payload["evidence"][0]["url_or_path"], str(notebook_path))


if __name__ == "__main__":
    unittest.main()
