import json
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.infra.chunking import chunk_notebook_path, chunk_python_text
from src.infra.settings import AppSettings
from src.infra.tools.local_rag import build_temp_retriever, build_upload_search_tool
from src.infra.tools.local_rag.ranking import rank_retrieval_rows
from src.infra.tools.local_rag.serialization import build_local_snippet, build_query_focused_snippet


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

        snippet = build_query_focused_snippet(
            text,
            query="업로드한 파일에서 groupby를 어떻게 쓰는지 찾아서 설명해줘.",
            max_length=120,
        )

        self.assertIn("groupby", snippet)
        self.assertNotIn("sales_q1 = pd.DataFrame", snippet)

    def test_build_query_focused_snippet_prefers_more_relevant_usage_over_earliest_match(self) -> None:
        text = (
            "# train_test_split helper notes\n"
            "from sklearn.model_selection import train_test_split\n"
            "# later usage\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X,\n"
            "    y,\n"
            "    test_size=0.2,\n"
            "    random_state=42,\n"
            ")\n"
        )

        snippet = build_query_focused_snippet(
            text,
            query="train_test_split random_state parameter",
            max_length=120,
        )

        self.assertIn("random_state=42", snippet)
        self.assertNotIn("helper notes", snippet)

    def test_build_local_snippet_uses_query_window_for_single_chunk_documents(self) -> None:
        text = "header\n" + ("value = 1\n" * 80) + "target_call(random_state=42)\n"

        snippet = build_local_snippet(
            text,
            query="random_state parameter",
            metadata={"document_chunk_count": 1, "document_char_count": len(text)},
        )

        self.assertIn("target_call(random_state=42)", snippet)
        self.assertNotIn("header", snippet)
        self.assertLess(len(snippet), 500)
        self.assertLessEqual(len(snippet.splitlines()), 5)

    def test_build_local_snippet_uses_query_window_for_short_documents(self) -> None:
        text = "header\n" + ("step = 1\n" * 40) + "target_call(random_state=42)\n"

        snippet = build_local_snippet(
            text,
            query="random_state parameter",
            metadata={"document_chunk_count": 3, "document_char_count": len(text)},
        )

        self.assertIn("target_call(random_state=42)", snippet)
        self.assertNotIn("header", snippet)
        self.assertLessEqual(len(snippet.splitlines()), 5)
        self.assertLessEqual(len(text), 1200)

    def test_build_local_snippet_preserves_full_chunk_for_explicit_extraction(self) -> None:
        text = "header\n" + ("step = 1\n" * 40) + "target_call(random_state=42)\n"

        snippet = build_local_snippet(
            text,
            query="show the exact code snippet with random_state",
            metadata={"document_chunk_count": 1, "document_char_count": len(text)},
        )

        self.assertEqual(snippet, text.strip())

    def test_build_local_snippet_places_concat_window_first_for_short_files(self) -> None:
        text = (
            "import pandas as pd\n"
            + ("setup_value = 1\n" * 32)
            + "all_sales = pd.concat([sales_q1, sales_q2], ignore_index=True)\n"
            + "print(all_sales.shape)\n"
        )

        snippet = build_local_snippet(
            text,
            query="concat ignore_index option",
            metadata={"document_chunk_count": 1, "document_char_count": len(text)},
            max_length=768,
        )

        self.assertTrue(snippet.startswith("all_sales = pd.concat"))
        self.assertIn("ignore_index=True", snippet)
        self.assertNotIn("import pandas as pd", snippet)

    def test_chunk_python_text_annotates_document_counts(self) -> None:
        text = "line = 1\n" * 240

        docs = chunk_python_text(
            path="uploads/session/sample.py",
            text=text,
            chunk_size=200,
            chunk_overlap=20,
        )

        self.assertGreater(len(docs), 1)
        self.assertTrue(all(doc.metadata["document_chunk_count"] == len(docs) for doc in docs))
        self.assertTrue(all(doc.metadata["document_char_count"] == len(text) for doc in docs))

    def test_chunk_python_text_adds_ast_code_metadata(self) -> None:
        docs = chunk_python_text(
            path="uploads/session/model.py",
            text=(
                "from sklearn.linear_model import LogisticRegression\n"
                "model = LogisticRegression(max_iter=200, random_state=42)\n"
            ),
            chunk_size=800,
            chunk_overlap=120,
        )

        metadata = json.loads(docs[0].metadata["code_metadata"])
        self.assertEqual(metadata["calls"][0]["call_name"], "LogisticRegression")
        self.assertEqual(metadata["calls"][0]["kwargs"]["max_iter"], "200")
        self.assertIn("max_iter=200", metadata["option_literals"])

    def test_chunk_notebook_path_adds_cell_ast_code_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            notebook_path = Path(temp_dir) / "sample_pipeline.ipynb"
            _write_notebook(
                notebook_path,
                "# Sample pipeline",
                "print('setup')\n",
                "from sklearn.linear_model import LogisticRegression\n"
                "model = LogisticRegression(max_iter=200)\n",
            )

            docs = chunk_notebook_path(
                path=str(notebook_path),
                chunk_size=800,
                chunk_overlap=120,
            )

        matching = [
            json.loads(doc.metadata["code_metadata"])
            for doc in docs
            if "code_metadata" in doc.metadata
            and "LogisticRegression" in doc.metadata["code_metadata"]
        ]
        self.assertEqual(matching[0]["cell_id"], 2)
        self.assertEqual(matching[0]["calls"][0]["call_name"], "LogisticRegression")
        self.assertEqual(matching[0]["calls"][0]["kwargs"]["max_iter"], "200")

    def test_chunk_notebook_path_normalizes_list_sources_without_list_repr(self) -> None:
        with TemporaryDirectory() as temp_dir:
            notebook_path = Path(temp_dir) / "sample_pipeline.ipynb"
            notebook_path.write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "cell_type": "code",
                                "execution_count": None,
                                "metadata": {},
                                "outputs": [],
                                "source": [
                                    "from sklearn.preprocessing import StandardScaler\r\n",
                                    "scaler = StandardScaler()\r\n",
                                ],
                            }
                        ],
                        "metadata": {},
                        "nbformat": 4,
                        "nbformat_minor": 5,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            docs = chunk_notebook_path(
                path=str(notebook_path),
                chunk_size=800,
                chunk_overlap=120,
            )

        self.assertEqual(len(docs), 1)
        expected_source = (
            "from sklearn.preprocessing import StandardScaler\n"
            "scaler = StandardScaler()\n"
        )
        self.assertEqual(docs[0].page_content, expected_source.strip())
        self.assertEqual(docs[0].metadata["document_char_count"], len(expected_source))
        snippet = build_local_snippet(
            docs[0].page_content,
            query="StandardScaler initialization",
            metadata=docs[0].metadata,
        )
        self.assertEqual(snippet, expected_source.strip())
        self.assertNotIn("['from sklearn", docs[0].page_content)
        self.assertNotIn("['from sklearn", snippet)

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

        ranked = rank_retrieval_rows(
            [(import_doc, 0.3), (usage_doc, 0.28)],
            query="sample_pipeline.ipynb 기준으로 train_test_split 파라미터를 찾아줘",
        )

        self.assertEqual(ranked[0][0].metadata["cell_id"], 2)

    @patch("src.infra.chroma_store.OpenAIEmbeddings", return_value=_FakeEmbeddings())
    def test_upload_rag_search_uses_canonical_copy_and_raw_l2_scores_without_userwarning(
        self,
        _mock_openai_embeddings,
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
                    upload_tool = build_upload_search_tool(settings)
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
            self.assertEqual(payload["diagnostics"]["route"], "upload")
            self.assertEqual(
                {
                    (item["kind"], item["tool"], item["url_or_path"])
                    for item in payload["evidence"]
                },
                {("local", "upload_search", str(notebook_path))},
            )
            self.assertTrue(all(0.0 <= item["score"] <= 1.0 for item in payload["evidence"]))


if __name__ == "__main__":
    unittest.main()
