import unittest

from langchain_core.documents import Document

from src.tools.local_rag import _build_query_focused_snippet, _rank_retrieval_rows


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
            query="업로드한 파일에서 groupby를 어떻게 쓰는지 찾아서 설명해줘.",
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
            query="sample_pipeline.ipynb 기준으로 train_test_split 파라미터를 찾아줘.",
        )

        self.assertEqual(ranked[0][0].metadata["cell_id"], 2)


if __name__ == "__main__":
    unittest.main()
