import unittest

from src.runtime.nodes.retrieval.formatting import format_evidence_for_prompt


class RetrievalFormattingTest(unittest.TestCase):
    def test_format_evidence_for_prompt_truncates_local_snippets_by_default(self) -> None:
        formatted = format_evidence_for_prompt(
            [
                {
                    "kind": "local",
                    "tool": "upload_search",
                    "source_id": "path:uploads/demo/sample.py#chunk=0;start=0;end=999",
                    "url_or_path": "uploads/demo/sample.py",
                    "snippet": (
                        "import pandas as pd "
                        + "x " * 120
                        + 'grouped = all_sales.groupby("region", as_index=False)["amount"].sum()'
                    ),
                    "score": 0.0,
                }
            ],
            max_snippet_chars=120,
        )

        self.assertIn("import pandas as pd", formatted)
        self.assertIn("groupby", formatted)
        self.assertIn("...", formatted)

    def test_format_evidence_for_prompt_can_preserve_local_snippets_for_extraction(self) -> None:
        formatted = format_evidence_for_prompt(
            [
                {
                    "kind": "local",
                    "tool": "upload_search",
                    "source_id": "path:uploads/demo/sample.py#chunk=0;start=0;end=999",
                    "url_or_path": "uploads/demo/sample.py",
                    "snippet": (
                        "import pandas as pd "
                        + "x " * 120
                        + 'grouped = all_sales.groupby("region", as_index=False)["amount"].sum()'
                    ),
                    "score": 0.0,
                }
            ],
            max_snippet_chars=120,
            preserve_local_snippets=True,
        )

        self.assertIn("import pandas as pd", formatted)
        self.assertIn("groupby", formatted)
        self.assertNotIn("...", formatted)

    def test_format_evidence_for_prompt_still_truncates_official_snippets(self) -> None:
        formatted = format_evidence_for_prompt(
            [
                {
                    "kind": "official",
                    "tool": "tavily_search",
                    "source_id": "url:https://docs.example.com/page",
                    "url_or_path": "https://docs.example.com/page",
                    "title": "Example Docs",
                    "snippet": (
                        "Official docs intro "
                        + "x " * 120
                        + "final official detail"
                    ),
                    "score": 0.9,
                }
            ],
            max_snippet_chars=120,
        )

        self.assertIn("Official docs intro", formatted)
        self.assertIn("final official detail", formatted)
        self.assertIn("...", formatted)

    def test_format_evidence_for_prompt_renders_code_metadata_before_snippet(self) -> None:
        formatted = format_evidence_for_prompt(
            [
                {
                    "kind": "local",
                    "tool": "upload_search",
                    "source_id": "path:uploads/demo/sample.py#chunk=0;start=0;end=120",
                    "url_or_path": "uploads/demo/sample.py",
                    "snippet": "model = LogisticRegression(max_iter=200)",
                    "score": 0.9,
                    "code_metadata": {
                        "calls": [
                            {
                                "call_name": "LogisticRegression",
                                "kwargs": {"max_iter": "200"},
                            }
                        ],
                        "option_literals": ["max_iter=200"],
                    },
                }
            ],
            max_snippet_chars=120,
        )

        self.assertIn("candidate_facts: max_iter=200", formatted)
        self.assertIn('"call_name": "LogisticRegression"', formatted)
        self.assertLess(formatted.index("code_metadata:"), formatted.index("snippet:"))


if __name__ == "__main__":
    unittest.main()
