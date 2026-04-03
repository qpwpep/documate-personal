import unittest

from src.nodes.retrieval.formatting import format_evidence_for_prompt


class RetrievalFormattingTest(unittest.TestCase):
    def test_format_evidence_for_prompt_keeps_tail_context_when_truncating(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
