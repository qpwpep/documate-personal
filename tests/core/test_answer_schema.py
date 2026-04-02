import unittest

from src.answer_schema import build_deterministic_grounded_payload, clean_grounded_text
from src.evidence import EvidenceItem


class AnswerSchemaTest(unittest.TestCase):
    def test_clean_grounded_text_removes_markdown_and_navigation_lines(self) -> None:
        cleaned = clean_grounded_text(
            "\n".join(
                [
                    "# NumPy broadcasting",
                    "Home > Docs > API",
                    "Table of contents",
                    "- Broadcasting expands compatible array shapes.",
                    "Previous: Indexing routines",
                ]
            )
        )

        self.assertEqual(cleaned, "Broadcasting expands compatible array shapes.")

    def test_build_deterministic_grounded_payload_uses_cleaned_fallback_claim_text(self) -> None:
        payload = build_deterministic_grounded_payload(
            evidence_items=[
                EvidenceItem(
                    kind="official",
                    tool="tavily_search",
                    source_id="url:https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
                    document_id="url:https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
                    url_or_path="https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
                    title="numpy.concatenate",
                    snippet="\n".join(
                        [
                            "On this page",
                            "- Use [`numpy.concatenate`](https://numpy.org) to join arrays along an existing axis.",
                            "Next: numpy.stack",
                        ]
                    ),
                    score=0.91,
                )
            ]
        )

        self.assertEqual(len(payload.claims), 1)
        self.assertEqual(
            payload.claims[0].text,
            "Use numpy.concatenate to join arrays along an existing axis.",
        )


if __name__ == "__main__":
    unittest.main()
