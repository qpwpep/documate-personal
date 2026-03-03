import unittest

from src.eval.scoring_rules import score_citation_compliance
from src.eval.schemas import BenchmarkCase, EvidenceItem


class CitationStructuredScoringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = BenchmarkCase(
            case_id="hybrid_seed_001",
            category="hybrid",
            query="hybrid query",
            require_official_citation=True,
            require_local_citation=True,
        )
        self.official = EvidenceItem(
            kind="official",
            source="https://numpy.org/doc/stable/user/basics.broadcasting.html",
            title="Broadcasting",
            snippet="official snippet",
            tool="tavily_search",
            source_id="url:https://numpy.org/doc/stable/user/basics.broadcasting.html",
        )
        self.local = EvidenceItem(
            kind="local",
            source="uploads/abc/sample_pipeline.ipynb",
            title=None,
            snippet="local snippet",
            tool="rag_search",
            source_id="path:uploads/abc/sample_pipeline.ipynb",
        )

    def test_full_compliance_scores_one(self) -> None:
        score = score_citation_compliance(
            case=self.case,
            response_evidence=[self.official, self.local],
            observed_evidence=[self.official, self.local],
            called_tools=["tavily_search", "rag_search"],
        )
        self.assertAlmostEqual(score, 1.0)

    def test_missing_one_tool_call_scores_partial(self) -> None:
        score = score_citation_compliance(
            case=self.case,
            response_evidence=[self.official, self.local],
            observed_evidence=[self.official, self.local],
            called_tools=["rag_search"],
        )
        self.assertAlmostEqual(score, 0.5)

    def test_invalid_domain_scores_zero_for_official(self) -> None:
        invalid_official = EvidenceItem(
            kind="official",
            source="https://example.com/not-allowed",
            title=None,
            snippet=None,
            tool="tavily_search",
            source_id="url:https://example.com/not-allowed",
        )
        score = score_citation_compliance(
            case=BenchmarkCase(
                case_id="docs_seed_001",
                category="docs_only",
                query="docs",
                require_official_citation=True,
                require_local_citation=False,
            ),
            response_evidence=[invalid_official],
            observed_evidence=[invalid_official],
            called_tools=["tavily_search"],
        )
        self.assertAlmostEqual(score, 0.0)

    def test_non_citation_case_is_one(self) -> None:
        score = score_citation_compliance(
            case=BenchmarkCase(case_id="tool_seed_001", category="tool_action", query="tool"),
            response_evidence=[],
            observed_evidence=[],
            called_tools=[],
        )
        self.assertAlmostEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
