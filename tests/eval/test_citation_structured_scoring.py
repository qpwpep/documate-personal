import unittest

from src.eval.scoring_rules import score_citation_compliance
from src.eval.schemas import BenchmarkCase, EvidenceItem


class CitationStructuredScoringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.upload_case = BenchmarkCase(
            case_id="hybrid_seed_001",
            category="hybrid",
            query="hybrid query",
            upload_fixture="sample_pipeline.ipynb",
            require_official_citation=True,
            require_local_citation=True,
        )
        self.official = EvidenceItem(
            kind="official",
            tool="tavily_search",
            source_id="url:https://numpy.org/doc/stable/user/basics.broadcasting.html",
            document_id="url:https://numpy.org/doc/stable/user/basics.broadcasting.html",
            url_or_path="https://numpy.org/doc/stable/user/basics.broadcasting.html",
            title="Broadcasting",
            snippet="official snippet",
            score=0.98,
        )
        self.upload_local = EvidenceItem(
            kind="local",
            tool="upload_search",
            source_id="path:uploads/abc/sample_pipeline.ipynb#cell=0;chunk=1;start=12;end=24",
            document_id="path:uploads/abc/sample_pipeline.ipynb",
            url_or_path="uploads/abc/sample_pipeline.ipynb",
            title=None,
            snippet="local snippet",
            score=0.77,
            cell_id=0,
            chunk_id=1,
            start_offset=12,
            end_offset=24,
        )
        self.local_index_case = BenchmarkCase(
            case_id="rag_seed_001",
            category="rag_only",
            query="local query",
            require_official_citation=False,
            require_local_citation=True,
        )
        self.local_index = EvidenceItem(
            kind="local",
            tool="rag_search",
            source_id="path:data/notebooks/example.ipynb#cell=2;chunk=0;start=0;end=16",
            document_id="path:data/notebooks/example.ipynb",
            url_or_path="data/notebooks/example.ipynb",
            title=None,
            snippet="local snippet",
            score=0.71,
            cell_id=2,
            chunk_id=0,
            start_offset=0,
            end_offset=16,
        )

    def test_full_compliance_scores_one(self) -> None:
        score = score_citation_compliance(
            case=self.upload_case,
            response_evidence=[self.official, self.upload_local],
            observed_evidence=[self.official, self.upload_local],
            called_tools=["tavily_search", "upload_search"],
        )
        self.assertAlmostEqual(score, 1.0)

    def test_missing_one_tool_call_scores_partial(self) -> None:
        score = score_citation_compliance(
            case=self.upload_case,
            response_evidence=[self.official, self.upload_local],
            observed_evidence=[self.official, self.upload_local],
            called_tools=["upload_search"],
        )
        self.assertAlmostEqual(score, 0.5)

    def test_local_index_case_still_accepts_rag_search(self) -> None:
        score = score_citation_compliance(
            case=self.local_index_case,
            response_evidence=[self.local_index],
            observed_evidence=[self.local_index],
            called_tools=["rag_search"],
        )
        self.assertAlmostEqual(score, 1.0)

    def test_invalid_domain_scores_zero_for_official(self) -> None:
        invalid_official = EvidenceItem(
            kind="official",
            tool="tavily_search",
            source_id="url:https://example.com/not-allowed",
            url_or_path="https://example.com/not-allowed",
            title=None,
            snippet=None,
            score=None,
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

    def test_chunk_level_source_id_is_accepted_when_document_id_matches(self) -> None:
        response_evidence = [
            EvidenceItem(
                kind="local",
                tool="upload_search",
                source_id="path:uploads/abc/sample_pipeline.ipynb#cell=1;chunk=2;start=40;end=64",
                document_id="path:uploads/abc/sample_pipeline.ipynb",
                url_or_path="uploads/abc/sample_pipeline.ipynb",
                snippet="chunk snippet",
                score=0.8,
                cell_id=1,
                chunk_id=2,
                start_offset=40,
                end_offset=64,
            )
        ]
        observed_evidence = [
            EvidenceItem(
                kind="local",
                tool="upload_search",
                source_id="path:uploads/abc/sample_pipeline.ipynb#cell=1;chunk=2;start=40;end=64",
                document_id="path:uploads/abc/sample_pipeline.ipynb",
                url_or_path="uploads/abc/sample_pipeline.ipynb",
                snippet="chunk snippet",
                score=0.8,
                cell_id=1,
                chunk_id=2,
                start_offset=40,
                end_offset=64,
            )
        ]
        score = score_citation_compliance(
            case=BenchmarkCase(
                case_id="upload_chunk_case",
                category="hybrid",
                query="upload query",
                upload_fixture="sample_pipeline.ipynb",
                require_local_citation=True,
            ),
            response_evidence=response_evidence,
            observed_evidence=observed_evidence,
            called_tools=["upload_search"],
        )
        self.assertAlmostEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
