from src.core.answer_schema import AnswerResponse
import unittest

from src.core.answer_schema import AnswerDocument, AnswerResponse, finalize_answer, text_document
from src.core.evidence import SearchHit, RetrievalScore
from src.core.contracts.debug import DebugPayload
from src.eval.online_runner.response_parser import parse_agent_response
from src.eval.metric_rules import score_citation_traceability, score_reference_coverage
from src.eval.config_models import BenchmarkCase
from tests.eval.response_fixtures import source_evidence, comparison_response


class CitationStructuredScoringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = BenchmarkCase(case_id="hybrid", category="hybrid", query="비교", upload_fixture="sample.ipynb", require_official_citation=True, require_local_citation=True)
        self.official = source_evidence()
        self.upload = source_evidence(official=False, text="업로드 비교")
        self.response = AnswerResponse.model_validate(comparison_response())
        self.hits = [SearchHit(evidence=item, rank=index, score=RetrievalScore(metric="rank", raw=index, direction="lower")) for index, item in enumerate([self.official, self.upload], 1)]

    def score(self, response=None, hits=None, tools=None):
        return score_citation_traceability(case=self.case, response=response or self.response, observed_hits=self.hits if hits is None else hits, called_tools=["tavily_search", "upload_search"] if tools is None else tools)

    def test_actual_content_refs_resolve_to_observed_source_versions(self) -> None:
        """Both source routes require used citations matching the observed snapshots."""
        self.assertEqual(self.score(), 1.0)

    def test_missing_retrieval_tool_scores_partial(self) -> None:
        """A source route without its recorded retrieval call is not fully traceable."""
        self.assertEqual(self.score(tools=["upload_search"]), 0.5)

    def test_citation_list_without_content_refs_does_not_pass(self) -> None:
        """Listing a source is insufficient when no displayed unit cites it."""
        response = self.response.model_copy(update={"content": text_document("출처 없는 설명")})
        self.assertEqual(self.score(response=response), 0.0)

    def test_changed_source_version_does_not_match_same_path(self) -> None:
        """A new snapshot at the same URI cannot substantiate an older citation."""
        changed = source_evidence(text="수정된 공식 설명")
        self.assertEqual(self.score(hits=[SearchHit(evidence=changed, rank=1, score=RetrievalScore(metric="rank", raw=1, direction="lower")), self.hits[1]]), 0.5)

    def test_changed_selection_is_not_accepted_even_with_copied_id(self) -> None:
        """Tampered selection metadata cannot pass by retaining a valid reference ID."""
        changed = self.official.model_copy(update={"selection": self.official.selection.model_copy(update={"end": 1})})
        changed_hit = self.hits[0].model_copy(update={"evidence": changed})
        parsed = parse_agent_response({
            "response": self.response.model_dump(mode="json"),
            "debug": DebugPayload(observed_hits=[changed_hit.model_dump(mode="json"), self.hits[1].model_dump(mode="json")]).model_dump(mode="json"),
        })
        self.assertTrue(any("debug.observed_hits[0] invalid" in error for error in parsed.response_errors))
        self.assertEqual(self.score(hits=parsed.observed_hits), 0.5)

    def test_unused_observed_source_does_not_change_score(self) -> None:
        """Uncited retrieval results do not dilute or improve citation traceability."""
        extra = SearchHit(evidence=source_evidence(text="다른 자료"), rank=3, score=RetrievalScore(metric="rank", raw=3, direction="lower"))
        self.assertEqual(self.score(hits=self.hits + [extra]), 1.0)

    def test_unapproved_official_domain_does_not_pass(self) -> None:
        """Official citations must still respect the configured official domains."""
        source = source_evidence(source_uri="https://example.com/reference")
        response = finalize_answer(text_document("설명", basis="source", refs=[source.id]), [source])
        self.assertEqual(self.score(response=response, hits=[SearchHit(evidence=source, rank=1, score=RetrievalScore(metric="rank", raw=1, direction="lower"))]), 0.0)

    def test_table_and_code_units_contribute_their_refs(self) -> None:
        """Code and table contents participate in the same traceability traversal."""
        document = AnswerDocument.model_validate({"blocks": [
            {"type": "code", "language": "python", "content": {"text": "x = 1", "basis": "example", "refs": [self.upload.id]}},
            {"type": "table", "columns": [{"text": "기준", "basis": "interaction", "refs": []}], "rows": [[{"text": "공식 설명", "basis": "source", "refs": [self.official.id]}]]},
        ]})
        self.assertEqual(self.score(response=finalize_answer(document, [self.official, self.upload])), 1.0)

    def test_unchecked_paraphrase_is_not_semantic_proof(self) -> None:
        """Reference coverage does not claim that a paraphrase has semantic support."""
        response = finalize_answer(text_document("원문에 대한 해석", basis="source", refs=[self.official.id]), [self.official])
        self.assertEqual(score_reference_coverage(case=self.case, response=response, observed_hits=self.hits), 1.0)
        self.assertEqual(response.checks[0].support_status, "not_evaluated")

    def test_exact_excerpt_is_verified_against_observed_source(self) -> None:
        """An exact source excerpt retains traceability and its separate support check."""
        response = finalize_answer(text_document(self.official.excerpt, basis="excerpt", refs=[self.official.id]), [self.official])
        self.assertEqual(score_reference_coverage(case=self.case, response=response, observed_hits=self.hits), 1.0)
