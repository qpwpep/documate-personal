from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from src.infra.tools.docs_search.serialization import DocsSearchFilterCounters, collect_docs_search_hits
from src.infra.tools.docs_search.url_validation import DocUrlValidationResult


class DocsSearchSerializationTest(unittest.TestCase):
    def test_collect_docs_search_hits_validates_same_priority_urls_in_parallel(self) -> None:
        active = 0
        max_active = 0
        lock = threading.Lock()

        def validate(url: str) -> DocUrlValidationResult:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return DocUrlValidationResult(ok=True, final_url=url, status_code=200)

        results = [
            {
                "url": f"https://numpy.org/doc/stable/reference/generated/numpy.item{i}.html",
                "title": f"numpy.item{i}",
                "content": f"NumPy item {i} reference.",
                "score": 0.9,
            }
            for i in range(3)
        ]
        counters = DocsSearchFilterCounters()

        with patch("src.infra.tools.docs_search.serialization.validate_doc_url", side_effect=validate):
            hits, _raw_scores = collect_docs_search_hits(
                results,
                allowed_domains=["numpy.org"],
                retrieval_warnings=[],
                query="numpy reference",
                filter_counters=counters,
            )

        self.assertEqual(len(hits), 3)
        self.assertEqual(counters.validated_url_count, 3)
        self.assertGreater(max_active, 1)

    def _collect(self, content: str, *, raw_content: str | None = None, query: str = "numpy reshape", score: float | None = 0.8, title: str | None = "numpy.reshape"):
        result = {
            "url": "https://numpy.org/doc/1.26/reference/generated/numpy.reshape.html#numpy.reshape",
            "title": title,
            "content": content,
            "raw_content": raw_content,
            "score": score,
        }
        with patch(
            "src.infra.tools.docs_search.serialization.validate_doc_url",
            side_effect=lambda url: DocUrlValidationResult(ok=True, final_url=url, status_code=200),
        ):
            hits, _ = collect_docs_search_hits(
                [result], allowed_domains=["numpy.org"], retrieval_warnings=[], query=query,
            )
        return hits[0]

    def test_provider_excerpt_preserves_version_uri_and_exact_content(self) -> None:
        content = "  Gives a new shape to an array.\nWithout changing its data.  "
        hit = self._collect(content)

        self.assertEqual(hit.evidence.snapshot.source_uri,
                         "https://numpy.org/doc/1.26/reference/generated/numpy.reshape.html#numpy.reshape")
        self.assertEqual(hit.evidence.snapshot.capture_scope, "provider_excerpt")
        self.assertEqual(hit.evidence.element.text, content)
        self.assertEqual(hit.evidence.excerpt, content)
        self.assertEqual(hit.score.normalized, 0.8)
        self.assertNotIn("confidence", hit.model_dump())

    def test_raw_provider_content_remains_original_when_metadata_is_extracted(self) -> None:
        raw = "# numpy.reshape\nParameters:\na array_like\nArray to be reshaped.\n"
        hit = self._collect("A short provider summary.", raw_content=raw)

        self.assertEqual(hit.evidence.snapshot.capture_scope, "provider_excerpt")
        self.assertEqual(hit.evidence.snapshot.parser_config["provider_field"], "raw_content")
        self.assertEqual(hit.evidence.element.text, raw)
        self.assertEqual(hit.evidence.excerpt, raw)
        self.assertIn("doc_metadata", hit.evidence.element.metadata)
        self.assertNotIn("param a", hit.evidence.excerpt)

    def test_query_does_not_change_source_revision_but_content_change_does(self) -> None:
        first = self._collect("Gives an array a new shape.", query="numpy reshape")
        another_query = self._collect("Gives an array a new shape.", query="array shape")
        changed = self._collect("Changes an array's shape in place.", query="numpy reshape")

        self.assertEqual(first.evidence.snapshot.snapshot_id, another_query.evidence.snapshot.snapshot_id)
        self.assertEqual(first.evidence.element.element_id, another_query.evidence.element.element_id)
        self.assertNotEqual(first.evidence.snapshot.snapshot_id, changed.evidence.snapshot.snapshot_id)

    def test_dedup_keeps_distinct_provider_excerpts_at_the_same_url(self) -> None:
        from src.infra.tools.docs_search.ranking import dedupe_docs_hits

        first = self._collect("First exact provider excerpt.")
        second = self._collect("Second exact provider excerpt.")
        hits = dedupe_docs_hits([first, second, first])

        self.assertEqual([hit.evidence.excerpt for hit in hits],
                         ["First exact provider excerpt.", "Second exact provider excerpt."])

    def test_long_content_selection_is_an_exact_slice_with_original_line_breaks(self) -> None:
        content = "Introduction line.\r\n" * 250 + "target_option:\n    Preserve this indentation.\n" + "End.\n" * 400
        hit = self._collect(content, query="target_option")
        selection = hit.evidence.selection

        self.assertGreater(selection.start, 0)
        self.assertEqual(content[selection.start:selection.end], hit.evidence.excerpt)
        self.assertIn("target_option:\n    Preserve this indentation.\n", hit.evidence.excerpt)
        self.assertEqual(hit.evidence.element.text, content)

    def test_missing_provider_score_stays_unknown(self) -> None:
        hit = self._collect("Gives an array a new shape.", score=None)

        self.assertIsNone(hit.score.raw)
        self.assertIsNone(hit.score.normalized)
        self.assertNotIn("score", hit.evidence.model_dump())

    def test_missing_provider_title_uses_source_uri_as_display_name(self) -> None:
        hit = self._collect("Gives an array a new shape.", title=None)

        self.assertEqual(hit.evidence.snapshot.title, hit.evidence.snapshot.source_uri)

    def test_duplicate_selection_keeps_the_best_retrieval_score(self) -> None:
        from src.infra.tools.docs_search.ranking import dedupe_docs_hits

        first = self._collect("Gives an array a new shape.", score=0.3)
        better = self._collect("Gives an array a new shape.", score=0.9)

        self.assertEqual(dedupe_docs_hits([first, better]), [better])


if __name__ == "__main__":
    unittest.main()
