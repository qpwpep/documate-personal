from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from src.infra.tools.docs_search.serialization import DocsSearchFilterCounters, collect_docs_search_evidence
from src.infra.tools.docs_search.url_validation import DocUrlValidationResult


class DocsSearchSerializationTest(unittest.TestCase):
    def test_collect_docs_search_evidence_validates_same_priority_urls_in_parallel(self) -> None:
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
            evidence, _raw_scores = collect_docs_search_evidence(
                results,
                allowed_domains=["numpy.org"],
                retrieval_warnings=[],
                query="numpy reference",
                filter_counters=counters,
            )

        self.assertEqual(len(evidence), 3)
        self.assertEqual(counters.validated_url_count, 3)
        self.assertGreater(max_active, 1)


if __name__ == "__main__":
    unittest.main()
