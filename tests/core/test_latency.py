import unittest

from src.core.latency import build_latency_breakdown


class LatencyTest(unittest.TestCase):
    def test_build_latency_breakdown_keeps_structured_empty_fallback_attempt(self) -> None:
        breakdown = build_latency_breakdown(
            raw_trace=[
                {
                    "kind": "synthesis_attempt",
                    "attempt": 1,
                    "mode": "structured_empty_fallback",
                    "structured_ms": 12,
                    "fallback_ms": 0,
                    "total_ms": 12,
                }
            ]
        )

        self.assertEqual(len(breakdown.synthesis_attempts), 1)
        self.assertEqual(breakdown.synthesis_attempts[0].mode, "structured_empty_fallback")


if __name__ == "__main__":
    unittest.main()
