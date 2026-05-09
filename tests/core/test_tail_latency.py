from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest

from src.infra.tail_latency import configure_tail_hedge, invoke_with_optional_hedge


class TailLatencyTest(unittest.TestCase):
    def test_invoke_with_optional_hedge_returns_primary_when_fast(self) -> None:
        call_count = 0

        def invoke() -> str:
            nonlocal call_count
            call_count += 1
            return "primary"

        with ThreadPoolExecutor(max_workers=2) as executor:
            result = invoke_with_optional_hedge(
                invoke,
                hedge_delay_seconds=0.05,
                executor=executor,
            )

        self.assertEqual(result.value, "primary")
        self.assertFalse(result.hedge_started)
        self.assertEqual(result.winner, "primary")
        self.assertEqual(call_count, 1)

    def test_invoke_with_optional_hedge_returns_first_successful_duplicate(self) -> None:
        lock = threading.Lock()
        call_count = 0

        def invoke() -> str:
            nonlocal call_count
            with lock:
                call_count += 1
                call_number = call_count
            if call_number == 1:
                time.sleep(0.05)
                return "slow"
            return "fast"

        with ThreadPoolExecutor(max_workers=2) as executor:
            result = invoke_with_optional_hedge(
                invoke,
                hedge_delay_seconds=0.005,
                executor=executor,
            )

        self.assertEqual(result.value, "fast")
        self.assertTrue(result.hedge_started)
        self.assertEqual(result.winner, "hedge")
        self.assertEqual(result.hedges_started, 1)
        self.assertEqual(call_count, 2)

    def test_invoke_with_optional_hedge_can_start_second_duplicate(self) -> None:
        lock = threading.Lock()
        call_count = 0

        def invoke() -> str:
            nonlocal call_count
            with lock:
                call_count += 1
                call_number = call_count
            if call_number < 3:
                time.sleep(0.05)
                return f"slow-{call_number}"
            return "fast"

        with ThreadPoolExecutor(max_workers=3) as executor:
            result = invoke_with_optional_hedge(
                invoke,
                hedge_delay_seconds=0.005,
                max_attempts=3,
                executor=executor,
            )

        self.assertEqual(result.value, "fast")
        self.assertTrue(result.hedge_started)
        self.assertEqual(result.winner, "hedge_2")
        self.assertEqual(result.hedges_started, 2)
        self.assertEqual(call_count, 3)

    def test_invoke_with_optional_hedge_waits_for_semantic_success_after_failed_primary(self) -> None:
        lock = threading.Lock()
        call_count = 0

        def invoke() -> str:
            nonlocal call_count
            with lock:
                call_count += 1
                call_number = call_count
            if call_number == 1:
                time.sleep(0.02)
                return "bad"
            time.sleep(0.03)
            return "good"

        with ThreadPoolExecutor(max_workers=2) as executor:
            result = invoke_with_optional_hedge(
                invoke,
                hedge_delay_seconds=0.005,
                is_success=lambda value: value == "good",
                executor=executor,
            )

        self.assertEqual(result.value, "good")
        self.assertTrue(result.hedge_started)
        self.assertEqual(result.winner, "hedge")
        self.assertIn("primary: unsuccessful_result", result.suppressed_errors)

    def test_invoke_with_optional_hedge_launches_hedge_after_fast_semantic_failure(self) -> None:
        call_count = 0

        def invoke() -> str:
            nonlocal call_count
            call_count += 1
            return "bad" if call_count == 1 else "good"

        with ThreadPoolExecutor(max_workers=2) as executor:
            result = invoke_with_optional_hedge(
                invoke,
                hedge_delay_seconds=0.05,
                is_success=lambda value: value == "good",
                executor=executor,
            )

        self.assertEqual(result.value, "good")
        self.assertTrue(result.hedge_started)
        self.assertEqual(result.winner, "hedge")
        self.assertEqual(call_count, 2)

    def test_invoke_with_optional_hedge_launches_hedge_after_fast_exception(self) -> None:
        call_count = 0

        def invoke() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("transient")
            return "good"

        with ThreadPoolExecutor(max_workers=2) as executor:
            result = invoke_with_optional_hedge(
                invoke,
                hedge_delay_seconds=0.05,
                executor=executor,
            )

        self.assertEqual(result.value, "good")
        self.assertTrue(result.hedge_started)
        self.assertEqual(result.winner, "hedge")
        self.assertEqual(call_count, 2)
        self.assertIn("primary: transient", result.suppressed_errors)

    def test_invoke_with_optional_hedge_respects_overall_timeout(self) -> None:
        def invoke() -> str:
            time.sleep(0.05)
            return "late"

        with ThreadPoolExecutor(max_workers=2) as executor:
            with self.assertRaises(TimeoutError):
                invoke_with_optional_hedge(
                    invoke,
                    hedge_delay_seconds=0.005,
                    overall_timeout_seconds=0.01,
                    executor=executor,
                )

    def test_invoke_with_optional_hedge_reports_limiter_drop(self) -> None:
        call_count = 0
        release_primary = threading.Event()

        def invoke() -> str:
            nonlocal call_count
            call_count += 1
            release_primary.wait(timeout=1.0)
            return "primary"

        configure_tail_hedge(max_concurrency=0)
        try:
            timer = threading.Timer(0.03, release_primary.set)
            timer.start()
            with ThreadPoolExecutor(max_workers=2) as executor:
                result = invoke_with_optional_hedge(
                    invoke,
                    hedge_delay_seconds=0.001,
                    executor=executor,
                )
        finally:
            release_primary.set()
            configure_tail_hedge(max_concurrency=8)
            timer.cancel()

        self.assertEqual(result.value, "primary")
        self.assertFalse(result.hedge_started)
        self.assertTrue(result.hedge_dropped)
        self.assertEqual(result.hedges_dropped, 1)
        self.assertEqual(call_count, 1)


if __name__ == "__main__":
    unittest.main()
