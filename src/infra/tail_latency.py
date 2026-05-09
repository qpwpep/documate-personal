from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
import threading
import time
from typing import Callable, Generic, TypeVar


T = TypeVar("T")
_DEFAULT_HEDGE_LIMITER = threading.BoundedSemaphore(value=8)
_DEFAULT_HEDGE_LIMITER_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class HedgedCallResult(Generic[T]):
    value: T
    hedge_started: bool = False
    hedge_dropped: bool = False
    winner: str = "primary"
    hedges_started: int = 0
    hedges_dropped: int = 0
    suppressed_errors: tuple[str, ...] = field(default_factory=tuple)


def configure_tail_hedge(*, max_concurrency: int) -> None:
    global _DEFAULT_HEDGE_LIMITER
    with _DEFAULT_HEDGE_LIMITER_LOCK:
        _DEFAULT_HEDGE_LIMITER = threading.BoundedSemaphore(value=max(0, int(max_concurrency)))


def invoke_with_optional_hedge(
    invoke: Callable[[], T],
    *,
    hedge_delay_seconds: float,
    max_attempts: int = 2,
    is_success: Callable[[T], bool] | None = None,
    overall_timeout_seconds: float | None = None,
    executor: ThreadPoolExecutor | None = None,
) -> HedgedCallResult[T]:
    """Run a callable and optionally race one duplicate after a short delay.

    Duplicates are only launched while earlier attempts are still pending or
    after a fast semantic failure. This is intended for idempotent,
    quality-equivalent outbound calls where long-tail latency is dominated by
    remote queueing or transport variance.
    """
    delay = max(0.0, float(hedge_delay_seconds or 0.0))
    attempt_limit = max(1, int(max_attempts or 1))
    if delay <= 0 or attempt_limit <= 1:
        return HedgedCallResult(value=invoke())

    owns_executor = executor is None
    pool = executor or ThreadPoolExecutor(
        max_workers=attempt_limit,
        thread_name_prefix="documate-tail-hedge",
    )
    started = time.perf_counter()
    primary = pool.submit(invoke)
    futures: dict[Future[T], str] = {primary: "primary"}
    hedge_started_count = 0
    hedge_dropped_count = 0
    hedge_opportunities_used = 0
    next_hedge_at = started + delay
    deadline = (
        None
        if overall_timeout_seconds is None or overall_timeout_seconds <= 0
        else started + float(overall_timeout_seconds)
    )

    def remaining_timeout(default: float | None = None) -> float | None:
        if deadline is None:
            return default
        remaining = deadline - time.perf_counter()
        return max(0.0, remaining)

    def value_is_successful(value: T) -> bool:
        if is_success is None:
            return True
        try:
            return bool(is_success(value))
        except Exception:
            return False

    try:
        def build_result(value: T, *, winner: str, suppressed_errors: list[str]) -> HedgedCallResult[T]:
            return HedgedCallResult(
                value=value,
                hedge_started=hedge_started_count > 0,
                hedge_dropped=hedge_dropped_count > 0,
                winner=winner,
                hedges_started=hedge_started_count,
                hedges_dropped=hedge_dropped_count,
                suppressed_errors=tuple(suppressed_errors),
            )

        def launch_next_hedge() -> bool:
            nonlocal hedge_started_count, hedge_dropped_count, hedge_opportunities_used, next_hedge_at
            if hedge_opportunities_used >= attempt_limit - 1:
                return False
            hedge_opportunities_used += 1
            next_hedge_at = time.perf_counter() + delay
            with _DEFAULT_HEDGE_LIMITER_LOCK:
                hedge_limiter = _DEFAULT_HEDGE_LIMITER
            if not hedge_limiter.acquire(blocking=False):
                hedge_dropped_count += 1
                return False
            label = "hedge" if hedge_started_count == 0 else f"hedge_{hedge_started_count + 1}"
            hedge_started_count += 1
            hedge = pool.submit(invoke)
            hedge.add_done_callback(lambda _future, limiter=hedge_limiter: limiter.release())
            futures[hedge] = label
            return True

        first_unsuccessful: HedgedCallResult[T] | None = None
        first_error: Exception | None = None
        suppressed_errors: list[str] = []

        while futures:
            timeout = remaining_timeout()
            if timeout is not None and timeout <= 0:
                break
            if hedge_opportunities_used < attempt_limit - 1:
                until_next_hedge = max(0.0, next_hedge_at - time.perf_counter())
                timeout = until_next_hedge if timeout is None else min(timeout, until_next_hedge)
            done, _pending = wait(
                list(futures),
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                launch_next_hedge()
                continue
            for future in done:
                winner = futures.pop(future)
                try:
                    value = future.result()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
                    suppressed_errors.append(f"{winner}: {exc}")
                    continue

                result = build_result(
                    value,
                    winner=winner,
                    suppressed_errors=suppressed_errors,
                )
                if value_is_successful(value):
                    for pending_future in futures:
                        pending_future.cancel()
                    return result
                if first_unsuccessful is None:
                    first_unsuccessful = result
                suppressed_errors.append(f"{winner}: unsuccessful_result")
            if not futures:
                launch_next_hedge()

        for pending_future in futures:
            pending_future.cancel()
        if first_unsuccessful is not None:
            return HedgedCallResult(
                value=first_unsuccessful.value,
                hedge_started=hedge_started_count > 0,
                hedge_dropped=hedge_dropped_count > 0,
                winner=first_unsuccessful.winner,
                hedges_started=hedge_started_count,
                hedges_dropped=hedge_dropped_count,
                suppressed_errors=tuple(suppressed_errors),
            )
        if first_error is not None:
            if hasattr(first_error, "add_note") and suppressed_errors:
                first_error.add_note("hedged call failures: " + "; ".join(suppressed_errors))
            raise first_error
        raise TimeoutError("hedged call exceeded overall timeout")
    finally:
        if owns_executor:
            pool.shutdown(wait=False, cancel_futures=True)


__all__ = ["HedgedCallResult", "configure_tail_hedge", "invoke_with_optional_hedge"]
