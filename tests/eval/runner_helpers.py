"""Prepare real case weights for tests that enter the runner after preflight."""

from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.online_runner import _run_single_case
from src.eval.online_runner.result_builder import build_case_result
from src.eval.result_models import CaseResult
from src.eval.weighting import resolve_case_weights


def run_case_with_weights(*, case: BenchmarkCase, config: BenchmarkConfig, **kwargs) -> CaseResult:
    return _run_single_case(
        case=case,
        config=config,
        resolved_weights=resolve_case_weights(case=case, profiles=config.weights),
        **kwargs,
    )


def build_result_with_weights(*, case: BenchmarkCase, config: BenchmarkConfig, **kwargs) -> CaseResult:
    return build_case_result(
        case=case,
        config=config,
        resolved_weights=resolve_case_weights(case=case, profiles=config.weights),
        **kwargs,
    )
