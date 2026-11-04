"""New judge preflight failures preserve existing decision-contract semantics."""

from __future__ import annotations

import pytest

from src.eval.decisions import CaseDecision, refresh_case_decision
from src.eval.result_models import CaseResult
from tests.eval.test_release_eval_contract import _case_result


def _unobserved_result(reason, *, eval_validity="incomplete"):
    case, result = _case_result()
    result.product_pass = None
    result.composite_quality_score = None
    result.eval_validity = eval_validity
    result.judge_status = "not_run"
    result.judge_status_reason = reason
    result.judge_input_issues = [reason]
    result.judge_pass = None
    result.llm_judge_score = None
    result.judge_subscores = None
    return case, result


@pytest.mark.parametrize("reason", ["input_budget_exceeded", "input_invalid"])
@pytest.mark.parametrize("eval_validity", ["incomplete", "invalid"])
def test_new_preflight_failure_is_not_a_product_quality_failure(reason, eval_validity):
    case, result = _unobserved_result(reason, eval_validity=eval_validity)

    decision = refresh_case_decision(result, case)

    assert result.product_pass is None
    assert result.composite_quality_score is None
    assert decision.passed is False
    assert result.release_pass is False
    assert "product_quality_below_floor" not in decision.failure_codes
    assert "judge_not_passed" in decision.failure_codes
    assert ("invalid_eval" if eval_validity == "invalid" else "incomplete_eval") in decision.failure_codes


def test_existing_unknown_quality_semantics_are_preserved_with_a_successful_judge():
    case, result = _case_result()
    result.product_pass = None
    result.composite_quality_score = None

    decision = refresh_case_decision(result, case)

    assert decision.passed is False
    assert result.release_pass is False
    assert "product_quality_below_floor" in decision.failure_codes
    assert "product_quality_unobserved" not in decision.failure_codes


@pytest.mark.parametrize("reason", ["input_budget_exceeded", "input_invalid"])
def test_new_preflight_refresh_and_reload_do_not_preserve_a_false_product_failure_label(reason):
    case, result = _unobserved_result(reason)
    result.gate_failures = ["product_quality_below_floor"]

    refresh_case_decision(result, case)
    reloaded = CaseResult.model_validate(result.model_dump(mode="json"))

    assert reloaded.release_pass is False
    assert "product_quality_below_floor" not in reloaded.gate_failures
    assert "product_quality_unobserved" in reloaded.gate_failures


@pytest.mark.parametrize("reason", ["input_incomplete", "missing_final_response"])
def test_existing_version_two_failure_codes_survive_reload(reason):
    _, result = _unobserved_result(reason)
    existing_codes = ["incomplete_eval", "judge_not_passed", "product_quality_below_floor"]
    result.gate_failures = existing_codes
    result.decision = CaseDecision(passed=False, failure_codes=existing_codes)
    result.release_pass = False

    reloaded = CaseResult.model_validate(result.model_dump(mode="json"))

    assert reloaded.release_pass is False
    assert reloaded.gate_failures == existing_codes
    assert reloaded.decision.failure_codes == existing_codes


def test_observed_product_quality_failure_still_blocks_release():
    case, result = _case_result()
    result.product_pass = False
    result.composite_quality_score = 0.5

    decision = refresh_case_decision(result, case)

    assert decision.passed is False
    assert result.release_pass is False
    assert "product_quality_below_floor" in decision.failure_codes
    assert "product_quality_unobserved" not in decision.failure_codes


def test_unobserved_quality_does_not_remove_a_runtime_failure():
    case, result = _unobserved_result("input_budget_exceeded")
    result.runtime_errors = ["request timeout"]

    decision = refresh_case_decision(result, case)

    assert decision.passed is False
    assert "runtime_error" in decision.failure_codes
    assert "incomplete_eval" in decision.failure_codes
    assert "product_quality_below_floor" not in decision.failure_codes
