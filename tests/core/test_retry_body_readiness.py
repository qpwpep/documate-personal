import pytest

from src.core.contracts import RetryState
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import RequestContract
from src.runtime.nodes.retry import build_retry_update


@pytest.mark.parametrize("body,missing,expected", [
    ({"kind": "compose", "instruction": "리스트를 설명"},
     [{"slot": "slack_intent", "reason": "unclear", "question": "보낼까요?"}], True),
    ({"kind": "unresolved", "question": "어떤 API인가요?"}, [], False),
    ({"kind": "copy_answer", "source": {"ref": "previous", "response_hash": "owned-answer"}}, [], False),
])
def test_body_repair_uses_canonical_body_readiness_independently_of_action_questions(body, missing, expected):
    contract = RequestContract.model_validate({"body": body, "missing_info": missing})
    plan = PlannerOutput(use_retrieval=False, tasks=[])

    retry, state, _ = build_retry_update(
        retry_context=RetryState(max_retries=2), retry_reason="missing_content",
        planner_output=plan, request_contract=contract,
        retrieval_errors=[], score_avg=None,
    )

    assert retry is expected
    assert state.needs_retry is expected
    assert state.attempt == int(expected)
    if expected:
        assert state.retry_scope == "reuse_hits_resynthesize"
        assert state.failed_routes == []
