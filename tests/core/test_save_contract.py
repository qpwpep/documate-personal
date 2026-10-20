from pathlib import Path

import pytest

from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts import RuntimeState, PlannerState, ResponseState
from src.core.contracts.graph_state import PendingAction
from src.core.request_contracts import RequestContract
from src.eval.config_models import BenchmarkCase
from src.eval.decisions import policy_for_case
from src.eval.tool_policy import assess_execution_policy
from src.infra.tools.save_text import build_save_text_tool
from src.runtime.agent_runtime.tool_execution import capture_tool_execution
from src.runtime.nodes.actions import make_action_postprocess_node


def _save_state(text="정확한 저장 본문", *, session_id="session-a"):
    contract = RequestContract.model_validate({
        "actions": {"save_text": {"intent": "requested", "evidence_ids": ["save"]}},
        "evidence": [{"id": "save", "turn_id": "turn-a", "quote": "저장해줘",
                      "scope": "current_request", "interpretation": "instruction"}],
    })
    return {
        "runtime": RuntimeState(user_input="저장해줘", session_id=session_id, request_contract=contract),
        "planner": PlannerState(),
        "response": ResponseState(result=finalize_answer(text_document(text), []), kind="answer",
                                  request_id=contract.request_id, contract_revision=contract.revision),
    }


def test_reentering_same_save_action_reuses_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    state = _save_state()
    node = make_action_postprocess_node(build_save_text_tool(), lambda **_: {}, False)
    first = node(state)["response"].result.actions[0]
    before = Path(first.file_path).stat().st_mtime_ns
    second = node(state)["response"].result.actions[0]
    assert first.operation.operation_id == second.operation.operation_id
    assert first.artifact == second.artifact
    assert Path(first.file_path).stat().st_mtime_ns == before
    assert len(list(tmp_path.glob("*.txt"))) == 1


@pytest.mark.parametrize("failure", ["changed_body", "missing_artifact", "unreadable_artifact"])
def test_rejected_save_reuse_records_new_blocked_attempt(tmp_path, monkeypatch, failure):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    state = _save_state()
    node = make_action_postprocess_node(build_save_text_tool(), lambda **_: {}, False)
    contract = state["runtime"].request_contract
    with capture_tool_execution("request") as recorder:
        first = node(state)["response"].result.actions[0]
        pending = PendingAction(
            contract=contract, response=state["response"].result, body_prepared=True,
            completed_actions=("save_text",), save_operation=first.operation, save_receipt=first,
        )
        state["runtime"] = state["runtime"].model_copy(update={"pending_action": pending})
        if failure == "changed_body":
            state["response"] = state["response"].model_copy(update={
                "result": finalize_answer(text_document("변경된 저장 본문"), []),
            })
        elif failure == "missing_artifact":
            Path(first.file_path).unlink()
        else:
            read_bytes = Path.read_bytes

            def read_existing_artifact(path):
                if path == Path(first.file_path):
                    raise PermissionError("Artifact read is unavailable.")
                return read_bytes(path)

            monkeypatch.setattr(Path, "read_bytes", read_existing_artifact)
        receipt = node(state)["response"].result.actions[-1]
    evidence = recorder.snapshot()

    assert receipt.status == ("unknown" if failure == "unreadable_artifact" else "error")
    assert receipt.invocation_id != first.invocation_id
    assert evidence.events[-1].phase == "blocked"
    assert evidence.events[-1].invocation_id == receipt.invocation_id
    assert evidence.events[-1].reason_code == receipt.error_code
    policy = policy_for_case(BenchmarkCase(
        case_id="save-reuse", category="tool_action", query="저장해줘.", expected_tools=["save_text"],
    ))
    assessment = assess_execution_policy(
        policy=policy, evidence=evidence, request_id="request", prior_turns=[], actions=[receipt],
        tool_calls=["save_text"], tool_call_count=1,
    )
    assert assessment.status == "compliant"
