from pathlib import Path
import pytest

from src.core.answer_schema import AnswerDocument, finalize_answer, text_document, export_answer_text
from src.core.contracts import RuntimeState, PlannerState, ResponseState
from src.core.contracts.graph_state import PendingAction
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import RequestContract, WireRequestContract
from src.infra.tools.save_text import build_save_text_tool
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.actions.receipts import build_save_receipt
from src.runtime.nodes.planner import make_planner_node


def test_success_claim_without_artifact_proof_is_not_success(tmp_path):
    receipt = build_save_receipt({"status": "success", "file_path": str(tmp_path / "missing.txt")})
    assert receipt.status != "success"


@pytest.mark.parametrize("result", [None, {}, {"status": "nonsense"}])
def test_missing_or_malformed_tool_result_is_unknown(result):
    receipt = build_save_receipt(result)
    assert receipt.status == "unknown"
    assert receipt.verification == "unverifiable"


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


def test_real_save_receipt_binds_request_answer_and_exact_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    state = _save_state()
    result = make_action_postprocess_node(build_save_text_tool(), lambda **_: {}, False)(state)
    receipt = result["response"].result.actions[0]
    assert receipt.status == "success"
    assert receipt.operation.session_id == "session-a"
    assert receipt.operation.request_id == state["runtime"].request_contract.request_id
    assert receipt.operation.answer_hash == state["response"].result.content_hash
    assert receipt.verification == "verified"
    expected = export_answer_text(state["response"].result, include_sources=True).encode("utf-8-sig")
    assert Path(receipt.file_path).read_bytes() == expected
    assert receipt.artifact.byte_count == len(expected)


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


def test_completed_pending_save_is_reverified_without_another_tool_call(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    state = _save_state()
    node = make_action_postprocess_node(build_save_text_tool(), lambda **_: {}, False)
    first = node(state)["response"].result.actions[0]
    pending = PendingAction(contract=state["runtime"].request_contract, response=state["response"].result,
                            completed_actions=("save_text",), save_operation=first.operation,
                            save_receipt=first, phase="awaiting_destination")
    state["runtime"] = state["runtime"].model_copy(update={"pending_action": pending})
    replay = node(state)
    assert replay["response"].result.actions[0] == first
    assert replay.get("messages", []) == []
    Path(first.file_path).write_bytes(b"tampered")
    corrupted = node(state)
    assert corrupted["response"].result.actions[0].status == "error"
    assert "save_text" not in corrupted["runtime"].pending_action.completed_actions


def test_completed_receipt_cannot_claim_a_different_current_answer_was_saved(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    state = _save_state()
    node = make_action_postprocess_node(build_save_text_tool(), lambda **_: {}, False)
    first = node(state)["response"].result.actions[0]
    pending = PendingAction(contract=state["runtime"].request_contract, response=state["response"].result,
                            completed_actions=("save_text",), save_operation=first.operation,
                            save_receipt=first, phase="awaiting_destination")
    state["runtime"] = state["runtime"].model_copy(update={"pending_action": pending})
    state["response"] = state["response"].model_copy(update={"result": finalize_answer(text_document("다른 답변"), [])})
    result = node(state)
    assert result["response"].result.actions[0].status != "success"
    assert Path(first.file_path).read_bytes() == "정확한 저장 본문".encode("utf-8-sig")
    assert len(list(tmp_path.glob("*.txt"))) == 1


def test_separate_unresolved_request_does_not_inherit_old_save(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    original = _save_state()
    node = make_action_postprocess_node(build_save_text_tool(), lambda **_: {}, False)
    receipt = node(original)["response"].result.actions[0]
    pending = PendingAction(contract=original["runtime"].request_contract,
                            response=original["response"].result, save_operation=receipt.operation,
                            save_receipt=receipt, phase="awaiting_delivery")
    new_contract = RequestContract.model_validate({"body": {"kind": "unresolved", "question": "어떤 본문인가요?"}})
    new_state = {**original, "runtime": original["runtime"].model_copy(update={
        "request_contract": new_contract, "pending_action": pending,
    })}
    waiting = node(new_state)["runtime"].pending_action
    assert waiting.contract.request_id == new_contract.request_id
    assert waiting.save_operation is None
    assert waiting.save_receipt is None
    assert Path(receipt.file_path).read_bytes() == "정확한 저장 본문".encode("utf-8-sig")


def test_corrected_answer_with_same_export_gets_its_own_binding(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    original = _save_state("a b")
    node = make_action_postprocess_node(build_save_text_tool(), lambda **_: {}, False)
    first = node(original)["response"].result.actions[0]
    corrected = finalize_answer(AnswerDocument.model_validate({"blocks": [{
        "type": "paragraph", "content": [{"text": "a"}, {"text": "b"}],
    }]}), [])
    assert export_answer_text(corrected) == export_answer_text(original["response"].result)
    assert corrected.content_hash != original["response"].result.content_hash
    pending = PendingAction(contract=original["runtime"].request_contract,
                            response=original["response"].result, save_operation=first.operation,
                            save_receipt=first, body_prepared=True, phase="awaiting_delivery")
    query = "본문의 a와 b를 각각 독립된 내용 단위로 수정해줘"

    class CorrectionPlannerBoundary:
        def invoke(self, _messages):
            return PlannerOutput(use_retrieval=False, tasks=[], request_contract=WireRequestContract.model_validate({
                "relation": "correction", "target_request_id": pending.contract.request_id,
                "body": {"kind": "transform_answer", "source": {"ref": "pending"},
                         "instruction": query, "evidence_ids": ["change"]},
                "evidence": [{"id": "change", "turn_id": "turn-b", "quote": query,
                              "scope": "current_request", "interpretation": "instruction"}],
            }))

    state = {**original, "runtime": original["runtime"].model_copy(update={
        "request_contract": None, "pending_action": pending, "user_input": query, "current_turn_id": "turn-b",
    })}
    state.update(make_planner_node(CorrectionPlannerBoundary(), False)(state))
    contract = state["runtime"].request_contract
    state["response"] = ResponseState(result=corrected, kind="answer", request_id=contract.request_id,
                                      contract_revision=contract.revision)
    second = node(state)["response"].result.actions[0]
    assert second.status == "success"
    assert second.operation.answer_hash == corrected.content_hash
    assert second.operation.operation_id != first.operation.operation_id
    assert second.file_path != first.file_path
    assert Path(first.file_path).read_bytes() == Path(second.file_path).read_bytes()


@pytest.mark.parametrize("same_export", [False, True])
def test_unchanged_save_obligation_rejects_answer_drift(tmp_path, monkeypatch, same_export):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    state = _save_state("a b")
    node = make_action_postprocess_node(build_save_text_tool(), lambda **_: {}, False)
    first = node(state)["response"].result.actions[0]
    # A lost delivery result leaves an unfinished operation for the same body.
    pending = PendingAction(contract=state["runtime"].request_contract, response=state["response"].result,
                            save_operation=first.operation, save_receipt=first,
                            body_prepared=True, phase="awaiting_delivery")
    document = AnswerDocument.model_validate({"blocks": [{
        "type": "paragraph", "content": [{"text": "a"}, {"text": "b"}],
    }]}) if same_export else text_document("different body")
    state["runtime"] = state["runtime"].model_copy(update={"pending_action": pending})
    state["response"] = state["response"].model_copy(update={"result": finalize_answer(document, [])})

    result = node(state)
    receipt = result["response"].result.actions[0]
    assert receipt.status == "error"
    assert receipt.error_code == "idempotency_conflict"
    assert receipt.operation == first.operation
    assert Path(first.file_path).read_bytes() == "a b".encode("utf-8-sig")
    assert len(list(tmp_path.glob("*.txt"))) == 1
