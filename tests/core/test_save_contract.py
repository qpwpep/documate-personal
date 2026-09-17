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












