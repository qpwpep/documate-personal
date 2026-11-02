"""Exact answer exports share one checked snapshot and one stable artifact identity."""
from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from src.core.answer_schema import AnswerResponse, finalize_answer, text_document
from src.core.answer_schema.rendering import render_answer_text
from src.core.decisions import decision_fingerprint
from src.core.request_contracts import BoundInputText, RequestContract, check_answer_contract
from src.core.save_contract import SaveOperation


def answer_export_snapshot(response: AnswerResponse) -> AnswerResponse:
    """Revalidate the recorded body, source bindings and checks; remove only receipts."""
    return AnswerResponse.model_validate(response.model_dump(mode="json", exclude={"actions"}))


def input_copy_snapshot(contract: RequestContract) -> AnswerResponse:
    """Revalidate the bound input and derive the only permitted literal copy."""
    if contract.body.kind != "copy_input":
        raise ValueError("an input copy requires bound input text")
    source = BoundInputText.model_validate(contract.body.source.model_dump(mode="json"))
    return finalize_answer(text_document(source.text), [])


def exact_copy_ready(contract: RequestContract | None, source: AnswerResponse | None) -> bool:
    """New semantic obligations still need evaluation; literal fidelity does not."""
    if contract is None or not contract.can_prepare_body():
        return False
    if contract.body.kind == "copy_input":
        checked = input_copy_snapshot(contract)
    elif contract.body.kind == "copy_answer" and source is not None:
        checked = answer_export_snapshot(source)
        if checked.content_hash != contract.body.source.response_hash:
            return False
    else:
        return False
    if not checked.content.blocks:
        return False
    check = check_answer_contract(contract.answer, checked.content,
                                  evidence=[citation.evidence for citation in checked.citations])
    return check.valid and not check.unchecked_semantic


def build_answer_export_operation(response: AnswerResponse, *, session_id: str) -> SaveOperation:
    snapshot = answer_export_snapshot(response)
    fingerprint = _snapshot_export_key(snapshot)
    operation_id = uuid5(NAMESPACE_URL, f"answer-export:{session_id}:{fingerprint}").hex
    return SaveOperation.for_text(
        render_answer_text(snapshot, include_sources=True), operation_id=operation_id,
        session_id=session_id, request_id=f"answer-export:{fingerprint}", contract_revision=1,
        target_kind="copy_answer", source_hash=snapshot.content_hash, answer_hash=snapshot.content_hash,
    )


def answer_export_key(response: AnswerResponse) -> str:
    """Identify the checked export independently of its latest semantic report."""
    return _snapshot_export_key(answer_export_snapshot(response))


def _snapshot_export_key(snapshot: AnswerResponse) -> str:
    # Reports stay in displayed snapshots. Rechecking an unchanged body and
    # cited sources must not create a new obligation for the same export bytes.
    return decision_fingerprint(snapshot.model_dump(mode="json", exclude={"semantic_validation"}))
