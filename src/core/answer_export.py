"""Exact answer exports share one checked snapshot and one stable artifact identity."""
from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from src.core.answer_schema import AnswerResponse, export_answer_text
from src.core.decisions import decision_fingerprint
from src.core.request_contracts import RequestContract, check_answer_contract
from src.core.save_contract import SaveOperation


def answer_export_snapshot(response: AnswerResponse) -> AnswerResponse:
    """Revalidate the recorded body, source bindings and checks; remove only receipts."""
    return AnswerResponse.model_validate(response.model_dump(mode="json", exclude={"actions"}))


def exact_copy_ready(contract: RequestContract | None, source: AnswerResponse | None) -> bool:
    """New semantic obligations still need evaluation; an unchanged export does not."""
    if contract is None or source is None or contract.body.kind != "copy_answer" or not contract.can_prepare_body():
        return False
    checked = answer_export_snapshot(source)
    if checked.content_hash != contract.body.source.response_hash or not checked.content.blocks:
        return False
    check = check_answer_contract(contract.answer, checked.content,
                                  evidence=[citation.evidence for citation in checked.citations])
    return check.valid and not check.unchecked_semantic


def build_answer_export_operation(response: AnswerResponse, *, session_id: str) -> SaveOperation:
    snapshot = answer_export_snapshot(response)
    fingerprint = answer_export_key(snapshot)
    operation_id = uuid5(NAMESPACE_URL, f"answer-export:{session_id}:{fingerprint}").hex
    return SaveOperation.for_text(
        export_answer_text(snapshot, include_sources=True), operation_id=operation_id,
        session_id=session_id, request_id=f"answer-export:{fingerprint}", contract_revision=1,
        target_kind="copy_answer", source_hash=snapshot.content_hash, answer_hash=snapshot.content_hash,
    )


def answer_export_key(response: AnswerResponse) -> str:
    return decision_fingerprint(answer_export_snapshot(response).model_dump(mode="json"))
