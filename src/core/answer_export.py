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


def input_copy_snapshot(source: BoundInputText) -> AnswerResponse:
    """Derive the literal copy from input already validated and bound by the planner."""
    return finalize_answer(text_document(source.text), [])


def exact_copy_ready(contract: RequestContract | None, source: AnswerResponse | None) -> bool:
    """Check the current contract against an already checked copy source."""
    if contract is None or not contract.can_prepare_body() or source is None:
        return False
    if contract.body.kind == "copy_answer":
        if source.content_hash != contract.body.source.response_hash:
            return False
    elif contract.body.kind != "copy_input":
        return False
    if not source.content.blocks:
        return False
    check = check_answer_contract(contract.answer, source.content,
                                  evidence=[citation.evidence for citation in source.citations])
    return check.valid and not check.unchecked_semantic


def build_answer_export_operation(snapshot: AnswerResponse, *, session_id: str) -> SaveOperation:
    """Bind a validated answer snapshot to its stable export operation."""
    fingerprint = answer_export_key(snapshot)
    operation_id = uuid5(NAMESPACE_URL, f"answer-export:{session_id}:{fingerprint}").hex
    return SaveOperation.for_text(
        render_answer_text(snapshot, include_sources=True), operation_id=operation_id,
        session_id=session_id, request_id=f"answer-export:{fingerprint}", contract_revision=1,
        target_kind="copy_answer", source_hash=snapshot.content_hash, answer_hash=snapshot.content_hash,
    )


def answer_export_key(snapshot: AnswerResponse) -> str:
    """Identify a validated export independently of its receipts and semantic report."""
    # Reports stay in displayed snapshots. Rechecking an unchanged body and
    # cited sources must not create a new obligation for the same export bytes.
    return decision_fingerprint({
        **snapshot.model_dump(mode="json", exclude={"actions", "semantic_validation"}),
        "actions": [],
    })
