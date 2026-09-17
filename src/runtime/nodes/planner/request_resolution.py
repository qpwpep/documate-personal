from __future__ import annotations

import hashlib
from uuid import uuid4

from src.core.contracts import GraphState
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.contracts.graph_state import PendingAction
from src.core.request_contracts import (
    AcknowledgeBody, ActionContract, ActionRequest, AnswerContract, AnswerReference, BoundAnswerReference,
    BoundInputText, ComposeBody, CopyAnswerBody, CopyInputBody, MissingInformation,
    RequestContract, TransformAnswerBody, TransformInputBody, UnresolvedBody,
    WireCopyAnswerBody, WireCopyInputBody, WireRequestContract, WireTransformAnswerBody,
    WireTransformInputBody, validate_contract_evidence,
)
from src.runtime.nodes.planner.prompt_builder import planner_user_utterances


def _missing(slot: str, reason: str, question: str) -> MissingInformation:
    return MissingInformation(slot=slot, reason=reason, question=question)


def _bind_local_evidence(proposal: WireRequestContract, *, request_id: str, revision: int, turn_id: str) -> WireRequestContract:
    """Local model labels cannot address or overwrite a prior revision's evidence."""
    ids = {item.id: f"ev:{request_id}:{revision}:{turn_id}:{index}"
           for index, item in enumerate(proposal.evidence, 1)}
    data = proposal.model_dump(mode="python")
    data["evidence"] = tuple(item.model_copy(update={"id": ids[item.id]}) for item in proposal.evidence)
    for name in ("save_text", "slack_notify"):
        data["actions"][name]["evidence_ids"] = tuple(ids[item] for item in data["actions"][name]["evidence_ids"])
    for group in ("content", "format"):
        for item in data["answer"][group]:
            item["evidence_ids"] = tuple(ids[key] for key in item["evidence_ids"])
    if "evidence_ids" in data["body"]:
        data["body"]["evidence_ids"] = tuple(ids[key] for key in data["body"]["evidence_ids"])
    return WireRequestContract.model_validate(data)


def _bind_input(body, utterances: dict[str, str], original: RequestContract | None):
    selector = body.source
    text = utterances.get(selector.turn_id)
    if text is None:
        if original is not None and isinstance(original.body, (CopyInputBody, TransformInputBody)):
            retained = original.body.source
            if retained.turn_id == selector.turn_id and retained.text == selector.quote:
                return retained, None
        return None, _missing("input_reference", "unknown_id", "원문 발화를 찾지 못했습니다. 작업할 텍스트를 다시 지정해 주세요.")
    starts = []
    start = text.find(selector.quote)
    while start >= 0:
        starts.append(start)
        start = text.find(selector.quote, start + 1)
    if not starts:
        return None, _missing("input_reference", "quote_not_found", "지정한 문구가 원문과 일치하지 않습니다. 작업할 텍스트를 정확히 지정해 주세요.")
    if selector.occurrence is None and len(starts) != 1:
        return None, _missing("input_reference", "ambiguous_scope", "같은 문구가 여러 번 있습니다. 어느 부분을 사용할지 알려 주세요.")
    occurrence = selector.occurrence or 1
    if occurrence > len(starts):
        return None, _missing("input_reference", "ambiguous_scope", "지정한 순서의 문구를 찾지 못했습니다. 사용할 부분을 알려 주세요.")
    start = starts[occurrence - 1]
    return BoundInputText(turn_id=selector.turn_id, text=selector.quote, start=start, end=start + len(selector.quote),
                          content_hash=hashlib.sha256(selector.quote.encode("utf-8")).hexdigest()), None


def _bind_body(body, runtime, utterances, original=None, *, declared_missing=()):
    if isinstance(body, (WireCopyInputBody, WireTransformInputBody)):
        selected, missing = _bind_input(body, utterances, original)
        if missing:
            return UnresolvedBody(question=missing.question, instruction=getattr(body, "instruction", "")), (missing,)
        if isinstance(body, WireCopyInputBody):
            return CopyInputBody(source=selected), ()
        return TransformInputBody(source=selected, instruction=body.instruction, evidence_ids=body.evidence_ids), ()
    if isinstance(body, (WireCopyAnswerBody, WireTransformAnswerBody)):
        response = runtime.previous_response if body.source.ref == "previous" else (runtime.pending_action.response if runtime.pending_action else None)
        if response is None or not response.content.blocks:
            missing = _missing("answer_reference", "unknown_id", "참조한 답변이 준비되어 있지 않습니다. 사용할 본문을 지정해 주세요.")
            return UnresolvedBody(question=missing.question, instruction=getattr(body, "instruction", "")), (missing,)
        source = BoundAnswerReference(ref=body.source.ref, response_hash=response.content_hash)
        if isinstance(body, WireCopyAnswerBody):
            return CopyAnswerBody(source=source), ()
        return TransformAnswerBody(source=source, instruction=body.instruction, evidence_ids=body.evidence_ids), ()
    if isinstance(body, UnresolvedBody):
        return body, (() if any(item.slot != "slack_destination" for item in declared_missing)
                      else (_missing("subject", "unclear", body.question),))
    return body, ()


def _merge_pending(proposal: WireRequestContract, pending, *, current_turn_id: str):
    original = pending.contract
    evidence = {item.id: item for item in original.evidence}
    current_ids = set()
    for item in proposal.evidence:
        previous = evidence.get(item.id)
        if previous is not None and previous != item:
            raise ValueError("pending evidence identity cannot be reassigned")
        if previous is None and item.turn_id == current_turn_id:
            current_ids.add(item.id)
        evidence[item.id] = item

    actions = {}
    updated_intent_slots = set()
    for name in ("save_text", "slack_notify"):
        incoming = getattr(proposal.actions, name)
        explicit = bool(current_ids.intersection(incoming.evidence_ids))
        if explicit and incoming.intent in {"requested", "forbidden", "unresolved"}:
            updated_intent_slots.add("save_intent" if name == "save_text" else "slack_intent")
        selected = incoming if incoming.intent in {"forbidden", "unresolved"} or explicit else getattr(original.actions, name)
        if name in pending.completed_actions and selected.intent == "requested" and not (incoming.intent == "requested" and explicit):
            selected = ActionRequest()
        actions[name] = selected

    def requirements(previous, incoming):
        result = list(previous)
        for item in incoming:
            if item in result:
                continue
            if not current_ids.intersection(item.evidence_ids):
                raise ValueError("pending requirements need a new user instruction to change")
            result = [old for old in result if old.kind != item.kind]
            result.append(item)
        return tuple(result)

    answer = AnswerContract(content=requirements(original.answer.content, proposal.answer.content),
                            format=requirements(original.answer.format, proposal.answer.format),
                            preferences=tuple(dict.fromkeys([*original.answer.preferences, *proposal.answer.preferences])))
    body = proposal.body
    neutral_body = isinstance(body, ComposeBody) and not body.instruction.strip() and not body.evidence_ids
    unfinished_body_copy = (proposal.relation == "supplement" and isinstance(body, WireCopyAnswerBody)
                            and body.source.ref == "pending" and not pending.body_prepared)
    missing = list(proposal.missing_info)
    missing.extend(item for item in original.missing_info
                   if item.slot in {"save_intent", "slack_intent"} and item.slot not in updated_intent_slots)
    incoming_destination_gap = next((item for item in missing if item.slot == "slack_destination"), None)
    previous_destination_gap = next((item for item in original.missing_info if item.slot == "slack_destination"), None)
    destination = proposal.slack_destination
    if incoming_destination_gap is not None:
        if incoming_destination_gap.reason != "not_provided" or destination is None or not destination.has_destination():
            destination = None
            if previous_destination_gap is not None and previous_destination_gap.reason != "not_provided":
                missing.append(previous_destination_gap)
    elif destination is None and previous_destination_gap is not None:
        missing.append(previous_destination_gap)
    elif destination is None:
        destination = original.slack_destination
    original_body = original.body_request or original.to_wire().body
    unfinished_content = not pending.body_prepared and (
        original_body.kind in {"transform_input", "transform_answer", "extract"}
        or (isinstance(original_body, ComposeBody) and bool(original_body.instruction.strip()))
        or any(item.slot == "subject" for item in original.missing_info)
    )
    acknowledge_with_remaining_work = isinstance(body, AcknowledgeBody) and proposal.relation != "cancel" and (
        unfinished_content or any(item.intent in {"requested", "unresolved"} for item in actions.values())
    )
    if neutral_body or unfinished_body_copy or acknowledge_with_remaining_work:
        if pending.response is not None and pending.body_prepared:
            body = WireCopyAnswerBody(source=AnswerReference(ref="pending"))
        else:
            body = original_body
            missing.extend(item for item in original.missing_info if item.slot in {"subject", "input_reference", "answer_reference"})
            if isinstance(body, (WireCopyAnswerBody, WireTransformAnswerBody)) and pending.response is not None:
                body = body.model_copy(update={"source": AnswerReference(ref="pending")})
    if proposal.relation == "cancel":
        actions = {name: getattr(proposal.actions, name) for name in actions}
        body = AcknowledgeBody(evidence_ids=getattr(proposal.body, "evidence_ids", ()))
        missing = [item for item in missing if item.slot in {"pending_request", "save_intent", "slack_intent"}]
        destination = None
    elif actions["slack_notify"].intent not in {"requested", "unresolved"}:
        missing = [item for item in missing if item.slot != "slack_destination"]
    retained_ids = set(getattr(body, "evidence_ids", ()))
    for item in (*actions.values(), *answer.content, *answer.format):
        retained_ids.update(item.evidence_ids)
    evidence = {key: item for key, item in evidence.items() if key in retained_ids or item.turn_id == current_turn_id}
    return WireRequestContract.model_validate({
        **proposal.model_dump(mode="python"), "actions": ActionContract(**actions), "answer": answer,
        "body": body, "evidence": tuple(evidence.values()), "missing_info": tuple(missing),
        "slack_destination": destination,
    })


def pending_body_changed(pending: PendingAction, contract: RequestContract) -> bool:
    """Compare accepted body facts, not turn labels or newly bound evidence IDs.

    Copying the checked pending answer continues its frozen delivery even when
    the original request composed or transformed it. An unresolved replacement
    still changes that obligation before its final body can be prepared.
    """
    def answer_facts(answer):
        return {
            "content": tuple(item.model_dump(exclude={"evidence_ids"}) for item in answer.content),
            "format": tuple(item.model_dump(exclude={"evidence_ids"}) for item in answer.format),
            "preferences": answer.preferences,
        }

    if answer_facts(contract.answer) != answer_facts(pending.contract.answer):
        return True
    if (isinstance(contract.body, CopyAnswerBody) and pending.response is not None
            and contract.body.source.response_hash == pending.response.content_hash):
        return False
    body, previous = contract.body, pending.contract.body
    if isinstance(body, UnresolvedBody):
        body = contract.body_request or body
        previous = pending.contract.body_request or previous
    # The same answer can be addressed as previous or pending without changing
    # its bound identity. Input source offsets and hashes remain significant.
    exclude = {"evidence_ids": True, "source": {"ref"}}
    return body.model_dump(exclude=exclude) != previous.model_dump(exclude=exclude)


def resolve_request_contract(proposal: WireRequestContract | None, state: GraphState, *, max_turns: int) -> RequestContract:
    """Bind exact references and preserve confirmed facts independently of missing slots."""
    runtime = get_runtime_state(state)
    if runtime.request_contract is not None:
        return runtime.request_contract
    if proposal is None:
        raise ValueError("initial planner response omitted request_contract")
    utterances = planner_user_utterances(state, max_turns)
    pending = runtime.pending_action
    matches_pending = pending is not None and proposal.target_request_id == pending.contract.request_id
    inherits_pending = matches_pending and proposal.relation in {"correction", "supplement", "cancel"}
    errors = validate_contract_evidence(proposal, utterances)
    if errors:
        raise ValueError("; ".join(errors))
    original = pending.contract if inherits_pending else None
    request_id = original.request_id if original is not None else str(uuid4())
    revision = original.revision + 1 if original is not None else 1
    proposal = _bind_local_evidence(proposal, request_id=request_id, revision=revision, turn_id=runtime.current_turn_id)
    if inherits_pending:
        proposal = _merge_pending(proposal, pending, current_turn_id=runtime.current_turn_id or next(reversed(utterances)))
    missing = list(proposal.missing_info)
    if proposal.target_request_id is not None and not matches_pending:
        missing.append(_missing("pending_request", "unknown_id", "지정한 보류 요청이 없습니다. 처리할 요청을 다시 지정해 주세요."))
    elif proposal.relation in {"supplement", "cancel"} and proposal.target_request_id is None:
        missing.append(_missing("pending_request", "not_provided", "보충하거나 취소할 요청을 특정하지 못했습니다. 해당 요청을 알려 주세요."))

    destination_bound = proposal.slack_destination is not None and proposal.slack_destination.has_destination()
    if proposal.slack_destination is not None:
        old_destination = original.slack_destination if original is not None else None
        for name, value in proposal.slack_destination.model_dump().items():
            if value and not (old_destination is not None and getattr(old_destination, name) == value) and not any(value in text for text in utterances.values()):
                missing.append(_missing("slack_destination", "unknown_id", "지정한 Slack 대상을 발화에서 확인하지 못했습니다. 대상을 다시 알려 주세요."))
                destination_bound = False
                break
    body, body_missing = _bind_body(proposal.body, runtime, utterances, original, declared_missing=proposal.missing_info)
    missing.extend(body_missing)
    current_evidence = {item.id for item in proposal.evidence if item.turn_id == runtime.current_turn_id}
    for name, slot in (("save_text", "save_intent"), ("slack_notify", "slack_intent")):
        action = getattr(proposal.actions, name)
        historical_only = action.intent == "requested" and not inherits_pending and not current_evidence.intersection(action.evidence_ids)
        if action.intent == "unresolved" or historical_only:
            missing.append(_missing(slot, "unclear", "저장 여부를 알려 주세요." if name == "save_text" else "Slack으로 전송할지 알려 주세요."))
    configured = runtime.session_metadata.slack_destination
    if proposal.actions.slack_notify.intent == "requested" and not (
        (proposal.slack_destination and proposal.slack_destination.has_destination()) or (configured and configured.has_destination())
    ):
        missing.append(_missing("slack_destination", "not_provided", "Slack으로 보낼 channel_id, user_id 또는 email을 알려 주세요."))
    unique = {}
    missing_priority = {"not_provided": 0, "unclear": 1, "unknown_id": 2, "quote_not_found": 2, "ambiguous_scope": 2}
    for item in missing:
        existing = unique.get(item.slot)
        if existing is None or missing_priority[item.reason] > missing_priority[existing.reason]:
            unique[item.slot] = item
    if destination_bound and "slack_destination" in unique and unique["slack_destination"].reason == "not_provided":
        del unique["slack_destination"]
    destination = None if "slack_destination" in unique else proposal.slack_destination
    body_blockers = [item for item in unique.values() if item.slot in {"subject", "input_reference", "answer_reference", "pending_request"}]
    if body_blockers and not isinstance(body, UnresolvedBody):
        body = UnresolvedBody(question=body_blockers[0].question, instruction=getattr(body, "instruction", ""))
    elif isinstance(body, ComposeBody) and not body.instruction.strip() and any(
        item.slot in {"save_intent", "slack_intent"} for item in unique.values()
    ) and not any(getattr(proposal.actions, name).intent == "requested" for name in ("save_text", "slack_notify")):
        question = next(item.question for item in unique.values() if item.slot in {"save_intent", "slack_intent"})
        body = UnresolvedBody(question=question)
    return RequestContract.model_validate({
        **proposal.model_dump(mode="python", exclude={"body", "missing_info"}),
        "request_id": request_id,
        "revision": revision,
        "body": body, "body_request": proposal.body, "missing_info": tuple(unique.values()),
        "slack_destination": destination,
    })
