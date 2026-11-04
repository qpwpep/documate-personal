from __future__ import annotations

import hashlib
import re
from uuid import uuid4

from src.core.answer_schema import text_document
from src.core.contracts import GraphState
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.request_bodies import body_to_wire
from src.core.request_contracts import (
    AcknowledgeBody, ActionContract, ActionRequest, AnswerContract, AnswerReference, BoundAnswerReference,
    BoundInputText, ComposeBody, CopyAnswerBody, CopyInputBody, MissingInformation,
    RequestContract, TransformAnswerBody, TransformInputBody, UnresolvedBody,
    WireCopyAnswerBody, WireCopyInputBody, WireRequestContract, WireTransformAnswerBody,
    WireTransformInputBody, check_answer_contract, evidence_scope_covers,
)
from src.core.slack_contract import ExplicitRecipient, OmittedRecipient, UnresolvedRecipient


def _missing(slot: str, reason: str, question: str) -> MissingInformation:
    return MissingInformation(slot=slot, reason=reason, question=question)


def current_recipient_evidence_ids(contract, *, current_turn_id: str, current_utterance: str) -> tuple[str, ...]:
    """Ground a complete recipient in a scoped quote of the current raw turn."""
    recipient = contract.slack_recipient
    if isinstance(recipient, OmittedRecipient):
        return ()
    if isinstance(recipient, ExplicitRecipient):
        if recipient.selector.kind == "email":
            left = r"(?<![A-Za-z0-9.!#$%&'*+/=?^_`{|}~@-])"
            right = r"(?![A-Za-z0-9!#$%&'*+/=?^_`{|}~@-]|\.[A-Za-z0-9._-])"
        else:
            left, right = r"(?<![A-Za-z0-9_])", r"(?![A-Za-z0-9_])"
        pattern = re.compile(left + re.escape(recipient.selector.value) + right)
    else:
        if not recipient.raw_input:
            return ()
        pattern = re.compile(re.escape(recipient.raw_input))
    allowed = {"instruction", "correction", "reference"}
    if isinstance(recipient, UnresolvedRecipient):
        allowed.add("negation")
    grounded = []
    for item in contract.evidence:
        if (item.id not in recipient.evidence_ids or item.turn_id != current_turn_id
                or item.scope not in {"current_request", "slack_recipient"}
                or item.interpretation not in allowed):
            continue
        # Check the complete utterance, so a shortened quote cannot turn the
        # prefix of an ID/email into an authorized different recipient.
        for quote in re.finditer(re.escape(item.quote), current_utterance):
            match = pattern.search(current_utterance, quote.start())
            if match is not None and match.end() <= quote.end():
                grounded.append(item.id)
                break
    return tuple(grounded)


def _bind_local_evidence(proposal: WireRequestContract, *, request_id: str, revision: int, turn_id: str) -> WireRequestContract:
    """Local model labels cannot address or overwrite a prior revision's evidence."""
    ids = {item.id: f"ev:{request_id}:{revision}:{turn_id}:{index}"
           for index, item in enumerate(proposal.evidence, 1)}
    data = proposal.model_dump(mode="python")
    data["evidence"] = tuple(item.model_copy(update={"id": ids[item.id]}) for item in proposal.evidence)
    for name in ("save_text", "slack_notify"):
        data["actions"][name]["evidence_ids"] = tuple(ids[item] for item in data["actions"][name]["evidence_ids"])
    for group in ("content", "format", "sentence_counts"):
        for item in data["answer"][group]:
            item["evidence_ids"] = tuple(ids[key] for key in item["evidence_ids"])
    if "evidence_ids" in data["body"]:
        data["body"]["evidence_ids"] = tuple(ids[key] for key in data["body"]["evidence_ids"])
    if "evidence_ids" in data["slack_recipient"]:
        data["slack_recipient"]["evidence_ids"] = tuple(ids[key] for key in data["slack_recipient"]["evidence_ids"])
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
        response = runtime.previous_response if body.source.ref == "previous" else (runtime.pending_request.response if runtime.pending_request else None)
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


def _apply_requested_copy_layout(body, body_request, answer, runtime):
    """An explicit layout edit changes the operation, never the bound source."""
    layouts = [item for item in answer.format if item.kind != "line_count" and item.mode != "preferred"]
    if not layouts:
        return body, body_request
    if isinstance(body, CopyInputBody):
        document = text_document(body.source.text)
    elif isinstance(body, CopyAnswerBody):
        response = (runtime.previous_response if body.source.ref == "previous"
                    else runtime.pending_request.response if runtime.pending_request else None)
        if response is None:
            return body, body_request
        document = response.content
    else:
        return body, body_request
    check = check_answer_contract(AnswerContract(format=answer.format), document)
    # A line-count mismatch may require deleting or rewriting content. Do not
    # infer authorization for that from a request to change only presentation.
    if any(key.startswith("line_count:") for key in (*check.missing_required, *check.forbidden_present)):
        return body, body_request
    # Even an existing layout may cover only part of the requested content.
    # Let the transformation and format validation apply the user's full scope.
    evidence_ids = tuple(dict.fromkeys(
        key for item in layouts for key in item.evidence_ids
    ))
    instruction = (
        "Preserve the selected source's wording and information. Apply only the explicitly requested "
        "layout in answer.format. Do not translate, summarize, delete source content or add explanations "
        "or examples. Preserve source whitespace inside a surrounding code block."
    )
    transform = TransformInputBody if isinstance(body, CopyInputBody) else TransformAnswerBody
    wire_transform = WireTransformInputBody if isinstance(body, CopyInputBody) else WireTransformAnswerBody
    return (transform(source=body.source, instruction=instruction, evidence_ids=evidence_ids),
            wire_transform(source=body_request.source, instruction=instruction, evidence_ids=evidence_ids))


def _merge_pending(proposal: WireRequestContract, pending, *, current_turn_id: str, current_utterance: str):
    original = pending.contract
    # The proposal's IDs were bound to a new revision before this merge.
    evidence = {item.id: item for item in (*original.evidence, *proposal.evidence)}
    current_ids = {item.id for item in proposal.evidence if item.turn_id == current_turn_id}

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

    def current_requirement_instruction(item, group):
        allowed = ({"negation", "correction", "instruction"} if item.mode == "forbidden"
                   else {"instruction", "correction", "reference"})
        return any(key in current_ids and evidence[key].interpretation in allowed
                   and evidence_scope_covers(evidence[key].scope, f"answer.{group}.{item.kind}")
                   for key in item.evidence_ids)

    def content_requirements(previous, incoming):
        result = list(previous)
        targets = {ref: index for index, ref in enumerate(original.content_requirements_by_ref)}
        replaced = set()
        for item in incoming:
            if item in result:
                continue
            if not current_requirement_instruction(item, "content"):
                raise ValueError("pending requirements need a new user instruction to change")
            if item.replaces is not None:
                if item.replaces not in targets or item.replaces in replaced:
                    raise ValueError("content correction must select one distinct offered condition in the current pending revision")
                replaced.add(item.replaces)
                result[targets[item.replaces]] = item.model_copy(update={"replaces": None})
                continue
            matching = next((index for index, old in enumerate(result)
                             if old.condition_key == item.condition_key and old.mode == item.mode), None)
            if matching is None:
                result.append(item)
            else:
                old = result[matching]
                result[matching] = old.model_copy(update={
                    "evidence_ids": tuple(dict.fromkeys((*old.evidence_ids, *item.evidence_ids))),
                })
        return tuple(result)

    def format_requirements(previous, incoming):
        result = list(previous)
        for item in incoming:
            if item in result:
                continue
            if not current_requirement_instruction(item, "format"):
                raise ValueError("pending requirements need a new user instruction to change")
            if item.kind == "line_count":
                # One required/preferred count can change without deleting
                # independent prohibitions on other counts.
                result = [old for old in result if old.kind != item.kind or not (
                    (item.mode in {"required", "preferred"} and old.mode == item.mode)
                    or (item.mode != "preferred" and old.value == item.value)
                )]
            else:
                result = [old for old in result if old.kind != item.kind]
            result.append(item)
        return tuple(result)

    sentence_counts = list(original.answer.sentence_counts)
    for item in proposal.answer.sentence_counts:
        if item in sentence_counts:
            continue
        if not current_requirement_instruction(item, "sentence_counts"):
            raise ValueError("pending sentence counts need a new user instruction to change")
        matching = next((index for index, old in enumerate(sentence_counts) if old.scope == item.scope), None)
        if matching is None:
            sentence_counts.append(item)
        else:
            # Keep scope indices stable for a prepared answer and its citations.
            sentence_counts[matching] = item

    answer = AnswerContract(content=content_requirements(original.answer.content, proposal.answer.content),
                            format=format_requirements(original.answer.format, proposal.answer.format),
                            sentence_counts=tuple(sentence_counts),
                            preferences=tuple(dict.fromkeys([*original.answer.preferences, *proposal.answer.preferences])))
    body = proposal.body
    neutral_body = isinstance(body, ComposeBody) and not body.instruction.strip() and not body.evidence_ids
    unfinished_body_copy = (proposal.relation == "supplement" and isinstance(body, WireCopyAnswerBody)
                            and body.source.ref == "pending" and not pending.body_prepared)
    missing = list(proposal.missing_info)
    missing.extend(item for item in original.missing_info
                   if item.slot in {"save_intent", "slack_intent"} and item.slot not in updated_intent_slots)
    recipient = proposal.slack_recipient
    confirmed_ids = current_recipient_evidence_ids(proposal, current_turn_id=current_turn_id,
                                                    current_utterance=current_utterance)
    if isinstance(recipient, OmittedRecipient):
        recipient = original.slack_recipient
        missing.extend(item for item in original.missing_info if item.slot == "slack_destination")
    elif recipient.model_dump(exclude={"evidence_ids"}) == original.slack_recipient.model_dump(exclude={"evidence_ids"}):
        if current_ids.intersection(recipient.evidence_ids) and not confirmed_ids:
            raise ValueError("current recipient reconfirmation must match the complete user-specified recipient")
        recipient = original.slack_recipient
        if confirmed_ids:
            recipient = recipient.model_copy(update={
                "evidence_ids": tuple(dict.fromkeys((*recipient.evidence_ids, *confirmed_ids))),
            })
    elif not confirmed_ids:
        raise ValueError("pending recipient changes must be grounded in the current user instruction")
    original_body = original.body_request or body_to_wire(original.body)
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
        recipient = OmittedRecipient()
    elif actions["slack_notify"].intent not in {"requested", "unresolved"}:
        missing = [item for item in missing if item.slot != "slack_destination"]
    retained_ids = set(getattr(body, "evidence_ids", ()))
    for item in (*actions.values(), *answer.content, *answer.format, *answer.sentence_counts):
        retained_ids.update(item.evidence_ids)
    retained_ids.update(getattr(recipient, "evidence_ids", ()))
    evidence = {key: item for key, item in evidence.items() if key in retained_ids or item.turn_id == current_turn_id}
    return WireRequestContract.model_validate({
        **proposal.model_dump(mode="python"), "actions": ActionContract(**actions), "answer": answer,
        "body": body, "evidence": tuple(evidence.values()), "missing_info": tuple(missing),
        "slack_recipient": recipient,
    })


def resolve_request_contract(proposal: WireRequestContract, state: GraphState, *, utterances: dict[str, str]) -> RequestContract:
    """Bind a provenance-validated proposal and preserve confirmed pending facts."""
    runtime = get_runtime_state(state)
    pending = runtime.pending_request
    matches_pending = pending is not None and proposal.target_request_id == pending.contract.request_id
    inherits_pending = matches_pending and proposal.relation in {"correction", "supplement", "cancel"}
    if not inherits_pending and any(item.replaces is not None for item in proposal.answer.content):
        raise ValueError("content correction requires the offered pending request")
    if (any(item.slot == "slack_destination" and item.reason != "not_provided" for item in proposal.missing_info)
            and not isinstance(proposal.slack_recipient, UnresolvedRecipient)):
        raise ValueError("recipient ambiguity must be represented by the typed unresolved intent")
    original = pending.contract if inherits_pending else None
    request_id = original.request_id if original is not None else str(uuid4())
    revision = original.revision + 1 if original is not None else 1
    proposal = _bind_local_evidence(proposal, request_id=request_id, revision=revision, turn_id=runtime.current_turn_id)
    if inherits_pending:
        proposal = _merge_pending(proposal, pending, current_turn_id=runtime.current_turn_id,
                                  current_utterance=utterances[runtime.current_turn_id])
    missing = list(proposal.missing_info)
    if proposal.target_request_id is not None and not matches_pending:
        missing.append(_missing("pending_request", "unknown_id", "지정한 보류 요청이 없습니다. 처리할 요청을 다시 지정해 주세요."))
    elif proposal.relation in {"supplement", "cancel"} and proposal.target_request_id is None:
        missing.append(_missing("pending_request", "not_provided", "보충하거나 취소할 요청을 특정하지 못했습니다. 해당 요청을 알려 주세요."))

    recipient = proposal.slack_recipient
    if isinstance(recipient, ExplicitRecipient):
        retained = (original is not None and recipient.model_dump(exclude={"evidence_ids"})
                    == original.slack_recipient.model_dump(exclude={"evidence_ids"}))
        if not retained and not current_recipient_evidence_ids(proposal, current_turn_id=runtime.current_turn_id,
                                                               current_utterance=runtime.user_input):
            recipient = UnresolvedRecipient(raw_input=recipient.selector.value, reason="unverified",
                                            evidence_ids=recipient.evidence_ids)
    if isinstance(recipient, UnresolvedRecipient):
        missing.append(_missing("slack_destination", "unknown_id" if recipient.reason == "unverified" else "unclear",
                                "지정한 Slack 수신자를 확정하지 못했습니다. 정확한 채널 ID, 사용자 ID 또는 이메일을 알려 주세요."))
    body, body_missing = _bind_body(proposal.body, runtime, utterances, original, declared_missing=proposal.missing_info)
    missing.extend(body_missing)
    current_evidence = {item.id for item in proposal.evidence if item.turn_id == runtime.current_turn_id}
    for name, slot in (("save_text", "save_intent"), ("slack_notify", "slack_intent")):
        action = getattr(proposal.actions, name)
        historical_only = action.intent == "requested" and not inherits_pending and not current_evidence.intersection(action.evidence_ids)
        if action.intent == "unresolved" or historical_only:
            missing.append(_missing(slot, "unclear", "저장 여부를 알려 주세요." if name == "save_text" else "Slack으로 전송할지 알려 주세요."))
    configured = runtime.session_metadata.slack_recipient
    if proposal.actions.slack_notify.intent == "requested" and not (
        not isinstance(recipient, OmittedRecipient) or configured is not None
    ):
        missing.append(_missing("slack_destination", "not_provided", "Slack으로 보낼 channel_id, user_id 또는 email을 알려 주세요."))
    unique = {}
    missing_priority = {"not_provided": 0, "unclear": 1, "unknown_id": 2, "quote_not_found": 2, "ambiguous_scope": 2}
    for item in missing:
        existing = unique.get(item.slot)
        if existing is None or missing_priority[item.reason] > missing_priority[existing.reason]:
            unique[item.slot] = item
    if isinstance(recipient, ExplicitRecipient) and "slack_destination" in unique and unique["slack_destination"].reason == "not_provided":
        del unique["slack_destination"]
    gap = unique.get("slack_destination")
    if gap is not None and gap.reason != "not_provided" and not isinstance(recipient, UnresolvedRecipient):
        raise ValueError("recipient gap contradicts the preserved typed recipient intent")
    body_blockers = [item for item in unique.values() if item.slot in {"subject", "input_reference", "answer_reference", "pending_request"}]
    if body_blockers and not isinstance(body, UnresolvedBody):
        body = UnresolvedBody(question=body_blockers[0].question, instruction=getattr(body, "instruction", ""))
    elif isinstance(body, ComposeBody) and not body.instruction.strip() and any(
        item.slot in {"save_intent", "slack_intent"} for item in unique.values()
    ) and not any(getattr(proposal.actions, name).intent == "requested" for name in ("save_text", "slack_notify")):
        question = next(item.question for item in unique.values() if item.slot in {"save_intent", "slack_intent"})
        body = UnresolvedBody(question=question)
    body, body_request = _apply_requested_copy_layout(body, proposal.body, proposal.answer, runtime)
    return RequestContract.model_validate({
        **proposal.model_dump(mode="python", exclude={"body", "missing_info"}),
        "request_id": request_id,
        "revision": revision,
        "body": body, "body_request": body_request, "missing_info": tuple(unique.values()),
        "slack_recipient": recipient,
    })
