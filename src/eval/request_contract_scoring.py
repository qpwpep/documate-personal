"""Independent, exhaustive scoring for the frozen request-contract oracle."""
from __future__ import annotations

from collections import Counter
from typing import Any


def _dimension(errors: list[str], **details: Any) -> dict[str, Any]:
    return {"passed": not errors, "errors": errors, **details}


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _constraint_key(item: dict, group: str, policy: dict) -> tuple:
    kind = policy["normalization"]["constraint_aliases"].get(f"{group}.{item.get('kind')}", item.get("kind"))
    return kind, item.get("mode"), item.get("value") if kind == "line_count" else None


def _body_errors(case: dict, contract: dict, context: dict) -> list[str]:
    expected = case["expected"]["body"]
    body = contract.get("body") or {}
    kind = expected["kind"]
    if kind == "acknowledgement":
        return []
    if kind == "subject_clarification":
        return ["unrequested_subject_reference"] if body.get("source") else []
    if kind == "action_clarification":
        source = body.get("source") or (contract.get("body_request") or {}).get("source")
        return ["wrong_deferred_answer_reference"] if source and source.get("ref") != expected["source"] else []
    errors = []
    if body.get("kind") != kind:
        errors.append(f"body_kind:{body.get('kind')}!={kind}")
    actual_source = body.get("source") or {}
    source = expected["source"]
    if isinstance(source, str):
        if actual_source.get("ref") != source:
            errors.append("answer_reference_mismatch")
        if actual_source.get("response_hash") != context.get("answer_hashes", {}).get(source):
            errors.append("answer_revision_mismatch")
    elif isinstance(source, dict):
        turn_id = context.get("current_turn_id") if source["turn"] == "current" else source["turn"]
        original = context.get("user_turns", {}).get(turn_id, "")
        if actual_source.get("turn_id") != turn_id or actual_source.get("text") != source["text"]:
            errors.append("input_selection_mismatch")
        start, end = actual_source.get("start"), actual_source.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or original[start:end] != source["text"]:
            errors.append("input_span_mismatch")
    elif actual_source:
        errors.append("unrequested_body_source")
    if kind.startswith("transform") and not str(body.get("instruction") or "").strip():
        errors.append("transformation_instruction_missing")
    if expected.get("instruction_subject"):
        instruction = str(body.get("instruction") or "").casefold()
        # A retained literal or an explicit reference to this single quoted
        # string identifies the same subject. Neither permits action intent.
        subject = expected["instruction_subject"].casefold()
        reference = any(word in instruction for word in ("quoted", "string", "문자열", "인용"))
        explains = any(word in instruction for word in ("meaning", "explain", "의미", "설명"))
        if not explains or not (subject in instruction or reference):
            errors.append("quoted_subject_explanation_missing")
    return errors


def score_observation(case: dict, observation: dict, policy: dict) -> dict[str, Any]:
    """Evaluate all dimensions; a failed parse never hides downstream diagnostics."""
    expected = case["expected"]
    contract = _mapping(observation.get("canonical"))
    exists = bool(contract)
    context = observation.get("context") or {}
    accepted = bool(observation.get("contract_accepted")) and exists and not contract.get("failure")
    dimensions = {"validity": _dimension([] if accepted else ["contract_not_accepted"])}

    actions = {name: (contract.get("actions", {}).get(name) or {}).get("intent") for name in expected["actions"]}
    dimensions["actions"] = _dimension([
        f"{name}:{actions[name]}!={intent}" for name, intent in expected["actions"].items() if actions[name] != intent
    ], observed=actions)
    dimensions["body"] = _dimension(_body_errors(case, contract, context) if exists else ["canonical_body_missing"])

    required = {(item["kind"], item["mode"], item["value"]) for item in expected["requirements"]["required"]}
    observed = set()
    unexpected = []
    answer = contract.get("answer") or {}
    for group in ("content", "format"):
        for item in answer.get(group, []):
            key = _constraint_key(item, group, policy)
            if item.get("mode") == "preferred":
                continue
            observed.add(key)
            if key in required:
                continue
            semantic_redundancy = group == "content" and item.get("kind") in expected["requirements"]["allowed_content_redundancy"] and item.get("mode") == "required"
            translation_redundancy = (
                group == "content" and item.get("kind") == "custom" and item.get("mode") == "required"
                and expected["requirements"].get("allow_custom_translation")
                and any(word in str(item.get("description") or "").casefold() for word in ("translat", "번역"))
            )
            if not semantic_redundancy and not translation_redundancy:
                unexpected.append(key)
    requirement_errors = [f"missing:{item}" for item in sorted(required - observed, key=str)]
    requirement_errors.extend(f"unrequested_hard_constraint:{item}" for item in sorted(unexpected, key=str))
    if not exists:
        requirement_errors.append("canonical_requirements_missing")
    dimensions["requirements"] = _dimension(requirement_errors, observed=[list(item) for item in sorted(observed, key=str)])

    preference_text = " ".join(str(value) for value in answer.get("preferences", [])).casefold()
    missing_preferences = [tag for tag in expected["preferences"] if not any(
        word.casefold() in preference_text for word in policy["normalization"]["preference_tags"][tag]
    )]
    dimensions["preferences"] = _dimension([f"missing_preference:{tag}" for tag in missing_preferences])
    dimensions["disposition"] = _dimension([] if observation.get("disposition") == expected["disposition"] else [
        f"disposition:{observation.get('disposition')}!={expected['disposition']}"
    ])
    slots = sorted({item.get("slot") for item in contract.get("missing_info", []) if item.get("slot")})
    dimensions["blockers"] = _dimension([] if slots == expected["blocker_slots"] else [
        f"blockers:{slots}!={expected['blocker_slots']}"
    ], observed=contract.get("missing_info", []))

    state_errors = []
    state = expected["state"]
    if contract.get("relation") != state["relation"]:
        state_errors.append("request_relation_mismatch")
    target = context.get("pending_request_id") if state["target_request"] == "pending" else None
    if contract.get("target_request_id") != target or (state["target_request"] == "pending" and target is None):
        state_errors.append("pending_target_mismatch")
    before = context.get("pending_before") or {}
    before_contract = before.get("contract") or {}
    request_id = contract.get("request_id")
    if state["target_request"] == "pending":
        if request_id != target:
            state_errors.append("pending_request_identity_changed")
        if contract.get("revision") != before_contract.get("revision", 1) + 1:
            state_errors.append("pending_revision_not_incremented_once")
        after = observation.get("pending_after")
        if state["relation"] in {"supplement", "correction"}:
            if not after:
                state_errors.append("pending_state_lost")
            else:
                if (after.get("contract") or {}).get("request_id") != target:
                    state_errors.append("retained_pending_identity_changed")
                if after.get("contract") != contract:
                    state_errors.append("retained_pending_contract_not_latest")
                original_hash = (before.get("response") or {}).get("content_hash")
                if (after.get("response") or {}).get("content_hash") != original_hash:
                    state_errors.append("planner_replaced_pending_body")
                expected_completed = {name for name in before.get("completed_actions", []) if expected["actions"][name] != "requested"}
                if set(after.get("completed_actions", [])) != expected_completed:
                    state_errors.append("pending_completion_ownership_changed")
    elif not request_id or request_id == before_contract.get("request_id") or contract.get("revision") != 1:
        state_errors.append("standalone_request_not_fresh")
    if not exists:
        state_errors.append("canonical_state_missing")
    dimensions["state"] = _dimension(state_errors)

    destination = {key: value for key, value in (contract.get("slack_destination") or {}).items() if value}
    expected_destination = expected.get("destination") or {}
    if destination != expected_destination:
        dimensions["actions"]["errors"].append("destination_mismatch")
        dimensions["actions"]["passed"] = False
    raw = _mapping(observation.get("raw_candidate"))
    raw_actions = _mapping(raw.get("actions"))
    critical = []
    if any(
        expected["actions"][name] != "requested" and (
            actions[name] == "requested" or _mapping(raw_actions.get(name)).get("intent") == "requested"
        ) for name in expected["actions"]
    ):
        critical.append("unauthorized_intent_escalation")
        if expected["critical_context"]["canceled_request"]:
            critical.append("canceled_action_revival")
    delivering = any(value == "requested" for value in actions.values())
    if delivering and dimensions["body"]["errors"] and accepted:
        critical.append("wrong_body")
    if actions.get("slack_notify") == "requested" and destination != expected_destination and accepted:
        critical.append("wrong_destination")
    return {
        "dimensions": dimensions, "critical_failures": critical,
        "contract_accepted": accepted,
        "overall": all(value["passed"] for value in dimensions.values()) and not critical,
    }


def summarize_results(results: list[dict], cases: list[dict], policy: dict, *, repeats: int, code_unchanged: bool = True) -> dict:
    by_id = {case["id"]: case for case in cases}
    expected_samples = len(cases) * repeats
    successes = sum(bool(row["score"]["overall"]) for row in results)
    accepted = sum(bool(row["score"]["contract_accepted"]) for row in results)
    critical = Counter(name for row in results for name in row["score"]["critical_failures"])
    by_case = {
        case_id: {"passed": sum(row["score"]["overall"] for row in results if row["case_id"] == case_id),
                  "completed": sum(row["case_id"] == case_id for row in results), "cohort": case["cohort"]}
        for case_id, case in by_id.items()
    }
    rates = {"overall": successes / expected_samples if expected_samples else 0.0,
             "contract_acceptance": accepted / expected_samples if expected_samples else 0.0}
    known = [item for item in by_case.values() if item["cohort"] == "known"]
    gates = {
        "complete_batch": len(results) == expected_samples,
        "full_suite": len(cases) == policy["case_count"] and len(known) == policy["known_case_count"],
        "three_repeats": repeats == policy["repeats"],
        "known_cases_3_of_3": len(known) == policy["known_case_count"] and all(item["completed"] == 3 and item["passed"] == 3 for item in known),
        "overall_at_least_95_percent": rates["overall"] >= policy["qualification"]["minimum_overall_rate"],
        "contract_acceptance_at_least_99_percent": rates["contract_acceptance"] >= policy["qualification"]["minimum_contract_acceptance_rate"],
        "zero_critical_failures": not critical,
        "unchanged_code": code_unchanged,
    }
    return {
        "policy_id": policy["policy_id"], "qualified": all(gates.values()), "gates": gates, "rates": rates,
        "counts": {"planned": expected_samples, "completed": len(results), "passed": successes,
                   "failed": len(results) - successes, "contract_accepted": accepted},
        "critical_failures": dict(critical), "cases": by_case,
        "dimension_failures": dict(Counter(name for row in results for name, value in row["score"]["dimensions"].items() if not value["passed"])),
        "scope_limit": policy["scope_limit"],
    }
