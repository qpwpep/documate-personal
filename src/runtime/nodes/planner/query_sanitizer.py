from __future__ import annotations

import json
import re

from src.core.planner_schema import InitialPlannerOutput


def _contains_anchor(text: str, aspect: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(aspect)}(?![A-Za-z0-9_])", text, re.I))


def ground_planner_requirement_aspects(
    planner_output: InitialPlannerOutput,
    *,
    user_utterances: dict[str, str],
    current_turn_id: str,
    previous_answer: str,
) -> InitialPlannerOutput:
    """Bind anchors to user text or a current reference to a selected answer item.

    Reference meaning belongs to the initial interpretation, like other request
    evidence. The server verifies its literal provenance, not arbitrary assistant
    words or a second natural-language interpretation via keyword heuristics.
    """
    evidence = {item.id: item for item in planner_output.request_contract.evidence}
    referenced = set()
    nonselecting = set()
    for reference in planner_output.aspect_references:
        item = evidence.get(reference.evidence_id)
        current_evidence = (item is not None and item.turn_id == current_turn_id
                            and item.scope == f"retrieval.{reference.requirement_id}.aspects"
                            and item.quote in user_utterances.get(current_turn_id, ""))
        identity = (reference.requirement_id, reference.aspect)
        if current_evidence and item.interpretation in {"negation", "mention", "quotation"}:
            nonselecting.add(identity)
        if (current_evidence and item.interpretation == "reference"
                and reference.answer_quote in previous_answer
                and _contains_anchor(reference.answer_quote, reference.aspect)):
            referenced.add(identity)
    grounded_tasks = []
    missing_references = []
    for task in planner_output.tasks:
        aspects = [aspect for aspect in task.requirement.aspects
                   if any(_contains_anchor(utterance, aspect) for utterance in user_utterances.values())
                   or (task.requirement_id, aspect) in referenced]
        missing_references.extend({"requirement_id": task.requirement_id, "aspect": aspect}
                                  for aspect in task.requirement.aspects if aspect not in aspects
                                  and (task.requirement_id, aspect) not in nonselecting
                                  and _contains_anchor(previous_answer, aspect))
        grounded_tasks.append(task.model_copy(update={
            "requirement": task.requirement.model_copy(update={"aspects": aspects}),
        }))
    if missing_references:
        # Dropping a resolvable selected subject would weaken the search contract.
        # Let the existing structured boundary request its one bounded repair.
        raise ValueError(
            "Missing valid aspect_references for anchors found only in previous_answer_for_aspects: "
            + json.dumps(missing_references, ensure_ascii=False)
            + ". Recheck the original current user request. If it selects an answer item, retain its literal aspect "
            "and add aspect_references with an exact answer_quote and local evidence_id; the evidence must use "
            "the current turn_id, exact selecting user quote, interpretation=reference, and "
            "scope=retrieval.<requirement_id>.aspects. current_request instruction evidence alone is insufficient. "
            "If the user does not select that subject, remove the unsupported aspect. Do not drop a selected "
            "subject merely to satisfy validation, or promote other answer words and guessed values."
        )
    return planner_output.model_copy(update={"tasks": grounded_tasks})
