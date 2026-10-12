from __future__ import annotations

import re

from src.core.planner_schema import PlannerOutput


def ground_planner_requirement_aspects(
    planner_output: PlannerOutput,
    *,
    user_utterances: list[str],
) -> PlannerOutput:
    """Keep mandatory literal anchors only when present in the original user ledger."""
    grounded_tasks = []
    for task in planner_output.tasks:
        grounded_aspects = []
        for aspect in task.requirement.aspects:
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(aspect)}(?![A-Za-z0-9_])"
            if any(re.search(pattern, utterance, re.I) for utterance in user_utterances):
                grounded_aspects.append(aspect)
        grounded_tasks.append(task.model_copy(update={
            "requirement": task.requirement.model_copy(update={"aspects": grounded_aspects}),
        }))
    return planner_output.model_copy(update={"tasks": grounded_tasks})
