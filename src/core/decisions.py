"""Provider-neutral results of named predicate decisions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


def decision_fingerprint(value: Any) -> str:
    """Bind a decision to canonical content, independent of dictionary order."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def evidence_plan_fingerprint(plan: Any) -> str:
    """Prepared packets belong to the exact plan used to construct them."""
    return decision_fingerprint(plan.model_dump(mode="json"))


def prepared_packet_fingerprint(evidence: list[Any], requirement_map: dict[str, list[str]]) -> str:
    """Bind approved generation inputs to all source revisions and exact ranges."""
    return decision_fingerprint({"evidence": [item.model_dump(mode="json") for item in evidence],
                                 "requirement_map": requirement_map})


@dataclass(frozen=True)
class DecisionResult:
    """Validated answers, with individual refusals retained by question name.

    Every requested name occurs in exactly one collection. Refusals carry no
    probability or provider text; the caller decides their recovery policy.
    """

    probabilities: dict[str, float]
    refused_names: set[str]
