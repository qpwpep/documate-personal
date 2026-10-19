"""Provider-neutral results of named predicate decisions."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionResult:
    """Validated answers, with individual refusals retained by question name.

    Every requested name occurs in exactly one collection. Refusals carry no
    probability or provider text; the caller decides their recovery policy.
    """

    probabilities: dict[str, float]
    refused_names: set[str]
