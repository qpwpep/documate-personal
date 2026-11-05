"""Reuse judgments only for identical model inputs and evaluator instructions."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from src.core.contracts.graph_state import DecisionCacheEntry
from src.core.decisions import DecisionResult, decision_fingerprint
from src.runtime.nodes.validation.decision_input import encode_decision_input
from src.runtime.nodes.validation.models import ValidationAssessment, ValidationSnapshot
from src.runtime.agent_runtime.validation_diagnostics import record_decisions


class DecisionEvaluator(Protocol):
    model_name: str
    max_questions_per_call: int

    def evaluate(self, *, input: str, questions: list[dict[str, str]],
                 stage: str, attempt: int = 1, reserve_calls: int = 0) -> DecisionResult: ...


@dataclass(frozen=True)
class CachedDecisionJob:
    name: str
    entry_name: str
    entry: dict[str, object]
    instructions: str
    fingerprint: str


def decision_job_fingerprint(*, collection: str, entry: dict[str, object], instructions: str,
                             model: str, encode_input: Callable[[dict[str, object]], str] = encode_decision_input) -> str:
    return decision_fingerprint({
        "model": model, "input": encode_input({collection: [{"name": "<entry>", **entry}]}),
        "instructions": instructions,
    })


def uncached_job_count(jobs: list[CachedDecisionJob], snapshot: ValidationSnapshot,
                       assessment: ValidationAssessment) -> int:
    known = {item.fingerprint for item in [*snapshot.decision_cache, *assessment.decision_cache]}
    return len({job.fingerprint for job in jobs} - known)


def evaluate_cached_jobs(
    *, snapshot: ValidationSnapshot, assessment: ValidationAssessment, decisions_client: DecisionEvaluator,
    jobs: list[CachedDecisionJob], collection: str, stage: str, attempt: int, reserve_calls: int = 0,
    encode_input: Callable[[dict[str, object]], str] = encode_decision_input,
) -> DecisionResult:
    cache = {item.fingerprint: item for item in [*snapshot.decision_cache, *assessment.decision_cache]}
    pending: dict[str, CachedDecisionJob] = {}
    aliases: dict[str, list[str]] = {}
    cache_hits = 0
    for job in jobs:
        aliases.setdefault(job.fingerprint, []).append(job.name)
        if job.fingerprint in cache or job.fingerprint in pending:
            cache_hits += 1
        else:
            pending[job.fingerprint] = job
    cached_names = {job.name for job in jobs if job.fingerprint in cache}
    record_decisions(DecisionResult(
        probabilities={job.name: cache[job.fingerprint].probability for job in jobs
                       if job.fingerprint in cache and cache[job.fingerprint].probability is not None},
        refused_names={job.name for job in jobs
                       if job.fingerprint in cache and cache[job.fingerprint].probability is None},
    ), cached_names=cached_names)
    if pending:
        entries = []
        questions = []
        entries_by_content = {}
        for job in pending.values():
            identity = decision_fingerprint(job.entry)
            if identity not in entries_by_content:
                entries_by_content[identity] = job.entry_name
                entries.append({"name": job.entry_name, **job.entry})
            entry_name = entries_by_content[identity]
            questions.append({"type": "predicate", "name": job.name,
                              "instructions": job.instructions.replace("<entry>", entry_name)})
        result = decisions_client.evaluate(input=encode_input({collection: entries}), questions=questions,
                                           stage=stage, attempt=attempt, reserve_calls=reserve_calls)
        for fingerprint, job in pending.items():
            probability = None if job.name in result.refused_names else result.probabilities[job.name]
            cache[fingerprint] = DecisionCacheEntry(fingerprint=fingerprint, probability=probability)
    probabilities = {}
    refused = set()
    for fingerprint, names in aliases.items():
        for name in names:
            probability = cache[fingerprint].probability
            if probability is None:
                refused.add(name)
            else:
                probabilities[name] = probability
    assessment.decision_cache = list(cache.values())
    assessment.cache_observations.append({
        "phase": collection, "cache_hits": cache_hits, "cache_misses": len(pending),
        "avoided_questions": len(jobs) - len(pending), "requested_questions": len(jobs),
        "evaluate_calls": int(bool(pending)),
    })
    result = DecisionResult(probabilities=probabilities, refused_names=refused)
    record_decisions(result, cached_names=cached_names)
    return result
