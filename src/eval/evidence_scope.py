"""Verify final citation origins without changing current-turn tool observations."""

from __future__ import annotations

from src.core.answer_schema import AnswerResponse
from src.core.contracts.provenance import AnswerProvenance
from src.core.evidence import EvidenceRef, SearchHit

from .metric_rules import _source_selection_contains
from .result_models import EvidenceAssessment, ScenarioTurnResult


_ROUTE_TOOLS = {"docs": "tavily_search", "upload": "upload_search"}


def _assess_response(
    *, response: AnswerResponse | None, provenance: AnswerProvenance | None,
    observed_hits: list[SearchHit], tool_calls: list[str],
    preceding: list[tuple[ScenarioTurnResult, EvidenceAssessment]],
) -> EvidenceAssessment:
    if provenance is None:
        return EvidenceAssessment(status="unavailable", errors=["answer_provenance is missing"])
    if response is None:
        return EvidenceAssessment(status="invalid", errors=["answer_provenance has no valid response"])

    errors: list[str] = []
    if provenance.response_hash != response.content_hash:
        errors.append("answer_provenance.response_hash does not match the final response")
    if (provenance.body_kind in {"copy_answer", "transform_answer"}
            and provenance.source is None and (response.citations or provenance.evidence_packet)):
        errors.append("answer_provenance.source is missing for a cited answer copy or transformation")
    inherited: list[EvidenceRef] = []
    if provenance.source is not None:
        source = provenance.source
        # A copied answer can keep its content hash across many turns. Resolve
        # only already assessed records, never a hash map containing this turn.
        parent = next((
            (turn, assessment) for turn, assessment in reversed(preceding)
            if turn.response is not None
            and turn.response.content_hash == source.response_hash
            and [citation.evidence.id for citation in turn.response.citations] == source.citation_ids
        ), None)
        if parent is None:
            errors.append("answer_provenance.source is not an earlier answer in this session")
        elif parent[1].status != "complete":
            errors.append("answer_provenance.source has no verified evidence origin")
        else:
            inherited = parent[1].verified_evidence

    packet = {item.id: item for item in provenance.evidence_packet}
    if len(packet) != len(provenance.evidence_packet):
        errors.append("answer_provenance.evidence_packet contains duplicate references")
    authorized: set[str] = set()
    for item in packet.values():
        current = _ROUTE_TOOLS[item.route] in tool_calls and any(
            _source_selection_contains(hit.evidence, item) for hit in observed_hits
        )
        inherited_origin = any(_source_selection_contains(parent, item) for parent in inherited)
        if current or inherited_origin:
            authorized.add(item.id)
        else:
            errors.append(f"answer_provenance.evidence_packet reference has no allowed origin: {item.id}")

    verified: list[EvidenceRef] = []
    for citation in response.citations:
        evidence = citation.evidence
        # Packet selections are already final. A citation cannot manufacture a
        # different selection merely because a broader raw search hit exists.
        if packet.get(evidence.id) != evidence:
            errors.append(f"final citation is not in answer_provenance.evidence_packet: {evidence.id}")
        elif evidence.id in authorized:
            verified.append(evidence)

    return EvidenceAssessment(status="invalid" if errors else "complete", errors=errors,
                              verified_evidence=verified)


def assess_evidence_scope(
    *, response: AnswerResponse | None, provenance: AnswerProvenance | None,
    observed_hits: list[SearchHit], tool_calls: list[str], session_id: str,
    prior_turns: list[ScenarioTurnResult],
) -> EvidenceAssessment:
    """Validate chronological source revisions and retain only their actual citations."""
    preceding: list[tuple[ScenarioTurnResult, EvidenceAssessment]] = []
    for turn in prior_turns:
        if turn.request_payload.get("session_id") != session_id:
            continue
        assessment = _assess_response(
            response=turn.response, provenance=turn.answer_provenance,
            observed_hits=turn.observed_hits, tool_calls=turn.tool_calls, preceding=preceding,
        )
        if turn.runtime_errors or turn.response_errors:
            assessment = EvidenceAssessment(status="invalid", errors=["source turn failed validation"])
        preceding.append((turn, assessment))
    return _assess_response(response=response, provenance=provenance, observed_hits=observed_hits,
                            tool_calls=tool_calls, preceding=preceding)
