"""Citation authorization follows the selected answer without changing tool observations."""

from src.core.answer_schema import AnswerDocument, finalize_answer
from src.core.evidence import build_evidence
from src.eval.config_models import BenchmarkCase
from src.eval.metric_rules import compute_rule_scores, score_answer_quality, tool_confusion_counts
from src.eval.result_models import ScenarioTurnResult
from tests.eval.response_fixtures import source_hit


def answer(*evidence):
    document = AnswerDocument.model_validate({"blocks": [
        {"type": "paragraph", "content": [{"text": item.excerpt, "basis": "source", "refs": [item.id]}]}
        for item in evidence
    ]})
    return finalize_answer(document, list(evidence), retrieval_required=True)


def provenance(response, packet, *, source=None, body_kind="compose"):
    from src.core.contracts.provenance import AnswerProvenance
    return AnswerProvenance.model_validate({
        "body_kind": body_kind, "response_hash": response.content_hash,
        "source": ({"ref": "previous", "response_hash": source.content_hash,
                    "citation_ids": [item.evidence.id for item in source.citations]} if source else None),
        "evidence_packet": [item.model_dump(mode="json") for item in packet],
    })


def turn(response, *, hits=(), source=None, packet=None, session="scenario", body_kind="compose"):
    refs = [item.evidence for item in response.citations] if packet is None else packet
    return ScenarioTurnResult(
        query="prepare", request_payload={"session_id": session}, response=response,
        observed_hits=list(hits), tool_calls=["upload_search"] if hits else [],
        answer_provenance=provenance(response, refs, source=source, body_kind=body_kind),
    )


def assess(response, *, prior=(), hits=(), source=None, packet=None, body_kind="compose", declared=None):
    from src.eval.evidence_scope import assess_evidence_scope
    refs = [item.evidence for item in response.citations] if packet is None else packet
    return assess_evidence_scope(
        response=response, provenance=declared or provenance(response, refs, source=source, body_kind=body_kind),
        observed_hits=list(hits), tool_calls=["upload_search"] if hits else ["save_text"],
        session_id="scenario", prior_turns=list(prior),
    )


def test_copied_citations_pass_without_inheriting_a_search_tool_call():
    """Copying a proven answer preserves citation credit while the final tool contract stays independent."""
    hit = source_hit(official=False)
    response = answer(hit.evidence)
    scope = assess(response, source=response, body_kind="copy_answer", prior=[turn(response, hits=[hit])])
    case = BenchmarkCase(case_id="copy", category="tool_action", query="save", expected_tools=["save_text"],
                         forbidden_tools=["upload_search"], require_local_citation=True)
    scores = compute_rule_scores(case=case, response=response, called_tools=["save_text"], observed_hits=[],
                                 runtime_errors=[], response_errors=[], judge_errors=[], evidence_scope=scope)
    assert scope.status == "complete"
    assert scope.verified_evidence == [hit.evidence]
    assert scores["citation_traceability"] == scores["reference_coverage"] == scores["tool_choice"] == 1
    case.expected_tools = ["upload_search"]
    case.forbidden_tools = []
    scores = compute_rule_scores(case=case, response=response, called_tools=["save_text"], observed_hits=[],
                                 runtime_errors=[], response_errors=[], judge_errors=[], evidence_scope=scope)
    assert scores["tool_choice"] == 0


def test_unselected_older_answer_cannot_authorize_a_citation():
    """A selected source answer does not grant access to another prior answer's sources."""
    one, two = source_hit(official=False, text="first"), source_hit(official=False, text="second")
    first, second = answer(one.evidence), answer(two.evidence)
    scope = assess(first, source=second, body_kind="transform_answer",
                   prior=[turn(first, hits=[one]), turn(second, hits=[two])])
    assert scope.status == "invalid"
    assert scope.verified_evidence == []


def test_unused_parent_search_hit_cannot_be_inherited():
    """Only the parent's actual citations carry forward, even if it retrieved additional hits."""
    one, two = source_hit(official=False, text="first"), source_hit(official=False, text="unused")
    parent, response = answer(one.evidence), answer(two.evidence)
    scope = assess(response, source=parent, body_kind="transform_answer",
                   prior=[turn(parent, hits=[one, two], packet=[one.evidence, two.evidence])])
    assert scope.status == "invalid"
    assert scope.verified_evidence == []


def test_transform_chain_cannot_restore_a_dropped_citation():
    """A citation omitted by an intermediate transform cannot reappear through its lineage."""
    one, two = source_hit(official=False, text="kept"), source_hit(official=False, text="dropped")
    first, second = answer(one.evidence, two.evidence), answer(one.evidence)
    scope = assess(answer(two.evidence), source=second, body_kind="copy_answer", prior=[
        turn(first, hits=[one, two]), turn(second, source=first, body_kind="transform_answer"),
    ])
    assert scope.status == "invalid"
    assert scope.verified_evidence == []


def test_repeated_copy_hashes_keep_the_valid_prior_proof():
    """Successive identical copies are evaluated in chronological order without creating a cycle."""
    hit = source_hit(official=False)
    response = answer(hit.evidence)
    scope = assess(response, source=response, body_kind="copy_answer", prior=[
        turn(response, hits=[hit]), turn(response, source=response, body_kind="copy_answer"),
    ])
    assert scope.status == "complete"
    assert scope.verified_evidence == [hit.evidence]


def test_current_raw_hit_cannot_expand_the_final_packet():
    """A final citation must use the actual selected packet, not merely a wider retrieved hit."""
    hit = source_hit(official=False, text="0123456789")
    narrow = build_evidence(snapshot=hit.evidence.snapshot, element=hit.evidence.element, start=0, end=4)
    scope = assess(answer(hit.evidence), hits=[hit], packet=[narrow])
    assert scope.status == "invalid"
    assert scope.verified_evidence == []


def test_narrow_current_packet_remains_traceable_to_a_larger_hit():
    """Budgeted packet selections are valid when retained exactly and contained by an observed hit."""
    hit = source_hit(official=False, text="0123456789")
    narrow = build_evidence(snapshot=hit.evidence.snapshot, element=hit.evidence.element, start=0, end=4)
    scope = assess(answer(narrow), hits=[hit])
    assert scope.status == "complete"
    assert scope.verified_evidence == [narrow]


def test_another_session_cannot_supply_a_source_proof():
    """Matching answer bytes in a different case do not authorize a scenario's citations."""
    hit = source_hit(official=False)
    response = answer(hit.evidence)
    scope = assess(response, source=response, body_kind="copy_answer",
                   prior=[turn(response, hits=[hit], session="different")])
    assert scope.status == "invalid"
    assert scope.verified_evidence == []


def test_a_parent_without_observed_origins_cannot_approve_its_copy():
    """Copying an already untraceable citation cannot turn it into observed evidence."""
    response = answer(source_hit(official=False).evidence)
    scope = assess(response, source=response, body_kind="copy_answer", prior=[turn(response)])
    assert scope.status == "invalid"
    assert scope.verified_evidence == []


def test_missing_provenance_is_unavailable_instead_of_a_proven_empty_scope():
    """Legacy absence is preserved as unknown even when a final response is structurally valid."""
    from src.eval.evidence_scope import assess_evidence_scope
    response = answer(source_hit(official=False).evidence)
    scope = assess_evidence_scope(response=response, provenance=None, observed_hits=[], tool_calls=[],
                                 session_id="scenario", prior_turns=[])
    assert scope.status == "unavailable"
    assert scope.errors
    assert scope.verified_evidence == []


def test_inherited_upload_and_current_official_evidence_keep_separate_tool_origins():
    """A transform may combine proven upload citations with newly retrieved official sources."""
    from src.eval.evidence_scope import assess_evidence_scope
    upload, docs = source_hit(official=False, text="upload"), source_hit(official=True, text="official")
    parent, response = answer(upload.evidence), answer(upload.evidence, docs.evidence)
    scope = assess_evidence_scope(
        response=response, provenance=provenance(response, [upload.evidence, docs.evidence],
                                                   source=parent, body_kind="transform_answer"),
        observed_hits=[docs], tool_calls=["tavily_search"], session_id="scenario",
        prior_turns=[turn(parent, hits=[upload])],
    )
    case = BenchmarkCase(case_id="mixed", category="hybrid", query="compare", expected_tools=["tavily_search"],
                         forbidden_tools=["upload_search"], require_local_citation=True, require_official_citation=True)
    scores = compute_rule_scores(case=case, response=response, called_tools=["tavily_search"], observed_hits=[docs],
                                 runtime_errors=[], response_errors=[], judge_errors=[], evidence_scope=scope)
    assert scope.status == "complete"
    assert scope.verified_evidence == [upload.evidence, docs.evidence]
    assert scores["citation_traceability"] == scores["reference_coverage"] == scores["tool_choice"] == 1


def test_reusing_a_long_excerpt_does_not_change_copy_penalties_or_tool_confusion():
    """Citation inheritance leaves answer-quality inputs and final-turn tool counts unchanged."""
    hit = source_hit(official=False, text="An intentionally long original excerpt. " * 5)
    response = answer(hit.evidence)
    scope = assess(response, source=response, body_kind="copy_answer", prior=[turn(response, hits=[hit])])
    case = BenchmarkCase(case_id="copy-long", category="tool_action", query="save", expected_tools=["save_text"],
                         forbidden_tools=["upload_search"], require_local_citation=True)
    scores = compute_rule_scores(case=case, response=response, called_tools=["save_text"], observed_hits=[],
                                 runtime_errors=[], response_errors=[], judge_errors=[], evidence_scope=scope)
    assert scores["answer_quality"] == score_answer_quality(case, hit.evidence.excerpt, []) == 1
    assert score_answer_quality(case, hit.evidence.excerpt, [hit]) < scores["answer_quality"]
    assert tool_confusion_counts(case, ["save_text"]) == (1, 0, 0)


def test_matching_content_hash_does_not_replace_actual_parent_citation_membership():
    """A parent with unresolved refs cannot satisfy a source descriptor claiming actual citations."""
    hit = source_hit(official=False)
    response = answer(hit.evidence)
    without_citations = finalize_answer(response.content, [], retrieval_required=True)
    assert without_citations.content_hash == response.content_hash
    assert without_citations.citations == []
    scope = assess(response, source=response, body_kind="copy_answer",
                   prior=[turn(without_citations, hits=[hit])])
    assert scope.status == "invalid"
    assert scope.verified_evidence == []


def test_cited_answer_reuse_requires_the_resolved_source_even_with_current_hits():
    """Fresh search results cannot hide an absent source binding in a cited copy or transform."""
    hit = source_hit(official=False)
    response = answer(hit.evidence)
    for body_kind in ("copy_answer", "transform_answer"):
        scope = assess(response, hits=[hit], body_kind=body_kind)
        assert scope.status == "invalid"
        assert any("source" in error for error in scope.errors)


def test_source_free_clarification_after_failed_binding_remains_valid():
    """A missing source may produce an uncited clarification with an explicitly empty packet."""
    from src.core.answer_schema import text_document
    response = finalize_answer(text_document("Choose an earlier answer first."), [])
    scope = assess(response, body_kind="copy_answer", packet=[])
    assert scope.status == "complete"
    assert scope.verified_evidence == []
