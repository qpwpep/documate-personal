from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.answer_schema import (
    AnswerDocument, AnswerResponse, ContentUnit, ParagraphBlock, SemanticCheck,
    SemanticValidation, export_answer_text, finalize_answer,
    semantic_evidence_fingerprint, text_document,
)
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import build_evidence


def source_answer():
    text = "The timeout is 30 seconds."
    snapshot = build_snapshot(
        source_uri="https://docs.example.com/timeout", title="Timeout",
        media_type="text/plain", source_type="official", content=text,
        parser="test", parser_version="1",
    )
    evidence = build_evidence(
        snapshot=snapshot,
        element=DocumentElement(element_id="timeout", kind="paragraph", text=text),
    )
    return finalize_answer(AnswerDocument(blocks=[ParagraphBlock(content=[
        ContentUnit(text="The timeout lasts half a minute.", basis="source", refs=[evidence.id]),
        ContentUnit(text="The timeout lasts 30 seconds.", basis="source", refs=[evidence.id]),
    ])]), [evidence])


def checked_answer(*, probability=0.97, both_units=True):
    response = source_answer()
    checks = [SemanticCheck(name="grounded", unit_id="b0.content.0", probability=probability)]
    if both_units:
        checks.append(SemanticCheck(name="grounded", unit_id="b0.content.1", probability=0.96))
    report = SemanticValidation(
        content_hash=response.content_hash,
        evidence_fingerprint=semantic_evidence_fingerprint(response.citations),
        threshold=0.8,
        checks=[*checks, SemanticCheck(name="request_fulfilled", probability=0.95)],
    )
    return AnswerResponse.model_validate(response.model_dump(mode="json") | {
        "semantic_validation": report.model_dump(mode="json"),
    })


@pytest.mark.parametrize("field", ["content_hash", "evidence_fingerprint"])
def test_semantic_report_must_describe_the_exact_response_revision(field):
    wire = checked_answer().model_dump(mode="json")
    wire["semantic_validation"][field] = "0" * 64

    with pytest.raises(ValidationError, match="semantic validation"):
        AnswerResponse.model_validate(wire)


def test_semantic_report_cannot_refer_to_an_unknown_content_unit():
    wire = checked_answer().model_dump(mode="json")
    wire["semantic_validation"]["checks"][0]["unit_id"] = "b9.content.0"

    with pytest.raises(ValidationError, match="unknown content unit"):
        AnswerResponse.model_validate(wire)


def test_semantic_report_rejects_duplicate_checks_instead_of_hiding_a_failed_result():
    wire = checked_answer().semantic_validation.model_dump(mode="json")
    wire["checks"].append(wire["checks"][0] | {"probability": 0.1})

    with pytest.raises(ValidationError, match="unique"):
        SemanticValidation.model_validate(wire)


@pytest.mark.parametrize("invalid", [-0.01, 1.01, float("nan")])
def test_semantic_probabilities_and_thresholds_are_finite_probabilities(invalid):
    wire = checked_answer().semantic_validation.model_dump(mode="json")

    with pytest.raises(ValidationError):
        SemanticCheck(name="grounded", unit_id="b0.content.0", probability=invalid)
    with pytest.raises(ValidationError):
        SemanticValidation.model_validate(wire | {"threshold": invalid})


def test_semantic_report_requires_a_check_and_uses_an_inclusive_threshold():
    wire = checked_answer(probability=0.8).semantic_validation.model_dump(mode="json")
    assert SemanticValidation.model_validate(wire).passed is True
    wire["checks"][0]["probability"] = 0.799
    assert SemanticValidation.model_validate(wire).passed is False
    with pytest.raises(ValidationError):
        SemanticValidation.model_validate(wire | {"checks": []})


def test_fingerprint_covers_full_cited_evidence_in_order_without_citation_numbers():
    response = source_answer()
    evidence = response.citations[0].evidence
    changed_snapshot = evidence.snapshot.model_copy(update={"title": "A changed title"})
    changed = build_evidence(snapshot=changed_snapshot, element=evidence.element)
    original = semantic_evidence_fingerprint(response.citations)

    assert semantic_evidence_fingerprint([evidence]) == original
    assert semantic_evidence_fingerprint([changed]) != original
    assert semantic_evidence_fingerprint([evidence, changed]) != semantic_evidence_fingerprint([changed, evidence])


def test_export_bytes_are_stable_when_the_same_answer_is_rechecked():
    original = export_answer_text(checked_answer(probability=0.99), include_sources=True).encode("utf-8")
    rechecked = export_answer_text(checked_answer(probability=0.98), include_sources=True).encode("utf-8")
    partial = export_answer_text(checked_answer(probability=0.4, both_units=False), include_sources=True).encode("utf-8")
    unchecked = export_answer_text(source_answer(), include_sources=True).encode("utf-8")

    assert original == rechecked == partial == unchecked
