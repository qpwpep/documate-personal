"""Short model references resolve only at the structured response boundary."""
import json
import time

from src.core.answer_schema import AnswerDocument, finalize_answer, iter_content_units, text_document
from src.runtime.nodes.synthesis.budgets import SynthesisBudgetProfile
from src.runtime.nodes.synthesis.context import build_synthesis_context, prepare_synthesis_inputs
from src.runtime.nodes.synthesis.pipeline import run_synthesis_pipeline
from src.runtime.nodes.synthesis.prompt_builder import build_synthesis_messages
from tests.core.test_synthesis_validation import ModelBoundary, _hit, _state


def _prepare(hits, *, limit=1800):
    state = _state(hits)
    context = build_synthesis_context(state=state, has_default_slack_destination=False)
    prepared = prepare_synthesis_inputs(
        state=state, context=context,
        budget_profile=SynthesisBudgetProfile("docs", limit, limit * 6, 6),
        max_turns=6, prompt_snippet_char_limit=limit, prompt_evidence_char_budget=limit * 6,
    )
    return state, prepared


def _json_block(messages, marker):
    content = next(str(message.content) for message in messages if str(message.content).startswith(marker))
    return json.loads(content[content.index("[", len(marker)):])


def test_prepared_packet_and_requirement_coverage_use_the_same_short_references():
    """Only reference fields are aliased, while internal source identity and ranges stay exact."""
    hits = [_hit("First setting."), _hit("Second setting.")]
    original = [hit.model_dump(mode="json") for hit in hits]
    _state_value, prepared = _prepare(hits)
    packet = _json_block(prepared.model_messages, "[Evidence Packet]")
    requirements = _json_block(prepared.model_messages, "[Retrieval Requirements]")

    assert prepared.reference_aliases == {f"e{i}": source.id for i, source in enumerate(prepared.evidence_packet, 1)}
    assert [source["id"] for source in packet] == ["e1", "e2"]
    assert requirements[0]["coverage"]["evidence_ids"] == ["e1", "e2"]
    assert [(source["excerpt"], source["snapshot_id"], source["selection"]) for source in packet] == [
        (source.excerpt, source.snapshot.snapshot_id, source.selection.model_dump(mode="json"))
        for source in prepared.evidence_packet
    ]
    assert [hit.model_dump(mode="json") for hit in hits] == original


def test_aliases_resolve_in_every_content_unit_without_rewriting_text_or_sources():
    """Aliases in all block layouts become canonical refs, and output remains byte-equivalent otherwise."""
    hit = _hit("e1 is part of the original source text.")
    _state_value, prepared = _prepare([hit])
    unit = {"text": hit.evidence.excerpt, "basis": "excerpt", "refs": ["e1"]}
    document = AnswerDocument.model_validate({"blocks": [
        {"type": "paragraph", "content": [unit]},
        {"type": "list", "ordered": False, "items": [unit]},
        {"type": "code", "language": "text", "content": unit},
        {"type": "heading", "level": 2, "content": unit},
        {"type": "table", "columns": [unit], "rows": [[unit]]},
    ]})
    original = document.model_dump(mode="json")
    expected_document = document.model_copy(deep=True)
    for _path, content in iter_content_units(expected_document):
        content.refs = [prepared.evidence_packet[0].id]
    expected = finalize_answer(expected_document, prepared.evidence_packet, retrieval_required=True)

    outcome = run_synthesis_pipeline(
        structured_synthesizer=ModelBoundary(malformed=document), structured_synthesizer_compact=None,
        prepared=prepared, compact_prepared=None, stage_started=time.perf_counter(),
    )

    assert outcome.result == expected
    assert outcome.evidence_packet == prepared.evidence_packet
    assert document.model_dump(mode="json") == original
    assert all(check.reference_status == "resolved" and check.support_status == "exact_match" for check in outcome.result.checks)


def test_unknown_alias_is_not_repaired_or_silently_attached_to_a_source():
    """An unrecognized short reference survives to the existing validation failure."""
    _state_value, prepared = _prepare([_hit()])
    document = text_document("Default mode is safe.", basis="source", refs=["e99"])
    outcome = run_synthesis_pipeline(
        structured_synthesizer=ModelBoundary(malformed=document), structured_synthesizer_compact=None,
        prepared=prepared, compact_prepared=None, stage_started=time.perf_counter(),
    )
    assert outcome.result.content == document
    assert outcome.result.checks[0].reference_status == "missing"
    assert outcome.result.citations == []


def test_legacy_canonical_refs_remain_valid_with_a_short_reference_packet():
    """Existing full source IDs remain accepted without changing the external answer format."""
    _state_value, prepared = _prepare([_hit()])
    source = prepared.evidence_packet[0]
    document = text_document(source.excerpt, basis="excerpt", refs=[source.id])
    outcome = run_synthesis_pipeline(
        structured_synthesizer=ModelBoundary(malformed=document), structured_synthesizer_compact=None,
        prepared=prepared, compact_prepared=None, stage_started=time.perf_counter(),
    )
    assert outcome.result == finalize_answer(document, [source], retrieval_required=True)


def test_compact_attempt_resolves_aliases_against_its_own_selected_ranges():
    """The same wire alias resolves to the compact attempt's range instead of the first attempt's ID."""
    hit = _hit("source detail " * 300)
    _state_value, prepared = _prepare([hit], limit=1800)
    _state_value, compact = _prepare([hit], limit=900)
    assert prepared.evidence_packet[0].id != compact.evidence_packet[0].id
    source = compact.evidence_packet[0]
    outcome = run_synthesis_pipeline(
        structured_synthesizer=ModelBoundary(error=TimeoutError("timeout")),
        structured_synthesizer_compact=ModelBoundary(malformed=text_document(source.excerpt, basis="excerpt", refs=["e1"])),
        prepared=prepared, compact_prepared=compact, stage_started=time.perf_counter(),
    )
    assert outcome.result == finalize_answer(text_document(source.excerpt, basis="excerpt", refs=[source.id]), [source], retrieval_required=True)
    assert outcome.evidence_packet == compact.evidence_packet


def test_direct_prompt_builder_without_aliases_keeps_the_existing_reference_contract():
    """Callers that do not opt into aliases continue receiving canonical source IDs."""
    hit = _hit()
    messages, _before, _after = build_synthesis_messages(
        state=_state([hit]), action_rules=[], evidence_packet=[hit.evidence], attempt=1, max_turns=6,
    )
    assert _json_block(messages, "[Evidence Packet]")[0]["id"] == hit.evidence.id
