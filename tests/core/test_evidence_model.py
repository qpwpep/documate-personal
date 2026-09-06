from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.core.documents import DocumentElement, SourceAnchor, TableCell, TableData, build_snapshot
from src.core.evidence import EvidenceRef, RetrievalScore, SearchHit, build_evidence, dedupe_search_hits, parse_search_hits


def evidence(**changes):
    snapshot = build_snapshot(source_uri="upload:///job.py", title="job.py", media_type="text/x-python", source_type="upload", content="retries = 5\n", parser="python", parser_version="1")
    element = DocumentElement(element_id="code", kind="code", text="retries = 5\n", language="python", anchors=[SourceAnchor(kind="code", line_start=8, line_end=8)])
    return build_evidence(snapshot=snapshot, element=element, **changes)


def test_selection_returns_exact_source_text_and_preserves_original_location():
    """Evidence excerpts come from the retained element, not a search summary."""
    item = evidence(start=10, end=11)
    assert item.excerpt == "5"
    assert item.element.text == "retries = 5\n"
    assert item.element.anchors[0].line_start == 8
    assert item.route == "upload"
    assert EvidenceRef.model_validate_json(item.model_dump_json()) == item


def test_serialized_evidence_remains_self_contained_without_source_file(tmp_path):
    """An old answer can display its captured source after an upload is deleted."""
    source = tmp_path / "job.py"
    source.write_text("retries = 5\n", encoding="utf-8")
    captured = evidence().model_dump_json()
    source.unlink()
    restored = EvidenceRef.model_validate_json(captured)
    assert restored.excerpt == "retries = 5\n"
    assert restored.snapshot.content_hash.startswith("sha256:")


def test_selected_table_cells_derive_excerpt_from_source_cells():
    """A table citation identifies stable cells while retaining merged table structure."""
    base = evidence()
    table = DocumentElement(element_id="table", kind="table", table=TableData(cells=[TableCell(cell_id="header", row=0, col=0, col_span=2, text="Options"), TableCell(cell_id="name", row=1, col=0, text="retries"), TableCell(cell_id="value", row=1, col=1, text="3")]))
    selected = build_evidence(snapshot=base.snapshot, element=table, cell_ids=["value", "name"])
    assert selected.excerpt == "retries | 3"
    assert selected.selection.cell_ids == ["name", "value"]
    assert selected.element.table.cells[0].col_span == 2


def test_whole_table_evidence_uses_structured_cells_without_a_duplicate_text_body():
    """A structured table can be cited without separately storing a flattened body."""
    base = evidence()
    table = DocumentElement(element_id="table", kind="table", table=TableData(cells=[TableCell(cell_id="name", row=0, col=0, text="retries"), TableCell(cell_id="value", row=0, col=1, text="3")]))
    selected = build_evidence(snapshot=base.snapshot, element=table)
    assert selected.excerpt == "retries | 3"
    assert selected.element.text == ""
    assert selected.selection.cell_ids == ["name", "value"]


@pytest.mark.parametrize("selection", [{"start": -1}, {"start": 11, "end": 3}, {"end": 100}, {"cell_ids": ["unknown"]}])
def test_invalid_source_selection_is_rejected(selection):
    """A citation cannot invent offsets or table cells absent from its source."""
    with pytest.raises((ValidationError, ValueError)):
        evidence(**selection)


def test_evidence_identity_uses_snapshot_and_source_selection_not_retrieval_score():
    """Re-ranking preserves citation identity but selecting another range changes it."""
    full = evidence()
    selected = evidence(start=10, end=11)
    assert full.id != selected.id
    first = SearchHit(evidence=full, score=RetrievalScore(metric="cosine_distance", raw=0.2, direction="lower"), rank=1)
    second = SearchHit(evidence=full, score=RetrievalScore(metric="cosine_distance", raw=0.5, direction="lower"), rank=3)
    assert dedupe_search_hits([first, second]) == [first]


def test_search_payload_reports_invalid_items_without_losing_valid_results():
    """Malformed tool hits are reported while valid hits remain available."""
    hit = SearchHit(evidence=evidence(), score=RetrievalScore(metric="rank", raw=1, direction="lower"), rank=1)
    errors = []
    payload = json.dumps({"hits": [hit.model_dump(mode="json"), {"old_schema": True}]})
    assert parse_search_hits(payload, errors=errors) == [hit]
    assert len(errors) == 1
    assert "[1]" in errors[0]


def test_retrieval_score_does_not_accept_answer_confidence():
    """Retrieval scores retain their metric and never masquerade as answer confidence."""
    with pytest.raises(ValidationError):
        RetrievalScore(metric="cosine", raw=0.7, normalized=1.5, direction="higher")


def test_provider_hit_can_retain_rank_when_score_is_unavailable():
    """A provider that omits scores must not make up a zero or confidence value."""
    hit = SearchHit(evidence=evidence(), score=RetrievalScore(metric="provider_relevance", raw=None, direction="higher"), rank=2)
    assert hit.score.raw is None
    assert hit.score.normalized is None
    assert hit.rank == 2


@pytest.mark.parametrize("part,field,value", [("element", "text", "retries = 7\n"), ("selection", "end", 5), ("element", "anchors", [{"kind": "code", "line_start": 99, "line_end": 99}])])
def test_citation_identity_rejects_changed_text_selection_or_source_location(part, field, value):
    """A captured citation cannot retain its ID after its content or location changes."""
    payload = evidence().model_dump(mode="json")
    payload[part][field] = value
    with pytest.raises(ValidationError, match="identity"):
        EvidenceRef.model_validate(payload)
