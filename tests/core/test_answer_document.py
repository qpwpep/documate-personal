import pytest
from pydantic import ValidationError

from src.core.answer_schema import (
    AnswerDocument, AnswerResponse, ContentUnit, ParagraphBlock, ListBlock, CodeBlock, TableBlock,
    HeadingBlock, ActionReceipt, finalize_answer, export_answer_text,
    iter_content_units, filter_document_units, text_document, ResponseIssue, build_grounded_response,
)
from src.core.documents import DocumentElement, SourceAnchor, TableCell, TableData, build_snapshot
from src.core.evidence import build_evidence


def source(text="retries = 3", name="guide.txt"):
    snapshot = build_snapshot(source_uri=f"upload:///{name}", title=name, media_type="text/plain", source_type="upload", content=text, parser="text", parser_version="1")
    return build_evidence(snapshot=snapshot, element=DocumentElement(element_id="body", kind="paragraph", text=text))


def test_document_rejects_independently_generated_legacy_bodies():
    with pytest.raises(ValidationError):
        AnswerDocument.model_validate({"blocks": [], "answer": "second body"})


def test_every_displayed_content_unit_is_checked_including_table_headers():
    unit = lambda text: ContentUnit(text=text, basis="source", refs=["missing"])
    doc = AnswerDocument(blocks=[
        HeadingBlock(content=unit("Heading")),
        ParagraphBlock(content=[unit("Paragraph")]),
        ListBlock(items=[unit("Item")]),
        CodeBlock(language="python", content=unit("x = 1\nprint(x)")),
        TableBlock(columns=[unit("Column")], rows=[[unit("Cell")]]),
    ])
    result = finalize_answer(doc, [])
    assert len(result.checks) == 6
    assert {c.unit_id for c in result.checks} == {p for p, _ in iter_content_units(doc)}
    assert all(c.reference_status == "missing" for c in result.checks)
    assert all(c.support_status == "not_evaluated" for c in result.checks)
    assert all(u.text in export_answer_text(result) for _, u in iter_content_units(doc))


def test_filtered_content_disappears_from_document_and_export():
    doc = AnswerDocument(blocks=[ParagraphBlock(content=[
        ContentUnit(text="Keep", basis="interaction"),
        ContentUnit(text="Remove this unsupported statement", basis="source", refs=["bad"]),
    ])])
    path, _ = next(iter_content_units(doc))
    filtered = filter_document_units(doc, {path})
    result = finalize_answer(filtered, [])
    assert export_answer_text(result) == "Keep"
    assert "unsupported" not in result.model_dump_json()


def test_code_whitespace_and_receipt_separation_are_preserved():
    code = "def run():\n    return 1\n"
    document = AnswerDocument(blocks=[CodeBlock(language="python", content=ContentUnit(text=code, basis="example"))])
    result = finalize_answer(document, [], actions=[ActionReceipt(kind="save_text", status="success", file_path="result.txt")])
    assert code in export_answer_text(result)
    assert "result.txt" not in export_answer_text(result)
    assert result.content == document


def test_response_detects_content_changed_after_checks():
    result = finalize_answer(text_document("original"), [])
    payload = result.model_dump(mode="json")
    payload["content"]["blocks"][0]["content"][0]["text"] = "changed"
    with pytest.raises(ValidationError):
        type(result).model_validate(payload)


def test_response_cannot_drop_its_revision_hash_to_accept_a_changed_body():
    """A nonempty response must retain the content revision used for its checks."""
    result = finalize_answer(text_document("original"), [])
    payload = result.model_dump(mode="json")
    payload.pop("content_hash")
    payload["content"]["blocks"][0]["content"][0]["text"] = "changed"
    with pytest.raises(ValidationError):
        AnswerResponse.model_validate(payload)


def test_export_rejects_an_in_memory_body_changed_after_finalization():
    """A delivery cannot export changed content under an earlier check revision."""
    result = finalize_answer(text_document("original"), [])
    result.content.blocks[0].content[0].text = "changed"
    with pytest.raises(ValidationError):
        export_answer_text(result, include_sources=True)


def test_table_shape_is_a_public_contract():
    unit = ContentUnit(text="label")
    with pytest.raises(ValidationError):
        TableBlock(columns=[unit, unit], rows=[[unit]])


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "reordered"])
def test_response_requires_one_check_per_displayed_unit_in_reading_order(change):
    """An API response cannot omit, duplicate or misassign a displayed unit's check."""
    result = finalize_answer(AnswerDocument(blocks=[ListBlock(items=[ContentUnit(text="first"), ContentUnit(text="second")])]), [])
    payload = result.model_dump(mode="json")
    if change == "missing":
        payload["checks"].pop()
    elif change == "duplicate":
        payload["checks"].append(payload["checks"][0])
    elif change == "unknown":
        payload["checks"][0]["unit_id"] = "b99.content.0"
    else:
        payload["checks"].reverse()
    with pytest.raises(ValidationError):
        AnswerResponse.model_validate(payload)


def test_citations_follow_first_reference_order_and_exclude_unused_packet_sources():
    """The user sees citations in answer order, independently of retrieval ranking."""
    first, second, unused = source("first", "1.txt"), source("second", "2.txt"), source("unused", "3.txt")
    result = finalize_answer(AnswerDocument(blocks=[ParagraphBlock(content=[ContentUnit(text="second", basis="excerpt", refs=[second.id]), ContentUnit(text="first", basis="excerpt", refs=[first.id])])]), [first, unused, second])
    assert [(citation.number, citation.evidence) for citation in result.citations] == [(1, second), (2, first)]
    assert export_answer_text(result) == "second [1] first [2]"


@pytest.mark.parametrize("change", ["unused", "reordered", "stale"])
def test_response_rejects_citations_that_no_longer_match_visible_references(change):
    """Source panels cannot contain another answer's unused or misnumbered citations."""
    first, second = source("first", "1.txt"), source("second", "2.txt")
    result = finalize_answer(AnswerDocument(blocks=[ParagraphBlock(content=[ContentUnit(text="first", basis="excerpt", refs=[first.id]), ContentUnit(text="second", basis="excerpt", refs=[second.id])])]), [first, second])
    payload = result.model_dump(mode="json")
    if change == "unused":
        payload["citations"].append({"number": 3, "evidence": source("unused").model_dump(mode="json")})
    elif change == "reordered":
        payload["citations"][0]["evidence"], payload["citations"][1]["evidence"] = payload["citations"][1]["evidence"], payload["citations"][0]["evidence"]
    else:
        payload["citations"].pop()
    with pytest.raises(ValidationError):
        AnswerResponse.model_validate(payload)


def test_source_summary_never_inherits_exact_match_or_semantic_verification():
    """Resolving a source ID does not certify the model's summary as supported."""
    item = source()
    result = finalize_answer(text_document("The setting is three.", basis="source", refs=[item.id]), [item])
    assert result.checks[0].reference_status == "resolved"
    assert result.checks[0].support_status == "not_evaluated"
    payload = result.model_dump(mode="json")
    payload["checks"][0]["support_status"] = "exact_match"
    with pytest.raises(ValidationError):
        AnswerResponse.model_validate(payload)


@pytest.mark.parametrize("changed_text", ["retries = 5\nmode = safe\n", "retries = 3", "retries = 3\nmode = safe"])
def test_exact_excerpt_check_is_recomputed_from_the_retained_source(changed_text):
    """Only an unchanged verbatim source selection earns exact_match."""
    item = source("retries = 3\nmode = safe\n")
    exact = finalize_answer(text_document(item.excerpt, basis="excerpt", refs=[item.id]), [item])
    changed = finalize_answer(text_document(changed_text, basis="excerpt", refs=[item.id]), [item])
    assert exact.checks[0].support_status == "exact_match"
    assert changed.checks[0].support_status == "unsupported"
    assert changed.checks[0].issues == ["excerpt_does_not_match_source"]


def test_response_preserves_reference_policy_for_example_validation():
    """Deserializing a response preserves the retrieval policy used for its checks."""
    result = finalize_answer(text_document("retries = 3", basis="example"), [], retrieval_required=True)
    assert result.retrieval_required is True
    assert result.checks[0].reference_status == "missing"
    assert AnswerResponse.model_validate_json(result.model_dump_json()) == result


def test_response_rejects_issue_attached_to_a_removed_content_unit():
    """A warning cannot point at a removed or unrelated body position."""
    with pytest.raises(ValidationError):
        finalize_answer(text_document("visible"), [], issues=[ResponseIssue(code="source_limit", message="limit", unit_id="b8.content.0")])


def test_filter_removes_an_entire_table_row_when_one_cell_is_rejected():
    """Filtering a bad table cell never shifts another value into its column."""
    unit = lambda text: ContentUnit(text=text)
    document = AnswerDocument(blocks=[TableBlock(columns=[unit("Setting"), unit("Value")], rows=[[unit("keep"), unit("3")], [unit("remove"), unit("wrong")]])])
    paths = {path for path, content in iter_content_units(document) if content.text != "wrong"}
    result = finalize_answer(filter_document_units(document, paths), [])
    assert result.content.blocks[0].rows == [[unit("keep"), unit("3")]]
    assert "remove" not in export_answer_text(result)


def test_table_source_fallback_keeps_merged_geometry_and_cell_selection_in_citation():
    """A fallback does not invent a rectangular answer table for merged source cells."""
    base = source("Options\nretries 3")
    element = DocumentElement(element_id="table", kind="table", table=TableData(cells=[TableCell(cell_id="heading", row=0, col=0, col_span=2, text="Options"), TableCell(cell_id="key", row=1, col=0, text="retries"), TableCell(cell_id="value", row=1, col=1, text="3")]))
    item = build_evidence(snapshot=base.snapshot, element=element, cell_ids=["key", "value"])
    result = build_grounded_response([item])
    assert result.content.blocks[0].content[0].text == "retries | 3"
    assert result.checks[0].support_status == "exact_match"
    assert result.citations[0].evidence == item
    assert result.citations[0].evidence.element.table.cells[0].col_span == 2


def test_delivery_export_preserves_source_location_limitations_and_generated_code_status():
    """Saved or sent answers retain the same provenance and uncertainty as the UI."""
    code = "def run():\n    return 3\n"
    snapshot = build_snapshot(source_uri="upload:///example.ipynb", title="example.ipynb", media_type="application/x-ipynb+json", source_type="upload", content=code, parser="notebook", parser_version="1", quality_issues=["A cell could not be read."])
    element = DocumentElement(element_id="native-cell", kind="code", language="python", text=code, heading_path=["Examples"], anchors=[SourceAnchor(kind="notebook", cell_id="native-cell", cell_index=2, line_start=1, line_end=2)])
    item = build_evidence(snapshot=snapshot, element=element)
    result = finalize_answer(AnswerDocument(blocks=[CodeBlock(language="python", content=ContentUnit(text=code, basis="example", refs=[item.id]))]), [item], issues=[ResponseIssue(code="answer_incomplete", message="The example does not cover every requested condition.")])
    delivered = export_answer_text(result, include_sources=True)
    assert f"```python\n{code}```" in delivered
    assert "코드 예시 · 실행 확인 안 됨" in delivered
    assert "The example does not cover every requested condition." in delivered
    assert "A cell could not be read." in delivered
    assert "의미적 지지는 별도로 검증하지 않았습니다" in delivered
    assert "native-cell" in delivered and "cell 3" in delivered
    assert "lines 1-2" in delivered and "Examples" in delivered
    assert item.snapshot.snapshot_id in delivered
    assert "코드 예시" not in export_answer_text(result)


def test_delivery_export_identifies_selected_table_cells_and_known_page_precision():
    """A table citation's text export retains selected cells and known page bounds."""
    base = source()
    table = DocumentElement(element_id="table", kind="table", table=TableData(cells=[TableCell(cell_id="key", row=0, col=0, text="retries"), TableCell(cell_id="value", row=0, col=1, text="3")]), anchors=[SourceAnchor(kind="page", page_no=3, bbox=(0.1, 0.2, 0.9, 0.8), coordinate_space="normalized", precision="element")])
    item = build_evidence(snapshot=base.snapshot, element=table, cell_ids=["value"])
    result = build_grounded_response([item], message="Only a source excerpt is available.")
    delivered = export_answer_text(result, include_sources=True)
    assert "table cells value" in delivered
    assert "page 3" in delivered
    assert "bbox (0.1, 0.2, 0.9, 0.8)" in delivered
    assert "precision element" in delivered
    assert "원문 발췌" in delivered
    assert "Only a source excerpt is available." in delivered
