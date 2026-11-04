"""Source positions bind to captured ranges, independently of prose and filenames."""
import json

import pytest
from pydantic import ValidationError

from src.core.answer_export import answer_export_snapshot
from src.core.answer_schema import AnswerDocument, AnswerResponse, document_hash, export_answer_text
from src.core.answer_schema.source_locations import SOURCE_LOCATION_PLACEHOLDER
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import build_evidence
from src.core.request_contracts import RequestContract
from src.runtime.nodes.synthesis.prompt_builder import build_synthesis_messages


CODE = 'grouped = sales.groupby("region")["amount"].sum()'


def evidence(text=None, *, start=0, end=None, exact=True, title="sample.py", notebook=False):
    text = text if text is not None else "\n" * 29 + CODE + "\n"
    snapshot = build_snapshot(
        source_uri="upload:///file-id", title=title, media_type="text/x-python",
        source_type="upload", content=text, parser="python", parser_version="1",
    )
    anchor = SourceAnchor(
        kind="notebook" if notebook else "code", start=0, end=len(text), line_start=1,
        line_end=len(text.splitlines()), precision="exact" if exact else "element",
        cell_id="cell-source" if notebook else None, cell_index=3 if notebook else None,
    )
    return build_evidence(snapshot=snapshot, element=DocumentElement(
        element_id="source", kind="code", text=text, anchors=[anchor],
    ), start=start, end=end)


def location_document(ref, *, start=None, end=None, text=SOURCE_LOCATION_PLACEHOLDER):
    return AnswerDocument.model_validate({"blocks": [{"type": "paragraph", "content": [{
        "text": text, "basis": "source", "refs": [ref.id],
        "source_location": {"ref": ref.id,
                            "start": ref.selection.start if start is None else start,
                            "end": ref.selection.end if end is None else end},
    }]}]})


@pytest.mark.parametrize("title", ["sales_analysis.py", "sales analysis.py"])
def test_bound_location_uses_source_identity_instead_of_parsing_names(title):
    ref = evidence('data = pd.read_csv("input.csv")\n', title=title)
    document = location_document(ref, end=len(ref.element.text) - 1)

    result = AnswerResponse.from_document(document, [ref], retrieval_required=True)

    assert all(not check.issues for check in result.checks)
    assert result.content.blocks[0].content[0].text == f"{title}:1"
    assert document.blocks[0].content[0].text == SOURCE_LOCATION_PLACEHOLDER
    assert SOURCE_LOCATION_PLACEHOLDER not in export_answer_text(result, include_sources=True)
    assert answer_export_snapshot(result) == result
    assert result.content.blocks[0].content[0].text in export_answer_text(result, include_sources=True)


@pytest.mark.parametrize("text", [CODE, "This source call is conditional."])
def test_location_binding_does_not_discard_generated_content(text):
    ref = evidence()
    document = location_document(ref, start=29, end=29 + len(CODE), text=text)

    result = AnswerResponse.from_document(document, [ref], retrieval_required=True)

    assert result.content.blocks[0].content[0].text == text
    assert result.checks[0].support_status == "unsupported"
    assert result.checks[0].issues == ["source_location_mismatch"]


@pytest.mark.parametrize("subject", ["DataFrame", "데이터셋", "표", "관측 자료"])
def test_general_row_count_prose_is_left_to_semantic_validation(subject):
    ref = evidence()
    text = f"sample.py의 {subject}은 29행입니다."
    document = AnswerDocument.model_validate({"blocks": [{"type": "paragraph", "content": [{
        "text": text, "basis": "source", "refs": [ref.id],
    }]}]})
    result = AnswerResponse.from_document(document, [ref], retrieval_required=True)
    assert result.checks[0].issues == []
    assert result.checks[0].support_status == "not_evaluated"
    assert result.content.blocks[0].content[0].text == text


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_server_generates_physical_range_from_offsets_including_blank_lines(newline):
    source = newline * 29 + CODE + newline + "print(grouped)" + newline
    ref = evidence(source)
    document = location_document(ref, start=source.index(CODE), end=len(source))
    result = AnswerResponse.from_document(document, [ref], retrieval_required=True)
    assert result.content.blocks[0].content[0].text == "sample.py:30–31"
    assert result.checks[0].support_status == "verified_location"
    assert not result.checks[0].issues


def test_repeated_literal_is_bound_to_the_selected_occurrence():
    ref = evidence("x = 1\n\nx = 1\n")
    first = AnswerResponse.from_document(location_document(ref, start=0, end=5), [ref])
    second = AnswerResponse.from_document(location_document(ref, start=7, end=12), [ref])
    assert first.content.blocks[0].content[0].text == "sample.py:1"
    assert second.content.blocks[0].content[0].text == "sample.py:3"


def test_blank_source_line_retains_its_physical_position():
    ref = evidence("x = 1\n\nx = 1\n")
    result = AnswerResponse.from_document(location_document(ref, start=6, end=7), [ref])
    assert result.content.blocks[0].content[0].text == "sample.py:2"
    assert not result.checks[0].issues


def test_midline_selection_and_notebook_cell_keep_their_original_locations():
    source = "\n" * 29 + "    " + CODE + "\n"
    start = source.index(CODE)
    ref = evidence(source, start=start, end=start + len(CODE), title="analysis notes.ipynb", notebook=True)
    result = AnswerResponse.from_document(location_document(ref), [ref])
    assert result.content.blocks[0].content[0].text == "analysis notes.ipynb · cell 4:30"
    assert not result.checks[0].issues


@pytest.mark.parametrize(("start", "end"), [(0, 5), (7, 14)])
def test_location_cannot_recover_text_outside_the_supplied_evidence(start, end):
    ref = evidence("hidden\nvisible\nhidden", start=8, end=13)
    result = AnswerResponse.from_document(location_document(ref, start=start, end=end), [ref])
    assert result.checks[0].support_status == "unsupported"
    assert result.checks[0].issues == ["source_location_mismatch"]


def test_unknown_provenance_cannot_generate_a_line_number():
    ref = evidence(exact=False)
    result = AnswerResponse.from_document(location_document(ref), [ref])
    assert result.checks[0].support_status == "unsupported"
    assert result.checks[0].issues == ["source_location_unverifiable"]
    assert result.content.blocks[0].content[0].text == SOURCE_LOCATION_PLACEHOLDER


def test_unknown_evidence_cannot_generate_a_line_number():
    ref = evidence()
    result = AnswerResponse.from_document(location_document(ref), [])
    assert result.checks[0].reference_status == "missing"
    assert result.checks[0].support_status == "unsupported"
    assert "source_location_unverifiable" in result.checks[0].issues


def test_location_must_bind_to_its_single_declared_reference():
    ref = evidence()
    payload = location_document(ref).model_dump(mode="json")
    payload["blocks"][0]["content"][0]["source_location"]["ref"] = "other-source"
    with pytest.raises(ValidationError, match="one reference"):
        AnswerDocument.model_validate(payload)


@pytest.mark.parametrize(("start", "end"), [(2, 2), (3, 2)])
def test_empty_or_reversed_location_ranges_are_rejected(start, end):
    with pytest.raises(ValidationError, match="nonempty range"):
        location_document(evidence(), start=start, end=end)


def test_revalidating_a_stored_response_never_rewrites_tampered_location_text():
    ref = evidence()
    result = AnswerResponse.from_document(location_document(ref, start=29, end=29 + len(CODE)), [ref])
    payload = result.model_dump(mode="json")
    payload["content"]["blocks"][0]["content"][0]["text"] = "sample.py의 29행입니다."
    payload["content_hash"] = document_hash(AnswerDocument.model_validate(payload["content"]))
    with pytest.raises(ValidationError, match="checks must match"):
        AnswerResponse.model_validate(payload)


def test_copying_a_checked_location_retains_the_same_text_and_hash():
    ref = evidence()
    result = AnswerResponse.from_document(location_document(ref, start=29, end=29 + len(CODE)), [ref])
    assert answer_export_snapshot(result).content_hash == result.content_hash
    assert AnswerResponse.from_document(result.content, [ref]) == result


def test_transform_prompt_aliases_locations_and_refs_without_changing_stored_answer():
    ref = evidence()
    original = AnswerResponse.from_document(location_document(ref, start=29, end=29 + len(CODE)), [ref])
    messages, _, _ = build_synthesis_messages(
        state=build_graph_state_input(user_input="위 답변을 영어로 바꿔줘"),
        request_contract=RequestContract(), action_rules=[], evidence_packet=[ref],
        attempt=1, max_turns=6, requirement_ids_by_evidence={},
        reference_aliases={"e1": ref.id}, retrieval_required=True, source_response=original,
    )
    bound_message = next(str(message.content) for message in messages
                         if str(message.content).startswith("[Bound Source Answer]"))
    record = json.loads(bound_message.split("\n", 2)[2])
    unit = record["content"]["blocks"][0]["content"][0]
    assert unit["refs"] == ["e1"]
    assert unit["source_location"] == {"ref": "e1", "start": 29, "end": 29 + len(CODE)}
    assert unit["text"] == "sample.py:30"
    assert record["response_hash"] == original.content_hash
    assert original.content.blocks[0].content[0].source_location.ref == ref.id
    assert answer_export_snapshot(original) == original
