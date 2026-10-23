"""Physical line claims are checked against the retained original text."""
import pytest

from src.core.answer_schema import AnswerDocument, AnswerResponse
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import build_evidence


CODE = 'grouped = sales.groupby("region")["amount"].sum()'


def evidence(text=None, *, start=0, end=None, exact=True):
    text = text or "\n" * 29 + CODE + "\n"
    snapshot = build_snapshot(
        source_uri="upload://sample.py", title="sample.py", media_type="text/x-python",
        source_type="upload", content=text, parser="python", parser_version="1",
    )
    anchor = SourceAnchor(
        kind="code", start=0, end=len(text), line_start=1,
        line_end=len(text.splitlines()), precision="exact" if exact else "element",
    )
    return build_evidence(snapshot=snapshot, element=DocumentElement(
        element_id="source", kind="code", text=text, anchors=[anchor],
    ), start=start, end=end)


def response(ref, claim, *, code=CODE, basis="source"):
    return AnswerResponse.from_document(AnswerDocument.model_validate({"blocks": [
        {"type": "code", "language": "python", "content": {
            "text": code, "basis": "source", "refs": [ref.id],
        }},
        {"type": "paragraph", "content": [{"text": claim, "basis": basis, "refs": [ref.id]}]},
    ]}), [ref], retrieval_required=True)


@pytest.mark.parametrize("line", [28, 29, 33])
def test_wrong_file_line_is_rejected_even_when_the_code_is_exact(line):
    result = response(evidence(), f"sample.py의 {line}행입니다.")
    assert result.checks[1].support_status == "unsupported"
    assert "source_location_mismatch" in result.checks[1].issues


def test_correct_line_still_requires_semantic_validation():
    result = response(evidence(), "sample.py의 30행입니다.")
    assert result.checks[1].support_status == "not_evaluated"
    assert result.checks[1].issues == []


@pytest.mark.parametrize("line", [30, 33])
def test_duplicate_literal_can_bind_to_each_known_physical_occurrence(line):
    ref = evidence("\n" * 29 + CODE + "\n\n\n" + CODE + "\n")
    assert response(ref, f"sample.py의 {line}행입니다.").checks[1].issues == []


def test_duplicate_literal_does_not_authorize_an_intervening_line():
    ref = evidence("\n" * 29 + CODE + "\n\n\n" + CODE + "\n")
    assert response(ref, "sample.py의 31행입니다.").checks[1].support_status == "unsupported"


@pytest.mark.parametrize("claim, expected", [
    ("sample.py의 30~31행입니다.", "not_evaluated"),
    ("sample.py의 29-30행입니다.", "unsupported"),
    ("In sample.py, lines 30–31.", "not_evaluated"),
    ("In sample.py, lines 29–30.", "unsupported"),
])
def test_multiline_literal_has_original_line_range(claim, expected):
    code = CODE + "\nprint(grouped)"
    ref = evidence("\n" * 29 + code + "\n")
    assert response(ref, claim, code=code).checks[1].support_status == expected


def test_midline_selection_keeps_physical_line_not_excerpt_relative_line():
    text = "\n" * 29 + "    " + CODE + "\n"
    ref = evidence(text, start=text.index(CODE), end=text.index(CODE) + len(CODE))
    assert response(ref, "sample.py의 30행입니다.").checks[1].issues == []
    assert response(ref, "sample.py의 1행입니다.").checks[1].support_status == "unsupported"


def test_unknown_provenance_cannot_certify_a_line_number():
    result = response(evidence(exact=False), "sample.py의 30행입니다.")
    assert result.checks[1].support_status == "unsupported"
    assert "source_location_unverifiable" in result.checks[1].issues


@pytest.mark.parametrize("claim", [
    "DataFrame에는 30행이 있습니다.",
    "결과 예시는 30행과 2개 열입니다.",
    "sample.py는 30행의 데이터를 처리합니다.",
])
def test_dataframe_row_counts_are_not_file_positions(claim):
    assert response(evidence(), claim).checks[1].issues == []


def test_literal_source_text_is_not_reinterpreted_as_a_location_claim():
    literal = 'message = "sample.py의 29행입니다."'
    ref = evidence("\n" * 29 + literal + "\n")
    result = response(ref, "원문 문자열은 `sample.py의 29행입니다.`입니다.", code=literal)
    assert all(not check.issues for check in result.checks)


def test_inline_literal_can_bind_its_own_location_without_a_code_block():
    ref = evidence()
    document = AnswerDocument.model_validate({"blocks": [{"type": "paragraph", "content": [{
        "text": f"sample.py의 29행은 `{CODE}`입니다.", "basis": "source", "refs": [ref.id],
    }]}]})
    result = AnswerResponse.from_document(document, [ref], retrieval_required=True)
    assert result.checks[0].support_status == "unsupported"


def test_relabeling_as_example_does_not_bypass_a_physical_location_check():
    assert response(evidence(), "sample.py의 29행입니다.", basis="example").checks[1].support_status == "unsupported"


@pytest.mark.parametrize("claim", [
    "`sample.py`의 **29**행입니다.",
    "원본 파일의 줄 번호: 29",
    "The original source is at line 29.",
])
def test_display_formatting_does_not_hide_a_wrong_location(claim):
    assert response(evidence(), claim).checks[1].support_status == "unsupported"


@pytest.mark.parametrize("claim", [
    "sample.py의 DataFrame은 29행입니다.",
    "sample.py는 총 29행의 데이터를 처리합니다.",
])
def test_file_mention_does_not_turn_a_data_row_count_into_a_location(claim):
    assert response(evidence(), claim).checks[1].issues == []


def test_no_literal_binding_does_not_certify_a_bare_file_position():
    ref = evidence()
    document = AnswerDocument.model_validate({"blocks": [{"type": "paragraph", "content": [{
        "text": "sample.py의 30행입니다.", "basis": "source", "refs": [ref.id],
    }]}]})
    result = AnswerResponse.from_document(document, [ref], retrieval_required=True)
    assert result.checks[0].issues == ["source_location_unverifiable"]


def test_other_file_with_same_line_number_cannot_support_the_claim():
    assert response(evidence(), "other.py의 30행입니다.").checks[1].support_status == "unsupported"


def test_equally_near_different_literals_require_an_unambiguous_binding():
    ref = evidence("\n" * 29 + CODE + "\nprint(grouped)\n")
    document = AnswerDocument.model_validate({"blocks": [
        {"type": "code", "content": {"text": CODE, "basis": "source", "refs": [ref.id]}},
        {"type": "paragraph", "content": [{"text": "sample.py의 30행입니다.", "basis": "source", "refs": [ref.id]}]},
        {"type": "code", "content": {"text": "print(grouped)", "basis": "source", "refs": [ref.id]}},
    ]})
    result = AnswerResponse.from_document(document, [ref], retrieval_required=True)
    assert result.checks[1].support_status == "unsupported"
    assert result.checks[1].issues == ["source_location_unverifiable"]


@pytest.mark.parametrize("text", [
    "Output format: line 1 contains the title; line 2 contains the answer.",
    "The first example prints line 1 and then line 2.",
])
def test_bare_english_line_in_instructions_is_not_a_source_position(text):
    document = AnswerDocument.model_validate({"blocks": [{"type": "paragraph", "content": [{
        "text": text, "basis": "interaction", "refs": [],
    }]}]})
    result = AnswerResponse.from_document(document, [], retrieval_required=False)
    assert result.checks[0].issues == []
    assert result.checks[0].support_status == "not_evaluated"


def test_web_prose_reference_does_not_make_output_lines_native_source_positions():
    text = "A heading introduces a section."
    snapshot = build_snapshot(
        source_uri="https://example.test/guide", title="Guide", media_type="text/html",
        source_type="official", content=text, parser="web", parser_version="1",
    )
    ref = build_evidence(snapshot=snapshot, element=DocumentElement(
        element_id="prose", kind="paragraph", text=text, anchors=[SourceAnchor(kind="web")],
    ))
    result = response(ref, "For the output, line 1 contains `A heading`.", code="A heading")
    assert result.checks[1].issues == []


def test_bare_english_line_with_native_code_binding_still_requires_correct_position():
    result = response(evidence(), "This is on line 29.")
    assert result.checks[1].issues == ["source_location_mismatch"]
