"""Sentence quotas use the confirmed scope and the entire displayed prose."""
import pytest
from pydantic import ValidationError

from src.core.answer_schema import AnswerDocument
from src.core.request_contracts import AnswerContract, RequestContract, check_answer_contract


def _contract(*requirements):
    return AnswerContract.model_validate({"sentence_counts": list(requirements)})


def _count(scope, minimum=1, maximum=1, **extra):
    return {"scope": scope, "minimum": minimum, "maximum": maximum, **extra}


def _paragraph(*units):
    return AnswerDocument.model_validate({"blocks": [{"type": "paragraph", "content": [
        {"text": text, "sentence_scope": scope} for text, scope in units
    ]}]})


@pytest.mark.parametrize("units", [
    [("일반 파일을 판별합니다. 일반 파일이면 참을 반환합니다.", 1)],
    [("일반 파일을 판별합니다.", 1), ("일반 파일이면 참을 반환합니다.", 1)],
])
def test_repeated_introduction_counts_in_the_same_scope(units):
    check = check_answer_contract(_contract(_count("일반 파일인 경우")), _paragraph(*units))
    assert check.missing_required == ["sentence_count:1"]


def test_intro_without_a_contract_scope_cannot_escape_the_count():
    document = _paragraph(("일반 파일을 판별합니다.", None), ("일반 파일이면 참을 반환합니다.", 1))
    check = check_answer_contract(_contract(_count("일반 파일인 경우")), document)
    assert "sentence_scope" in check.missing_required


def test_provenance_units_are_joined_before_counting_sentences():
    document = _paragraph(("`Path.is_file()`은 일반 파일을 가리킬 때", 1), ("`True`를 반환합니다.", 1))
    assert check_answer_contract(_contract(_count("is_file True 조건")), document).valid


def test_unterminated_prose_still_counts():
    document = _paragraph(("첫 문장입니다. 뒤에 덧붙인 설명", 1))
    assert check_answer_contract(_contract(_count("설명")), document).missing_required == ["sentence_count:1"]


@pytest.mark.parametrize("text", [
    "# 도입\n\n설명입니다.",
    "- 첫 설명\n- 두 번째 설명",
    "첫 설명\n\n두 번째 설명",
    "> 첫 설명\n\n> 두 번째 설명",
    "| 제목 |\n| --- |\n| 설명 |",
])
def test_markdown_blocks_within_a_unit_keep_their_prose_boundaries(text):
    check = check_answer_contract(_contract(_count("설명")), _paragraph((text, 1)))
    assert check.missing_required == ["sentence_count:1"]


def test_markdown_heading_does_not_merge_with_the_next_provenance_unit():
    document = _paragraph(("# 도입", 1), ("설명입니다.", 1))
    assert check_answer_contract(_contract(_count("설명")), document).missing_required == ["sentence_count:1"]


def test_fenced_markdown_code_is_not_prose():
    document = _paragraph(("```python\nx = 1.5\nprint(x)\n```", None), ("숫자를 출력합니다.", 1))
    assert check_answer_contract(_contract(_count("설명")), document).valid


def test_a_shared_sentence_cannot_fill_two_independent_scope_quotas():
    document = _paragraph(("일반 파일이면 참을 반환하며,", 1), ("디렉터리이면 참을 반환합니다.", 2))
    check = check_answer_contract(_contract(_count("파일 조건"), _count("디렉터리 조건")), document)
    assert "sentence_scope" in check.missing_required


def test_independent_content_scopes_and_whole_body_count_are_separate():
    contract = _contract(_count(None, 2, 2), _count("True 반환 조건"), _count("False 반환 조건"))
    document = _paragraph(("일반 파일이면 참을 반환합니다.", 2), ("경로가 없으면 거짓을 반환합니다.", 3))
    assert check_answer_contract(contract, document).valid


@pytest.mark.parametrize("text", [
    "Python 3.15.2에서 pathlib.Path.is_file()은 일반 파일이면 참을 반환합니다.",
    "`pathlib.Path('a.b').is_file()`은 `True`를 반환합니다.",
    "[공식 문서](https://docs.python.org/3.15/library/pathlib.html)에 설명되어 있습니다.",
    "**파일**이면 참을 반환합니다!",
    '경로가 파일인가요? 아니다! 확인합니다。 끝입니다！ 정말인가요？',
])
def test_markdown_identifiers_and_decimal_points_do_not_create_sentences(text):
    expected = 5 if text.startswith("경로가") else 1
    check = check_answer_contract(_contract(_count(None, expected, expected)), _paragraph((text, None)))
    assert check.valid


def test_partial_response_waives_only_the_minimum_of_an_absent_scope():
    contract = _contract(_count("파일 조건"), _count("디렉터리 조건"))
    document = _paragraph(("일반 파일이면 참을 반환합니다.", 1))
    assert check_answer_contract(contract, document).missing_required == ["sentence_count:2"]
    assert check_answer_contract(contract, document, partial=True).valid
    repeated = _paragraph(("일반 파일을 판별합니다. 일반 파일이면 참을 반환합니다.", 1))
    assert check_answer_contract(contract, repeated, partial=True).missing_required == ["sentence_count:1"]


def test_partial_response_keeps_whole_body_and_present_scope_minimums():
    contract = _contract(_count(None, 2, 2), _count("파일 조건", 2, 3), _count("디렉터리 조건"))
    check = check_answer_contract(contract, _paragraph(("일반 파일이면 참을 반환합니다.", 2)), partial=True)
    assert check.missing_required == ["sentence_count:1", "sentence_count:2"]


@pytest.mark.parametrize("block", [
    {"type": "heading", "content": {"text": "반복하는 설명입니다.", "sentence_scope": 1}},
    {"type": "list", "items": [{"text": "반복하는 설명입니다.", "sentence_scope": 1}]},
    {"type": "table", "columns": [{"text": "반복하는 설명입니다.", "sentence_scope": 1}], "rows": []},
])
def test_presentation_does_not_exempt_introductory_prose(block):
    document = _paragraph(("반복하는 설명입니다.", 1))
    document = AnswerDocument.model_validate({"blocks": [block, *document.model_dump()["blocks"]]})
    assert check_answer_contract(_contract(_count("설명")), document).missing_required == ["sentence_count:1"]


def test_standalone_code_does_not_count_as_prose():
    document = AnswerDocument.model_validate({"blocks": [
        {"type": "code", "content": {"text": "x = 1.5\nprint(x)"}},
        {"type": "paragraph", "content": [{"text": "숫자를 출력합니다.", "sentence_scope": 1}]},
    ]})
    assert check_answer_contract(_contract(_count("설명")), document).valid


@pytest.mark.parametrize("scope", [None, 2])
def test_required_scope_references_must_resolve(scope):
    assert "sentence_scope" in check_answer_contract(
        _contract(_count("설명")), _paragraph(("설명합니다.", scope)),
    ).missing_required


def test_preference_is_not_a_local_acceptance_gate():
    contract = _contract(_count("가능하면 간결하게", mode="preferred"))
    assert check_answer_contract(contract, _paragraph(("첫 문장입니다. 다음 문장입니다.", None))).valid


def test_scoped_counts_still_require_semantic_membership_but_global_counts_do_not():
    document = _paragraph(("사과를 설명합니다.", 1))
    scoped = check_answer_contract(_contract(_count("바나나 설명")), document)
    assert scoped.valid
    assert scoped.unchecked_semantic == ["required:sentence_scope:1"]
    assert not check_answer_contract(_contract(_count(None)), document).unchecked_semantic


@pytest.mark.parametrize("requirements", [
    [_count("설명", 2, 1)], [_count("설명", -1, 1)],
    [_count("설명"), _count("설명", 2, 2)], [_count(None), _count(None)],
    [_count(" ")], [_count("설명", True, 1)],
])
def test_invalid_or_duplicate_sentence_budgets_are_rejected(requirements):
    with pytest.raises(ValidationError):
        _contract(*requirements)


def test_sentence_count_requires_scoped_user_authority():
    payload = {"answer": {"sentence_counts": [_count("설명", evidence_ids=["e1"])]}, "evidence": [{
        "id": "e1", "turn_id": "current", "quote": "한 문장으로 설명해 줘.",
        "scope": "answer.sentence_counts", "interpretation": "instruction",
    }]}
    assert RequestContract.model_validate(payload).answer.sentence_counts[0].minimum == 1
    payload["evidence"][0]["interpretation"] = "quotation"
    with pytest.raises(ValidationError):
        RequestContract.model_validate(payload)
