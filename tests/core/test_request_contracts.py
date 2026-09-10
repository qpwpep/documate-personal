import pytest
from pydantic import ValidationError

from src.core.answer_schema import AnswerDocument, finalize_answer, text_document
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import build_evidence
from src.core.request_contracts import (
    ActionContract, ActionRequest, AnswerContract, BoundAnswerReference, CopyAnswerBody, ContentRequirement,
    ContractEvidence, FormatRequirement, RequestContract, check_answer_contract,
    resolve_body_response, validate_contract_evidence,
)


def evidence(quote="코드 예시 없이 설명해줘", scope="answer.content.code_example"):
    return ContractEvidence(id="e1", turn_id="current", quote=quote, scope=scope, interpretation="negation")


def test_forbidden_code_accepts_plain_explanation_and_rejects_code():
    """코드 금지는 설명을 허용하고 실제 코드 블록이 포함된 답변을 거부한다."""
    contract = AnswerContract(content=(ContentRequirement(kind="code_example", mode="forbidden", evidence_ids=("e1",)),))
    code = AnswerDocument.model_validate({"blocks": [{"type": "code", "content": {"text": "print(1)", "basis": "example"}}]})
    assert check_answer_contract(contract, text_document("개념을 설명합니다.")).valid
    assert check_answer_contract(contract, code).forbidden_present == ["code_example"]


def test_forbidden_code_also_rejects_a_markdown_fence_in_a_paragraph():
    """단락에 코드 블록을 숨겨도 동일한 금지 조건이 적용된다."""
    contract = AnswerContract(format=(FormatRequirement(kind="code_block", mode="forbidden", evidence_ids=("e1",)),))
    assert check_answer_contract(contract, text_document("```python\nprint(1)\n```")).forbidden_present == ["code_block"]


def test_preferred_layout_never_becomes_missing_required_content():
    """표 형식 선호를 충족하지 않아도 정상 설명을 재생성 대상으로 삼지 않는다."""
    contract = AnswerContract(format=(FormatRequirement(kind="table", mode="preferred", evidence_ids=("e1",)),))
    assert check_answer_contract(contract, text_document("설명입니다.")).valid


def test_everyday_example_does_not_require_a_code_block():
    """일상생활 사례 요구는 예시 본문으로 충족되며 코드를 강제하지 않는다."""
    contract = AnswerContract(content=(ContentRequirement(kind="example", mode="required", evidence_ids=("e1",)),))
    assert check_answer_contract(contract, text_document("책장 정리에 비유할 수 있습니다.", basis="example")).valid


def test_required_three_lines_checks_the_displayed_body():
    """세 줄 요구는 실제 본문 줄 수를 검사해 수정되지 않은 네 줄을 거부한다."""
    contract = AnswerContract(format=(FormatRequirement(kind="line_count", mode="required", value=3, evidence_ids=("e1",)),))
    assert check_answer_contract(contract, text_document("하나\n둘\n셋")).valid
    assert check_answer_contract(contract, text_document("하나\n둘\n셋\n넷")).missing_required == ["line_count:3"]


def test_required_and_forbidden_same_feature_is_rejected():
    """정정이 해소되지 않은 모순된 요구를 확정 계약으로 받아들이지 않는다."""
    with pytest.raises(ValidationError):
        AnswerContract(content=(
            ContentRequirement(kind="code_example", mode="required", evidence_ids=("e1",)),
            ContentRequirement(kind="code_example", mode="forbidden", evidence_ids=("e2",)),
        ))


def test_requested_action_without_supporting_user_evidence_is_rejected():
    """근거가 없는 저장 요청을 실행 가능한 계약으로 승격하지 않는다."""
    with pytest.raises(ValidationError):
        RequestContract(actions=ActionContract(save_text=ActionRequest(intent="requested")))


def test_contract_roundtrip_preserves_negation_and_scope_and_is_immutable():
    """경계 직렬화와 이후 단계가 확정된 금지 의사와 근거 범위를 바꾸지 않는다."""
    item = evidence("저장하지 마", "actions.save_text")
    contract = RequestContract(actions=ActionContract(save_text=ActionRequest(intent="forbidden", evidence_ids=("e1",))), evidence=(item,))
    parsed = RequestContract.model_validate_json(contract.model_dump_json())
    assert parsed == contract
    with pytest.raises(ValidationError):
        parsed.actions.save_text.intent = "requested"


def test_grounding_checks_only_actual_user_utterances():
    """모델이 발화에 없는 문구를 실행 판단의 근거로 만들면 거부한다."""
    contract = RequestContract(evidence=(evidence("저장해줘", "actions.save_text"),))
    assert validate_contract_evidence(contract, {"current": "저장해줘"}) == []
    assert validate_contract_evidence(contract, {"current": "Slack API 설명해줘"})


def test_quoted_mention_cannot_be_the_only_authority_for_requested_action():
    """인용이나 단순 언급만을 근거로 실행하는 계약은 거부한다."""
    item = evidence("저장해줘", "actions.save_text").model_copy(update={"interpretation": "quotation"})
    with pytest.raises(ValidationError):
        RequestContract(actions=ActionContract(save_text=ActionRequest(intent="requested", evidence_ids=("e1",))), evidence=(item,))


def test_reuse_resolves_the_exact_response_revision():
    """지정한 이전 본문이 달라졌으면 다른 답변을 대신 전달하지 않는다."""
    previous = finalize_answer(text_document("원래 본문"), [])
    contract = RequestContract(body=CopyAnswerBody(source=BoundAnswerReference(ref="previous", response_hash=previous.content_hash)))
    assert resolve_body_response(contract, previous_response=previous) == previous
    changed = finalize_answer(text_document("대상은 어디인가요?"), [])
    assert resolve_body_response(contract, previous_response=changed) is None


def test_semantic_comparison_is_not_proved_by_having_a_table():
    """비교의 의미 충족을 단순 형식 검사로 통과했다고 주장하지 않는다."""
    contract = AnswerContract(content=(ContentRequirement(kind="comparison", mode="required", evidence_ids=("e1",)),))
    check = check_answer_contract(contract, text_document("설명"))
    assert check.unchecked_semantic == ["required:comparison"]


def test_code_source_cannot_bypass_prohibition_by_using_a_paragraph():
    """코드 원문을 그대로 담은 문단도 코드 금지를 우회하지 못한다."""
    snapshot = build_snapshot(source_uri="uploads/code.py", title="code", media_type="text/x-python", source_type="upload",
                              content="forbidden_code()", parser="test", parser_version="1")
    source = build_evidence(snapshot=snapshot, element=DocumentElement(element_id="code", kind="code", text="forbidden_code()"))
    contract = AnswerContract(content=(ContentRequirement(kind="code_example", mode="forbidden", evidence_ids=("e1",)),))
    quoted_code = text_document(source.excerpt, basis="excerpt", refs=[source.id])
    explanation = text_document("이 부분은 함수를 호출합니다.", basis="source", refs=[source.id])
    assert check_answer_contract(contract, quoted_code, evidence=[source]).forbidden_present == ["code_example"]
    assert check_answer_contract(contract, explanation, evidence=[source]).valid


def test_body_instruction_does_not_authorize_a_slack_action():
    """본문 설명 범위에 한정된 지시는 전송 요청의 근거로 사용할 수 없다."""
    item = evidence("Explain this", "body.instruction").model_copy(update={"interpretation": "instruction"})
    with pytest.raises(ValidationError):
        RequestContract(actions=ActionContract(slack_notify=ActionRequest(intent="requested", evidence_ids=("e1",))), evidence=(item,))


def test_quoted_content_does_not_become_a_required_answer_feature():
    """번역 대상인 인용문을 필수 답변 형식으로 승격하지 않는다."""
    item = evidence("code example").model_copy(update={"interpretation": "quotation"})
    with pytest.raises(ValidationError):
        RequestContract(answer=AnswerContract(content=(ContentRequirement(kind="code_example", mode="required", evidence_ids=("e1",)),)), evidence=(item,))


@pytest.mark.parametrize("body", [
    {"operation": "reuse", "source": "current"},
    {"operation": "generate", "source": "previous"},
    {"operation": "transform", "source": "previous", "instruction": ""},
])
def test_incompatible_body_operation_and_source_cannot_be_confirmed(body):
    """원본이나 수정 내용이 없는 작업을 확정된 본문 작업으로 받아들이지 않는다."""
    with pytest.raises(ValidationError):
        RequestContract(body=body)


def test_content_and_format_cannot_require_and_prohibit_the_same_code():
    """내용과 형식 필드로 나누어 기록해도 상충하는 코드 조건을 거부한다."""
    with pytest.raises(ValidationError):
        AnswerContract(content=(ContentRequirement(kind="code_example", mode="required", evidence_ids=("e1",)),),
                       format=(FormatRequirement(kind="code_block", mode="forbidden", evidence_ids=("e2",)),))


def test_cancellation_with_unresolved_intent_is_not_ready_even_without_a_missing_slot():
    """An incomplete canonical boundary cannot promote an ambiguous cancellation."""
    contract = RequestContract(relation="cancel", target_request_id="pending-request",
                               actions=ActionContract(slack_notify=ActionRequest(intent="unresolved")))
    assert not contract.can_cancel_pending()
    assert contract.status == "unresolved"
