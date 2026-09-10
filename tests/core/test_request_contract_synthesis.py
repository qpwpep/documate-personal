from langchain_core.messages import HumanMessage

from src.core.answer_schema import AnswerDocument, export_answer_text, finalize_answer, text_document
from src.core.contracts import PlannerState, ResponseState, RuntimeState
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import (
    AnswerContract, BoundAnswerReference, ContentRequirement, ContractEvidence,
    CopyAnswerBody, FormatRequirement, RequestContract, TransformAnswerBody,
)
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_synthesis_validation import _hit, _state as retrieval_state


class ModelBoundary:
    def __init__(self, document=None, error=None):
        self.document = document or text_document("첫 줄\n둘째 줄\n셋째 줄")
        self.error = error
        self.messages = []

    def invoke(self, messages):
        self.messages = messages
        if self.error:
            raise self.error
        return self.document.model_dump(mode="json")


def _state(contract, query="요청", previous=None):
    return {
        "runtime": RuntimeState(user_input=query, request_contract=contract, previous_response=previous),
        "planner": PlannerState(output=PlannerOutput(use_retrieval=False, tasks=[])),
        "messages": [HumanMessage(content=query)],
    }


def test_transform_uses_bound_document_and_returns_three_new_lines():
    """이전 답변 수정은 고정한 원문을 모델에 제공하고 새 세 줄 본문을 검증한다."""
    previous = finalize_answer(text_document("이전 첫 줄\n이전 둘째 줄\n이전 셋째 줄\n이전 넷째 줄"), [])
    contract = RequestContract(
        body=TransformAnswerBody(source=BoundAnswerReference(ref="previous", response_hash=previous.content_hash),
                                 instruction="원래 답변을 세 줄로 줄인다."),
        answer=AnswerContract(format=(FormatRequirement(kind="line_count", mode="required", value=3, evidence_ids=("r1",)),)),
        evidence=(ContractEvidence(id="r1", turn_id="current", quote="세 줄로 줄여", scope="answer.format.line_count", interpretation="instruction"),),
    )
    model = ModelBoundary()
    state = _state(contract, "방금 답변을 세 줄로 줄여 저장해줘", previous)

    state.update(make_synthesize_node(model)(state))
    state.update(make_post_synthesis_validation_node(False)(state))

    assert export_answer_text(state["response"].result) == "첫 줄\n둘째 줄\n셋째 줄"
    assert state["response"].kind == "answer"
    prompt = "\n".join(str(message.content) for message in model.messages)
    assert "이전 넷째 줄" in prompt
    assert previous.content_hash in prompt
    assert state["runtime"].request_contract == contract


def test_raw_words_do_not_add_requirements_absent_from_contract():
    """인용·부정·언급 속 단어는 확정 계약에 없는 코드나 저장 요구를 만들지 않는다."""
    model = ModelBoundary(text_document("요청한 설명입니다."))
    state = _state(RequestContract(), "‘Slack 저장 코드 예시 체크리스트’라는 문장을 설명해줘")

    state.update(make_synthesize_node(model)(state))
    state.update(make_post_synthesis_validation_node(False)(state))

    assert state["response"].kind == "answer"
    prompt = "\n".join(str(message.content) for message in model.messages)
    assert "Produce the complete content to save" not in prompt
    assert "Include concrete code in a code block" not in prompt
    assert "Use a list for the requested checklist" not in prompt


def test_reuse_with_new_forbidden_code_does_not_return_old_code():
    """이전 답변 그대로 재사용도 현재 계약의 코드 금지를 우회하지 않는다."""
    document = AnswerDocument.model_validate({"blocks": [{
        "type": "code", "language": "python", "content": {"text": "unsafe_old_code()", "basis": "example"},
    }]})
    previous = finalize_answer(document, [])
    contract = RequestContract(
        body=CopyAnswerBody(source=BoundAnswerReference(ref="previous", response_hash=previous.content_hash)),
        answer=AnswerContract(content=(ContentRequirement(kind="code_example", mode="forbidden", evidence_ids=("r1",)),)),
        evidence=(ContractEvidence(id="r1", turn_id="current", quote="코드 없이", scope="answer.content.code_example", interpretation="negation"),),
    )
    state = _state(contract, previous=previous)

    state.update(make_synthesize_node(ModelBoundary(error=RuntimeError("failed")))(state))
    state.update(make_post_synthesis_validation_node(False)(state))

    assert state["response"].kind == "failure"
    assert "unsafe_old_code" not in export_answer_text(state["response"].result)


def test_unavailable_contract_cannot_reuse_previous_response():
    """계약 해석 실패를 저장 키워드나 이전 답변으로 복구하지 않는다."""
    previous = finalize_answer(text_document("예전 전달 본문"), [])
    state = _state(None, "save this to Slack", previous)

    result = make_synthesize_node(ModelBoundary(error=AssertionError("must not generate")))(state)["response"]

    assert result.kind == "failure"
    assert "예전 전달 본문" not in export_answer_text(result.result)


def test_preference_alone_does_not_fail_validation():
    """선호 형식을 쓰지 않아도 필수 위반으로 재생성하지 않는다."""
    contract = RequestContract(
        answer=AnswerContract(format=(FormatRequirement(kind="table", mode="preferred", evidence_ids=("r1",)),)),
        evidence=(ContractEvidence(id="r1", turn_id="current", quote="표면 좋아", scope="answer.format.table", interpretation="instruction"),),
    )
    state = _state(contract)
    state["response"] = ResponseState(result=finalize_answer(text_document("설명 문단"), []), kind="draft",
                                      request_id=contract.request_id, contract_revision=contract.revision)

    updates = make_post_synthesis_validation_node(False)(state)

    assert updates["response"].kind == "answer"
    assert not updates["retry"].needs_retry


def test_exhausted_generation_cannot_expose_forbidden_original_code():
    """일반·compact 생성 실패 후에도 코드 금지를 원문 발췌로 우회하지 않는다."""
    contract = RequestContract(
        answer=AnswerContract(content=(ContentRequirement(kind="code_example", mode="forbidden", evidence_ids=("r1",)),)),
        evidence=(ContractEvidence(id="r1", turn_id="current", quote="코드 없이", scope="answer.content.code_example", interpretation="negation"),),
    )
    state = retrieval_state([_hit("forbidden_original_code()", source="upload")], query="코드 없이 설명해줘")
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract})

    state.update(make_synthesize_node(
        ModelBoundary(error=TimeoutError("timeout")), ModelBoundary(error=TimeoutError("timeout")),
    )(state))
    state.update(make_post_synthesis_validation_node(False)(state))

    assert state["response"].kind == "failure"
    assert "forbidden_original_code" not in export_answer_text(state["response"].result)
    assert state["response"].result.citations == []
    assert state["runtime"].request_contract == contract


def test_validation_exhaustion_cannot_replace_forbidden_generated_code_with_original_code():
    """금지된 코드 생성물을 검증에서 제거한 뒤 같은 금지의 원문 코드로 복구하지 않는다."""
    from src.core.contracts.debug import RetryState

    contract = RequestContract(
        answer=AnswerContract(format=(FormatRequirement(kind="code_block", mode="forbidden", evidence_ids=("r1",)),)),
        evidence=(ContractEvidence(id="r1", turn_id="current", quote="코드 없이", scope="answer.format.code_block", interpretation="negation"),),
    )
    state = retrieval_state([_hit("original_code()", source="upload")])
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract})
    state["retry"] = RetryState(max_retries=0)
    document = AnswerDocument.model_validate({"blocks": [{
        "type": "code", "content": {"text": "generated_code()", "basis": "example", "refs": []},
    }]})

    state.update(make_synthesize_node(ModelBoundary(document))(state))
    state.update(make_post_synthesis_validation_node(False)(state))

    assert state["response"].kind == "failure"
    assert "generated_code" not in export_answer_text(state["response"].result)
    assert "original_code" not in export_answer_text(state["response"].result)


def test_transform_keeps_the_original_citation_revision():
    """이전 본문 수정은 새 검색이 없어도 원래 인용 범위와 검증 요구를 보존한다."""
    evidence = _hit("The configured value is 3.").evidence
    previous = finalize_answer(text_document(evidence.excerpt, basis="source", refs=[evidence.id]), [evidence], retrieval_required=True)
    contract = RequestContract(body=TransformAnswerBody(
        source=BoundAnswerReference(ref="previous", response_hash=previous.content_hash), instruction="짧게 설명한다.",
    ))
    state = _state(contract, previous=previous)
    model = ModelBoundary(text_document("설정값은 3입니다.", basis="source", refs=[evidence.id]))

    state.update(make_synthesize_node(model)(state))
    state.update(make_post_synthesis_validation_node(False)(state))

    assert state["response"].kind == "answer"
    assert state["response"].result.citations == previous.citations
    assert state["response"].result.retrieval_required
    assert export_answer_text(state["response"].result) != export_answer_text(previous)


def test_source_fallback_does_not_assume_excerpts_satisfy_semantic_requirements():
    """원문 구조 검사만으로 비교 내용 충족을 증명할 수 없으면 실패 안내로 종료한다."""
    contract = RequestContract(
        answer=AnswerContract(content=(ContentRequirement(kind="comparison", mode="required", evidence_ids=("r1",)),)),
        evidence=(ContractEvidence(id="r1", turn_id="current", quote="비교해줘", scope="answer.content.comparison", interpretation="instruction"),),
    )
    state = retrieval_state([_hit("A source excerpt without any comparison.")])
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract})

    response = make_synthesize_node(ModelBoundary(error=RuntimeError("generation failed")))(state)["response"]

    assert response.kind == "failure"
    assert response.result.citations == []
    assert "A source excerpt" not in export_answer_text(response.result)


def test_forbidden_code_cannot_be_disguised_as_a_paragraph_excerpt():
    """코드 원문을 정확히 복사한 문단도 같은 코드 금지 조건을 적용받는다."""
    from src.core.contracts.debug import RetryState

    hit = _hit("forbidden_code()", source="upload")
    contract = RequestContract(
        answer=AnswerContract(content=(ContentRequirement(kind="code_example", mode="forbidden", evidence_ids=("r1",)),)),
        evidence=(ContractEvidence(id="r1", turn_id="current", quote="코드 없이", scope="answer.content.code_example", interpretation="negation"),),
    )
    state = retrieval_state([hit])
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract})
    state["retry"] = RetryState(max_retries=0)
    model = ModelBoundary(text_document(hit.evidence.excerpt, basis="excerpt", refs=[hit.evidence.id]))

    state.update(make_synthesize_node(model)(state))
    state.update(make_post_synthesis_validation_node(False)(state))

    assert state["response"].kind == "failure"
    assert "forbidden_code" not in export_answer_text(state["response"].result)


def test_code_prohibition_allows_a_plain_explanation_citing_code():
    """코드 근거를 인용하는 자연어 설명을 코드 원문과 혼동하여 금지하지 않는다."""
    hit = _hit("retries = 3", source="upload")
    contract = RequestContract(
        answer=AnswerContract(content=(ContentRequirement(kind="code_example", mode="forbidden", evidence_ids=("r1",)),)),
        evidence=(ContractEvidence(id="r1", turn_id="current", quote="코드 없이", scope="answer.content.code_example", interpretation="negation"),),
    )
    state = retrieval_state([hit])
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract})
    model = ModelBoundary(text_document("재시도 횟수는 세 번입니다.", basis="source", refs=[hit.evidence.id]))

    state.update(make_synthesize_node(model)(state))
    state.update(make_post_synthesis_validation_node(False)(state))

    assert state["response"].kind == "answer"
    assert export_answer_text(state["response"].result) == "재시도 횟수는 세 번입니다. [1]"


def test_reuse_checks_original_code_provenance_before_returning_a_paragraph():
    """이전 문단을 그대로 재사용할 때도 원문의 코드 종류를 확인하여 금지를 지킨다."""
    evidence = _hit("previous_code()", source="upload").evidence
    previous = finalize_answer(text_document(evidence.excerpt, basis="excerpt", refs=[evidence.id]), [evidence])
    contract = RequestContract(
        body=CopyAnswerBody(source=BoundAnswerReference(ref="previous", response_hash=previous.content_hash)),
        answer=AnswerContract(content=(ContentRequirement(kind="code_example", mode="forbidden", evidence_ids=("r1",)),)),
        evidence=(ContractEvidence(id="r1", turn_id="current", quote="코드 없이", scope="answer.content.code_example", interpretation="negation"),),
    )

    response = make_synthesize_node(ModelBoundary(error=AssertionError("reuse must not generate")))(
        _state(contract, previous=previous),
    )["response"]

    assert response.kind == "failure"
    assert "previous_code" not in export_answer_text(response.result)
