import pytest

from src.core.answer_schema import AnswerDocument, export_answer_text, text_document
from src.core.llm_errors import LLMCallError
from src.core.request_contracts import (
    AnswerContract, ContentRequirement, ContractEvidence, FormatRequirement, RequestContract,
)
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_synthesis_validation import _hit, _state as retrieval_state
from tests.synthesis_fixtures import synthesis_excerpt_limits


class ModelBoundary:
    def __init__(self, document=None, error=None):
        self.document = document or text_document("첫 줄\n둘째 줄\n셋째 줄")
        self.error = error

    def invoke(self, _messages):
        if self.error:
            raise self.error
        return self.document.model_dump(mode="json")


def test_validation_exhaustion_cannot_replace_forbidden_generated_code_with_original_code():
    """금지된 코드 생성물을 검증에서 제거한 뒤 같은 금지의 원문 코드로 복구하지 않는다."""
    from src.core.contracts.debug import RetryState

    contract = RequestContract(
        answer=AnswerContract(format=(FormatRequirement(kind="code_block", mode="forbidden", evidence_ids=("r1",)),)),
        evidence=(ContractEvidence(id="r1", turn_id="current", quote="코드 없이", scope="answer.format.code_block", interpretation="negation"),),
    )
    state = retrieval_state([_hit("original_code()", source="upload")])
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract})
    state["retry"] = RetryState(max_retries=0, max_content_repairs=0)
    document = AnswerDocument.model_validate({"blocks": [{
        "type": "code", "content": {"text": "generated_code()", "basis": "example", "refs": []},
    }]})

    state.update(make_synthesize_node(ModelBoundary(document), excerpt_limits=synthesis_excerpt_limits())(state))
    state.update(make_post_synthesis_validation_node(False)(state))

    assert state["response"].kind == "failure"
    assert "generated_code" not in export_answer_text(state["response"].result)
    assert "original_code" not in export_answer_text(state["response"].result)


def test_source_fallback_does_not_assume_excerpts_satisfy_semantic_requirements():
    """원문 구조 검사만으로 비교 내용 충족을 증명할 수 없으면 실패 안내로 종료한다."""
    contract = RequestContract(
        answer=AnswerContract(content=(ContentRequirement(kind="comparison", mode="required", evidence_ids=("r1",)),)),
        evidence=(ContractEvidence(id="r1", turn_id="current", quote="비교해줘", scope="answer.content.comparison", interpretation="instruction"),),
    )
    state = retrieval_state([_hit("A source excerpt without any comparison.")])
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract})

    with pytest.raises(LLMCallError) as caught:
        make_synthesize_node(ModelBoundary(error=TimeoutError("generation failed")), excerpt_limits=synthesis_excerpt_limits())(state)
    assert caught.value.problem.code == "provider_unavailable"
    assert "response" not in state
    assert "A source excerpt" not in caught.value.problem.message
