from pathlib import Path

import pytest

from src.core.answer_schema import AnswerDocument, export_answer_text, text_document
from src.core.llm_errors import LLMCallError
from src.core.request_contracts import (
    AnswerContract, ContentRequirement, ContractEvidence, LayoutRequirement, RequestContract, check_answer_contract,
)
from src.infra.chunking import chunk_python_text
from src.core.evidence import build_evidence
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.helpers import make_decisions_client, mock_chat_model
from tests.core.test_synthesis_validation import _hit, _state as retrieval_state
from tests.synthesis_fixtures import prepare_synthesis_state


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
        answer=AnswerContract(format=(LayoutRequirement(kind="code_block", mode="forbidden", evidence_ids=("r1",)),)),
        evidence=(ContractEvidence(id="r1", turn_id="current", quote="코드 없이", scope="answer.format.code_block", interpretation="negation"),),
    )
    state = retrieval_state([_hit("original_code()", source="upload")])
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract})
    state["retry"] = RetryState(max_retries=0, max_content_repairs=0)
    document = AnswerDocument.model_validate({"blocks": [{
        "type": "code", "content": {"text": "generated_code()", "basis": "example", "refs": []},
    }]})

    prepare_synthesis_state(state)
    state.update(make_synthesize_node(mock_chat_model(ModelBoundary(document).invoke))(state))
    state.update(make_post_synthesis_validation_node(False, decisions_client=make_decisions_client())(state))

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

    prepare_synthesis_state(state)
    with pytest.raises(LLMCallError) as caught:
        make_synthesize_node(mock_chat_model(ModelBoundary(error=TimeoutError("generation failed")).invoke))(state)
    assert caught.value.problem.code == "provider_unavailable"
    assert "response" not in state
    assert "A source excerpt" not in caught.value.problem.message


def _native_calculation_source():
    path = Path(__file__).resolve().parents[2] / "data/benchmarks/fixtures/uploads/sample_data_analysis.py"
    raw = path.read_bytes()
    upload = chunk_python_text(path=str(path), text=raw.decode("utf-8"), source_content=raw,
                               chunk_size=2000, chunk_overlap=0)
    return upload.hydrate(upload.chunks[0])


def test_cited_native_code_line_in_paragraph_satisfies_code_without_creating_code_block():
    evidence = _native_calculation_source()
    code = 'grouped = all_sales.groupby("region", as_index=False)["amount"].sum()'
    document = text_document(f"계산식은 `{code}`입니다.", basis="source", refs=[evidence.id])
    contract = AnswerContract(
        content=(ContentRequirement(kind="code", mode="required"),),
        format=(LayoutRequirement(kind="code_block", mode="forbidden"),),
    )

    check = check_answer_contract(contract, document, evidence=[evidence])

    assert check.missing_required == []
    assert check.forbidden_present == []
    assert check.unchecked_semantic == []
    forbidden = AnswerContract(content=(ContentRequirement(kind="code", mode="forbidden"),))
    assert check_answer_contract(forbidden, document, evidence=[evidence]).forbidden_present == ["code"]


@pytest.mark.parametrize("quoted", ["sample_data_analysis.py", "지역별 합계", "amount", "all_sales.groupby"])
def test_filename_label_or_partial_identifier_does_not_prove_native_code_content(quoted):
    evidence = _native_calculation_source()
    document = text_document(f"`{quoted}`를 확인하세요.", basis="source", refs=[evidence.id])
    contract = AnswerContract(content=(ContentRequirement(kind="code", mode="required"),))

    check = check_answer_contract(contract, document, evidence=[evidence])

    assert check.unchecked_semantic == ["required:code"]


def test_inline_code_cannot_borrow_another_units_source():
    evidence = _native_calculation_source()
    code = 'grouped = all_sales.groupby("region", as_index=False)["amount"].sum()'
    document = AnswerDocument.model_validate({"blocks": [{"type": "paragraph", "content": [
        {"text": f"`{code}`", "basis": "source", "refs": []},
        {"text": "sample_data_analysis.py", "basis": "source", "refs": [evidence.id]},
    ]}]})
    contract = AnswerContract(content=(ContentRequirement(kind="code", mode="required"),))

    assert check_answer_contract(contract, document, evidence=[evidence]).unchecked_semantic == ["required:code"]


def test_inline_identifier_cannot_borrow_a_complete_line_outside_selected_range():
    text = "amount\nx = amount.sum()\n"
    upload = chunk_python_text(path="selected.py", text=text, source_content=text.encode(),
                               chunk_size=2000, chunk_overlap=0)
    original = upload.hydrate(upload.chunks[0])
    start = text.index("amount.sum")
    evidence = build_evidence(snapshot=original.snapshot, element=original.element,
                              start=start, end=start + len("amount"))
    document = text_document("`amount`를 사용합니다.", basis="source", refs=[evidence.id])
    contract = AnswerContract(content=(ContentRequirement(kind="code", mode="required"),))

    assert evidence.excerpt == "amount"
    assert check_answer_contract(contract, document, evidence=[evidence]).unchecked_semantic == ["required:code"]


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_selected_complete_code_line_can_exclude_indentation_and_line_ending(newline):
    code = "value = 1"
    text = f"if True:{newline}    {code}{newline}"
    upload = chunk_python_text(path="selected.py", text=text, source_content=text.encode(),
                               chunk_size=2000, chunk_overlap=0)
    original = upload.hydrate(upload.chunks[0])
    start = text.index(code)
    evidence = build_evidence(snapshot=original.snapshot, element=original.element,
                              start=start, end=start + len(code))
    document = text_document(f"`{code}`를 사용합니다.", basis="source", refs=[evidence.id])
    contract = AnswerContract(content=(ContentRequirement(kind="code", mode="required"),))

    assert check_answer_contract(contract, document, evidence=[evidence]).missing_required == []
    assert check_answer_contract(contract, document, evidence=[evidence]).unchecked_semantic == []
