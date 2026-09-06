from src.core.request_contracts import AnswerContract, infer_answer_contract


def test_retrieval_sources_do_not_impose_answer_layout():
    """검색 출처가 늘어도 사용자가 요구하지 않은 비교 형식을 강제하지 않는다."""
    assert infer_answer_contract("Explain official docs using uploaded code context.") == AnswerContract()


def test_code_example_request_preserves_required_product_behavior():
    """코드 예제 요청은 실제 코드 내용이 필요하다는 계약으로 전달된다."""
    assert infer_answer_contract("BeautifulSoup 샘플 코드 보여줘").code_example is True


def test_requested_comparison_and_checklist_are_independent_of_sources():
    """비교와 체크리스트는 출처 분리 섹션 대신 질문의 요구로 표현된다."""
    assert infer_answer_contract("Compare the two settings as a checklist.") == AnswerContract(
        comparison=True, checklist=True
    )


def test_ordered_steps_and_options_are_preserved():
    """단계와 옵션 요청은 답변 형식 및 내용 요구로 각각 보존된다."""
    assert infer_answer_contract("옵션을 단계별로 설명해줘") == AnswerContract(
        ordered_steps=True, options_summary=True
    )
