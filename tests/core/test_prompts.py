import os
import unittest

import pytest

from src.core.prompts import needs_search
from src.core.contracts.boundary.graph import build_graph_state_input
from src.infra.llm import build_llm_registry
from src.infra.settings import get_settings
from src.runtime.nodes.actions import is_action_only_request
from src.runtime.nodes.planner import make_planner_node


# Semantic cases are checked against the real provider, separately from local
# tests of the execution policy. No keyword parser supplies their expected routes.
LIVE_SOURCE_CASES = [
    {'id': 'known_01', 'query': 'Explain how pathlib handles local files according to the official docs.', 'expected_routes': ['docs']},
    {'id': 'known_02', 'query': '로컬 파일 처리의 주의점을 Python 공식 문서로 설명해줘.', 'expected_routes': ['docs']},
    {'id': 'known_03', 'query': 'FastAPI 파일 업로드 API를 공식 문서 기준으로 설명해줘.', 'expected_routes': ['docs']},
    {'id': 'known_04', 'query': 'Explain the difference between .py and .ipynb files using official docs.', 'expected_routes': ['docs']},
    {'id': 'known_05', 'query': '업로드 파일은 사용하지 말고 pandas merge를 공식 문서만으로 설명해줘.', 'expected_routes': ['docs']},
    {'id': 'known_06', 'query': 'Can you search local files for pandas merge?', 'expected_routes': ['upload']},
    {'id': 'known_07', 'query': '내가 올린 전처리 코드에서 인코딩 설정값을 찾아줘.', 'expected_routes': ['upload']},
    {'id': 'known_08', 'query': 'Find API calls in the uploaded notebook.', 'expected_routes': ['upload']},
    {'id': 'en_01', 'query': 'Could you explain how pandas assigns column names when reading a CSV file, using the official pandas documentation and a made-up example?', 'expected_routes': ['docs']},
    {'id': 'en_02', 'query': 'Could you explain which column names the CSV-loading code in the .ipynb notebook I uploaded in this session assigns to its DataFrame? Use the code and results in that notebook.', 'expected_routes': ['upload']},
    {'id': 'en_03', 'query': 'We are planning a new document search service in Python. Suggest an indexing design using official documentation or technical references; there is no project file to inspect.', 'expected_routes': ['docs']},
    {'id': 'en_04', 'query': 'I attached a .ipynb notebook containing the design notes for our Python document search service in this conversation. What indexing design do those notes specify?', 'expected_routes': ['upload']},
    {'id': 'en_05', 'query': 'Please compare the exception-handling examples in the Python tutorial notebook (.ipynb) I uploaded here with the official Python documentation, and identify any differences.', 'expected_routes': ['docs', 'upload']},
    {'id': 'en_06', 'query': 'Please summarize the exception-handling examples in the Python tutorial notebook (.ipynb) I uploaded here. Use that notebook alone and exclude official documentation.', 'expected_routes': ['upload']},
    {'id': 'en_07', 'query': 'How should a new FastAPI project handle request validation? Base your answer only on official documentation and leave out my attachments.', 'expected_routes': ['docs']},
    {'id': 'en_08', 'query': 'How does the FastAPI application in the .py file I attached to this session handle request validation? Base your answer only on that file and leave out official documentation.', 'expected_routes': ['upload']},
    {'id': 'ko_01', 'query': 'Python에서 JSON의 null과 빈 문자열을 읽으면 각각 어떤 값이 되나요? 공식 Python 문서를 근거로 간단한 예제를 들어 설명해 주세요.', 'expected_routes': ['docs']},
    {'id': 'ko_02', 'query': '이번 대화에 첨부한 .ipynb 노트북의 JSON 데이터 예제에서 null인 필드와 빈 문자열인 필드를 각각 찾아 주세요.', 'expected_routes': ['upload']},
    {'id': 'ko_03', 'query': '앞으로 만들 Python 결제 API에서 멱등성 키를 어떻게 설계하면 좋을지 공식 문서나 기술 레퍼런스를 근거로 알려줘. 현재 올려 둔 자료는 참고하지 마.', 'expected_routes': ['docs']},
    {'id': 'ko_04', 'query': '이번 세션에 Python 결제 API 코드를 담은 .py 파일을 올려 뒀어. 그 코드에서 멱등성 키를 어떻게 처리하는지 찾아줘.', 'expected_routes': ['upload']},
    {'id': 'ko_05', 'query': '이번 대화에 올린 .py 파일의 shutil.copy 사용 방식이 Python 공식 문서의 설명과 일치하는지 두 자료를 대조해 주실래요?', 'expected_routes': ['docs', 'upload']},
    {'id': 'ko_06', 'query': '이번 대화에 올린 .py 파일에서 shutil.copy가 어떤 경로를 복사하는지 알려 주실래요? 공식 문서는 제외하고 그 코드에 적힌 내용만 확인해 주세요.', 'expected_routes': ['upload']},
    {'id': 'ko_07', 'query': 'Python으로 PDF의 포함 글꼴을 조사하는 방법이 궁금해. 이번 세션의 첨부파일은 열지 말고 PyMuPDF 공식 문서를 바탕으로 설명해 줘.', 'expected_routes': ['docs']},
    {'id': 'ko_08', 'query': '이번 세션에 첨부한 .ipynb 노트북에 PDF 글꼴 분석 결과가 출력돼 있어. 그 출력에 어떤 글꼴이 포함 글꼴로 표시됐는지 알려줘. 일반적인 기술 문서 설명은 제외하고 노트북의 결과만 확인해 줘.', 'expected_routes': ['upload']},
    {'id': 'en_docs_correction', 'query': "Ignore my earlier request to open cleanup.py; I meant Python's official documentation: does Path.unlink(missing_ok=True) suppress permission errors as well as errors for a missing file?", 'expected_routes': ['docs']},
    {'id': 'ko_docs_negation', 'query': '올린 train.py에서 예외를 잡는지는 관심 없어요. 파이썬 공식 문서를 근거로, finally 안의 return이 앞서 발생한 예외에 어떤 영향을 주는지 설명해 주세요.', 'expected_routes': ['docs']},
    {'id': 'en_upload_reference', 'query': "In the analysis.ipynb I uploaded, does the final chart use the filtered table or the original one? Trace it back through that notebook's cells; 'it' means the table passed to the chart, and library documentation is unnecessary.", 'expected_routes': ['upload']},
    {'id': 'ko_upload_correction', 'query': '공식 문서에서 기본값을 찾아달라는 말은 취소할게요. 이 세션에 올린 fetch.py에서 timeout 값을 생략하는 호출이 있는지, 있다면 그 호출이 어느 함수 안에 있는지 확인해 주세요.', 'expected_routes': ['upload']},
    {'id': 'en_both_clause_context', 'query': "I don't need a general tutorial on pandas merging. Compare the join in my uploaded reconcile.py with the pandas official documentation and explain whether the code preserves rows whose join keys are null.", 'expected_routes': ['docs', 'upload']},
    {'id': 'ko_both_reference', 'query': '올린 cache_demo.ipynb에서 캐시를 비우는 셀을 먼저 찾아주세요. 그 셀이 앞서 만든 캐시 전체를 비우는지는 파이썬 공식 문서의 functools 캐시 API 설명과 실제 셀 코드를 대조해서 판단해 주세요.', 'expected_routes': ['docs', 'upload']},
    {'id': 'en_neither_negation', 'query': "Use only this pasted snippet, `retries = 3\\nprint(retries)`, and rename retries to attempts. Don't open any uploaded files or consult official documentation; those sources are outside this request.", 'expected_routes': []},
    {'id': 'ko_neither_reference', 'query': "아래에 붙인 문장 '데이터를 읽은 뒤 빈 행을 제거한다'를 영어로 번역해 주세요. 여기서 '그 내용'은 이 문장만 뜻하며, 업로드한 파일 내용과 공식 문서는 둘 다 참조하지 마세요.", 'expected_routes': []},
    {'id': 'general_technical_terms_do_not_require_a_file_0', 'query': 'Explain Python local variables from official docs.', 'expected_routes': ['docs']},
    {'id': 'general_technical_terms_do_not_require_a_file_1', 'query': 'Show a FastAPI implementation example from official docs.', 'expected_routes': ['docs']},
    {'id': 'general_technical_terms_do_not_require_a_file_2', 'query': 'Explain pandas best practice for performance.', 'expected_routes': ['docs']},
    {'id': 'general_technical_terms_do_not_require_a_file_3', 'query': '로컬 파일을 읽는 방법을 Python 공식 문서로 설명해줘.', 'expected_routes': ['docs']},
    {'id': 'general_technical_terms_do_not_require_a_file_4', 'query': 'Explain how to read local files using pathlib from official docs.', 'expected_routes': ['docs']},
    {'id': 'general_technical_terms_do_not_require_a_file_6', 'query': 'Explain Python context managers in practice using official docs.', 'expected_routes': ['docs']},
    {'id': 'general_technical_terms_do_not_require_a_file_7', 'query': 'Show pathlib examples for reading local files from official docs.', 'expected_routes': ['docs']},
    {'id': 'general_technical_terms_do_not_require_a_file_8', 'query': 'Explain how to search local files using pathlib from official docs.', 'expected_routes': ['docs']},
    {'id': 'general_file_operations_use_only_docs_1', 'query': 'Explain how to import local code according to the Python official docs.', 'expected_routes': ['docs']},
    {'id': 'upload_api_explanations_use_only_docs_1', 'query': 'Explain the FastAPI file upload API using official docs.', 'expected_routes': ['docs']},
    {'id': 'file_format_explanations_use_only_docs_1', 'query': '.py와 .ipynb 파일 형식의 차이를 공식 문서로 설명해줘.', 'expected_routes': ['docs']},
    {'id': 'future_project_guidance_uses_only_docs_0', 'query': '앞으로 만들 프로젝트의 구조를 FastAPI 공식 문서 기준으로 설명해줘.', 'expected_routes': ['docs']},
    {'id': 'future_project_guidance_uses_only_docs_1', 'query': 'Explain how to structure a project I plan to build using FastAPI official docs.', 'expected_routes': ['docs']},
    {'id': 'user_file_queries_require_upload_0', 'query': '내가 올린 파일에서 pandas merge 사용을 찾아줘.', 'expected_routes': ['upload']},
    {'id': 'user_file_queries_require_upload_1', 'query': 'Find pandas merge in my uploaded notebook.', 'expected_routes': ['upload']},
    {'id': 'user_file_queries_require_upload_2', 'query': 'Find pandas merge in my file.', 'expected_routes': ['upload']},
    {'id': 'user_file_comparisons_require_both_sources_0', 'query': '내가 올린 파일의 pandas merge 사용을 공식 문서와 비교해줘.', 'expected_routes': ['docs', 'upload']},
    {'id': 'user_file_comparisons_require_both_sources_1', 'query': 'Compare pandas merge in my uploaded notebook with official docs.', 'expected_routes': ['docs', 'upload']},
    {'id': 'uploaded_material_topics_do_not_add_external_docs_0', 'query': 'Find the retry policy our team recorded in the uploaded HTTP client specification.', 'expected_routes': ['upload']},
    {'id': 'uploaded_material_topics_do_not_add_external_docs_1', 'query': '이 업로드 문서에서 제너레이터 표현식이 어떻게 동작한다고 설명하는지 찾아 주세요.', 'expected_routes': ['upload']},
    {'id': 'uploaded_material_topics_do_not_add_external_docs_2', 'query': '업로드한 기획서에서 새 서비스의 문서 검색 기능을 어떻게 설계하기로 했는지 찾아 주세요.', 'expected_routes': ['upload']},
    {'id': 'uploaded_material_topics_do_not_add_external_docs_3', 'query': '파일을 업로드하는 기능의 장단점은 설명하지 말고 내가 업로드한 명세서에서 업로드 API의 요청 형식만 찾아 주세요.', 'expected_routes': ['upload']},
    {'id': 'uploaded_guide_comparison_keeps_both_sources_0', 'query': 'Would you please compare the JSON Lines file format described in my uploaded guide with official documentation?', 'expected_routes': ['docs', 'upload']},
    {'id': 'inline_code_explanation_does_not_require_an_upload_0', 'query': '이 코드에서 제너레이터 표현식이 어떻게 동작하는지 설명해 주세요: squares = (n * n for n in range(5))', 'expected_routes': [], 'optional_routes': ['docs']},
    {'id': 'file_lookup_survives_exclusions_of_other_content_0', 'query': 'Inspect my code and ignore blank lines.', 'expected_routes': ['upload']},
    {'id': 'file_lookup_survives_exclusions_of_other_content_1', 'query': 'Review my code without creating new files.', 'expected_routes': ['upload']},
    {'id': 'file_lookup_survives_exclusions_of_other_content_2', 'query': 'Do not use my files; inspect the uploaded notebook instead.', 'expected_routes': ['upload']},
    {'id': 'later_docs_only_instruction_replaces_file_request_0', 'query': 'Inspect my uploaded file. Use only official docs instead.', 'expected_routes': ['docs']},
    {'id': 'retained_0', 'query': 'Find pandas merge in local notebook examples.', 'expected_routes': ['upload']},
    {'id': 'retained_1', 'query': 'Find pandas merge in local vector index examples.', 'expected_routes': ['upload']},
    {'id': 'retained_2', 'query': 'Find pandas merge in my project code.', 'expected_routes': ['upload']},
    {'id': 'retained_3', 'query': '로컬 노트북 예제에서 pandas merge를 찾아줘.', 'expected_routes': ['upload']},
    {'id': 'retained_4', 'query': '프로젝트의 코드에서 pandas merge를 찾아줘.', 'expected_routes': ['upload']},
    {'id': 'retained_5', 'query': '기존 실습 자료에서 pandas merge를 찾아줘.', 'expected_routes': ['upload']},
    {'id': 'retained_6', 'query': 'Search locally for notebook examples.', 'expected_routes': ['upload']},
    {'id': 'retained_7', 'query': 'Find pandas merge in my local files.', 'expected_routes': ['upload']},
    {'id': 'retained_8', 'query': 'Find pandas merge in these local files.', 'expected_routes': ['upload']},
    {'id': 'retained_9', 'query': 'Search local files for pandas merge.', 'expected_routes': ['upload']},
    {'id': 'retained_10', 'query': 'Please find pandas merge in local files.', 'expected_routes': ['upload']},
    {'id': 'retained_11', 'query': 'Find pandas merge in my practice notebook.', 'expected_routes': ['upload']},
    {'id': 'retained_12', 'query': 'Explain how pathlib handles local files and compare with my project code.', 'expected_routes': ['upload'], 'optional_routes': ['docs']},
    {'id': 'retained_13', 'query': 'Could you find pandas merge in local files?', 'expected_routes': ['upload']},
    {'id': 'retained_14', 'query': 'Explain how Python imports local code according to the official docs.', 'expected_routes': ['docs']},
    {'id': 'retained_15', 'query': 'Explain how FastAPI handles file uploads according to the official docs.', 'expected_routes': ['docs']},
    {'id': 'retained_16', 'query': '내 다음 프로젝트에서 pandas로 CSV 파일을 읽으려고 해. read_csv의 dtype 옵션을 공식 문서 기준으로 설명해 줘.', 'expected_routes': ['docs']},
    {'id': 'retained_17', 'query': 'Do not search my local files; explain pathlib using only official docs.', 'expected_routes': ['docs']},
    {'id': 'retained_18', 'query': '업로드한 노트북은 참고하지 말고, scikit-learn 공식 문서에서 train_test_split의 stratify 사용 조건을 찾아 줘.', 'expected_routes': ['docs']},
    {'id': 'retained_19', 'query': 'Use official docs only to explain Python file uploads.', 'expected_routes': ['docs']},
]


class PromptsTest(unittest.TestCase):
    def test_needs_search_matches_library_explainer_request(self) -> None:
        self.assertTrue(needs_search("pandas에 대해 알려줘"))

    def test_needs_search_matches_korean_technical_request(self) -> None:
        self.assertTrue(needs_search("판다스의 성능 최적화를 알려줘"))

    def test_saving_an_answer_to_a_local_file_is_action_only(self) -> None:
        self.assertTrue(is_action_only_request("최종 답변을 로컬 파일로 저장해줘"))

    def test_project_lookup_before_saving_is_not_action_only(self) -> None:
        self.assertFalse(is_action_only_request("내 프로젝트 코드에서 merge 사용을 찾아서 저장해줘"))


@pytest.fixture(scope="module")
def live_planner():
    settings = get_settings().model_copy(update={"verbose": False})
    assert settings.openai_api_key, "OPENAI_API_KEY must be configured for the live planner check"
    return make_planner_node(build_llm_registry(settings).llm_planner, verbose=False)


@pytest.mark.skipif(os.getenv("LIVE_TEST") != "true", reason="Opt-in live provider check: LIVE_TEST=true")
@pytest.mark.parametrize("case", LIVE_SOURCE_CASES, ids=lambda case: case["id"])
def test_live_source_selection(case, live_planner):
    # A retriever handle is availability context only; no retrieval tool runs.
    state = build_graph_state_input(user_input=case["query"], messages=[], retriever=object())
    result = live_planner(state)
    planner = result["planner"]
    assert planner.status == "llm"
    # Unrequested uploads are forbidden; docs may be optional when the user
    # requests a technical explanation without specifying its evidence source.
    required = set(case["expected_routes"])
    allowed = required | set(case.get("optional_routes", []))
    actual = {task.route for task in planner.output.tasks}
    assert required <= actual <= allowed, (case["query"], actual, required, allowed)
    assert planner.guided_followup is None
    assert result["debug"].llm_calls, "A real provider response must be recorded"
