from __future__ import annotations

import random
from html import escape

import streamlit as st

from src.app.web.streamlit_state import QUICK_PROMPTS_STATE_KEY


_QUICK_PROMPT_COUNT = 4
_QUICK_PROMPTS = [
    "pandas merge 사용법을 공식 문서 기준으로 설명해줘",
    "업로드한 노트북에서 pandas concat 예제를 찾아줘",
    "matplotlib pie 차트 옵션을 정리해줘",
    "방금 답변을 txt 파일로 저장해줘",
    "pandas groupby 결과를 보기 좋게 정리해줘",
    "업로드한 코드에서 에러 가능성이 있는 부분을 찾아줘",
    "NumPy 배열 reshape 예제를 공식 문서 기준으로 알려줘",
    "FastAPI 라우터 구조를 간단히 설명해줘",
    "scikit-learn train_test_split 사용법을 정리해줘",
    "PyTorch DataLoader 기본 사용법을 알려줘",
    "BeautifulSoup으로 특정 태그 찾는 예제를 보여줘",
    "방금 답변을 Slack으로 보내줘",
]



def render_intro(default_docs: dict[str, str]) -> str | None:
    docs_badges = "".join(f"<span>{escape(key)}</span>" for key in list(default_docs.keys()))
    st.markdown(
        f"""
        <section class="dm-intro">
            <div class="dm-intro-kicker">DocuMate</div>
            <h1>무엇을 도와드릴까요?</h1>
            <p>공식 문서와 업로드한 코드 파일을 함께 확인해 근거가 남는 답변을 만듭니다.</p>
            <div class="dm-docs-row">{docs_badges}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    quick_prompts = _get_quick_prompts_for_session()
    selected_prompt: str | None = None
    for row_start in range(0, len(quick_prompts), 2):
        columns = st.columns(2)
        for offset, prompt in enumerate(quick_prompts[row_start : row_start + 2]):
            index = row_start + offset
            with columns[offset]:
                if st.button(prompt, key=f"quick_prompt_{index}", use_container_width=True):
                    selected_prompt = prompt
    return selected_prompt



def _get_quick_prompts_for_session() -> list[str]:
    saved_prompts = st.session_state.get(QUICK_PROMPTS_STATE_KEY)
    if isinstance(saved_prompts, list) and saved_prompts:
        return [str(prompt) for prompt in saved_prompts[:_QUICK_PROMPT_COUNT]]

    prompt_count = min(_QUICK_PROMPT_COUNT, len(_QUICK_PROMPTS))
    selected_prompts = random.sample(_QUICK_PROMPTS, prompt_count)
    st.session_state[QUICK_PROMPTS_STATE_KEY] = selected_prompts
    return selected_prompts
