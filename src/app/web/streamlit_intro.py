from __future__ import annotations

from html import escape

import streamlit as st

from src.app.web.streamlit_state import get_conversation


_DOC_PROMPTS = [
    "pandas merge 사용법을 공식 문서 기준으로 설명해줘",
    "NumPy 배열 reshape 예제를 공식 문서 기준으로 알려줘",
    "FastAPI 라우터 구조를 간단히 설명해줘",
    "scikit-learn train_test_split 사용법을 정리해줘",
]
_UPLOAD_PROMPTS = [
    "첨부한 자료의 핵심 내용을 요약하고 근거를 보여줘",
    "첨부한 자료에서 중요한 용어를 설명해줘",
]



def render_intro(default_docs: dict[str, str], *, has_uploads: bool = False) -> None:
    st.markdown(
        """
        <section class="dm-intro">
            <h1>문서와 자료에 질문하세요</h1>
            <p>공식 문서로 사용법을 확인하거나, 입력창의 + 버튼으로 파일을 첨부하세요.<br>
            답변의 ‘근거’에서 사용한 원문과 위치를 확인할 수 있습니다.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    conversation = get_conversation()
    quick_prompts = ([_DOC_PROMPTS[0], _UPLOAD_PROMPTS[0], _DOC_PROMPTS[1], _UPLOAD_PROMPTS[1]]
                     if has_uploads else _DOC_PROMPTS)
    st.markdown("**질문 예시**")
    if conversation.draft.text:
        st.caption("작성 중인 질문을 먼저 보내거나 지운 뒤 다른 예시를 선택하세요.")
    else:
        st.caption("예시를 선택하면 입력창에 채워집니다. 확인하거나 수정한 뒤 전송하세요.")
    with st.container(key=conversation.widget_key("default_prompts")):
        _render_examples(quick_prompts[:2], start_index=0)
    if not has_uploads:
        # The browser reveals these supported callbacks as soon as the native
        # file picker has a selection, without rerunning or losing that picker.
        with st.container(key=conversation.widget_key("selected_upload_prompts")):
            _render_examples(_UPLOAD_PROMPTS, start_index=4)
    with st.expander("다른 질문 예시와 지원 문서", expanded=False):
        _render_examples(quick_prompts[2:], start_index=2)
        st.markdown("**지원하는 공식 문서**")
        docs_badges = "".join(f"<span>{escape(key)}</span>" for key in default_docs)
        st.markdown(f'<div class="dm-docs-row">{docs_badges}</div>', unsafe_allow_html=True)


def _render_examples(prompts: list[str], *, start_index: int) -> None:
    conversation = get_conversation()
    columns = st.columns(2)
    for offset, prompt in enumerate(prompts):
        with columns[offset]:
            st.button(prompt, key=conversation.widget_key(f"quick_prompt_{start_index + offset}"),
                      use_container_width=True, disabled=bool(conversation.draft.text),
                      on_click=_prefill_question, args=(prompt,))


def _prefill_question(prompt: str) -> None:
    conversation = get_conversation()
    # The browser can flush a just-typed draft in the same click as this button.
    # Check again in the callback rather than trusting its last disabled state.
    if (not conversation.draft.text and not conversation.is_busy
            and not conversation.is_read_only and conversation.pending_upload is None
            and not conversation.saved_prompt):
        conversation.draft.edit(prompt)
