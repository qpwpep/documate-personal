from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from functools import lru_cache

import streamlit as st

from src.logging_utils import log_event


logger = logging.getLogger(__name__)


@dataclass
class SidebarInputs:
    slack_user_id: str
    slack_email: str
    slack_channel_id: str


def configure_page() -> None:
    st.set_page_config(page_title="Agent 챗봇 UI", layout="wide")
    st.title("📚 DocuMate: 공식 문서를 기반으로 학습을 돕는 AI 챗봇")


@lru_cache(maxsize=1)
def warn_if_utf8_mode_disabled_once() -> None:
    if sys.flags.utf8_mode == 1:
        return

    log_event(
        logger,
        logging.WARNING,
        "utf8_mode_disabled",
        suggested_command=(
            "uv run python -X utf8 -m streamlit run src/web/streamlit_app.py --server.port 8501"
        ),
    )


def render_sidebar() -> SidebarInputs:
    with st.sidebar:
        st.subheader("Slack 전송")
        return SidebarInputs(
            slack_user_id=st.text_input("User ID (Uxxxxx)", value=""),
            slack_email=st.text_input("Email (optional)", value=""),
            slack_channel_id=st.text_input("Channel ID (C/G/Dxxxxx, optional)", value=""),
        )


def render_intro(default_docs: dict[str, str]) -> None:
    docs_list = [f"`{key}`" for key in list(default_docs.keys())]
    result_string = ", ".join(docs_list)

    st.markdown(
        f"""
        <div style="height: 20px;"></div>

        #### **파이썬 오픈 소스 라이브러리의 활용법을 학습해보세요!**
        - **공식 문서 내용을 기반**으로 정확한 정보를 얻으실 수 있습니다.<br>
          (지원 문서 : {result_string})
        - 로컬 노트북 (AI 부트캠프 경진대회 baseline code) / 직접 파일을 업로드 하여 **활용 사례를 확인**하실 수 있습니다.
        - 결과를 **txt 파일로 저장**하거나 **슬랙에 공유**하실 수 있습니다.

        ---

        ##### ✅ 빠른 시작 예시
          `pandas merge 사용법 알려줘. 공식 문서 기준으로 설명해줘.`&nbsp;&nbsp;&nbsp;`matplotlib에서 pie 차트 옵션 정리해줘.`
          `이전 노트북에서 matplotlib histplot을 어떻게 썼는지 예제 코드 보여줘.`&nbsp;&nbsp;&nbsp;`업로드한 .ipynb 안에 있는 pandas concat 예제를 찾아줘.`
          `pandas concat 기본 사용법을 설명하고, 내 노트북에서 실제로 사용한 예제도 함께 보여줘.`
          <div style="height: 20px;"></div>

        ##### 📎 파일 업로드
        - `.py`, `.ipynb` 파일을 업로드 하여 해당 내용을 기반으로 서칭하기 : `이 파일에서 쓰인 pandas concat() 함수 예제를 찾아줘`
        (업로드 파일이 크면 **핵심 코드/셀만** 올리는 것이 더 빠릅니다.)
        <div style="height: 20px;"></div>

        ##### 💾 결과 txt 파일로 저장 및 Slack 공유
        - 결과를 txt 파일로 저장하기 : `이 답변을 txt로 저장해줘`&nbsp;&nbsp;&nbsp;`방금 결과를 파일로 저장해줘`
        - 결과를 Slack으로 보내기 : `이 답변을 Slack으로 보내줘`&nbsp;&nbsp;&nbsp;`이번 결과를 팀 채널에 공유해줘`

        ---
        """,
        unsafe_allow_html=True,
    )
