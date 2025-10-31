import streamlit as st
import requests
import os
import uuid
from dotenv import load_dotenv

# ========== tools를 참조하지 못하여 추가
import sys
from pathlib import Path
# 1. 현재 파일의 부모(web)의 부모(new_src)를 프로젝트 루트로 지정합니다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. 이 경로를 Python 모듈 탐색 경로에 추가합니다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import DEFAULT_DOCS
# =================================

# 챗봇 세션이 시작될 때 고유 ID 생성 (탭이 새로 열릴 때마다 1번 실행)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    print(f"[REQ ID: {st.session_state.session_id[:8]}] - session start")

load_dotenv()
# FastAPI 서버의 주소 설정
FASTAPI_URL = os.environ.get("FASTAPI_URL")

# FastAPI Agent API 호출 함수
def get_agent_response(user_input):
    """FastAPI의 /agent 엔드포인트에 요청을 보내고 응답을 받습니다."""
    endpoint = f"{FASTAPI_URL}/agent"
    print(f"debug >> user_input : {user_input}")
    try:
        # FastAPI 서버로 POST 요청 전송
        response = requests.post(
            endpoint,
            json={
                "query": user_input,
                "session_id": st.session_state.session_id,
                }, # AgentRequest
            timeout=60 # 응답 대기 시간을 60초로 설정
        )

        # 응답 상태 코드 확인
        if response.status_code == 200:
            print(f"debug >> response : {response.json().get("response", "FastAPI에서 응답을 받았습니다.")}")
            return response.json().get("response", "FastAPI에서 응답을 받았습니다.")
        else:
            return f"Agent 호출 실패: 상태 코드 {response.status_code}. 응답: {response.text}"

    except requests.exceptions.ConnectionError:
        return "FastAPI 서버에 연결할 수 없습니다. 서버(8000번 포트)가 실행 중인지 확인하세요."
    except Exception as e:
        return f"요청 중 예기치 않은 오류 발생: {e}"

# Streamlit 챗봇 UI 구성
st.set_page_config(page_title="Agent 챗봇 UI", layout="wide")
st.title("📚 Docs Agent 챗봇")

docs_list = [f"`{key}`" for key in list(DEFAULT_DOCS.keys())]
result_string = ", ".join(docs_list)

desc_markdown = f"""
<span style="font-size: 24px;"> 공식 문서 기반 답변을 제공합니다.</span>

<span style="font-size: 18px;"> 지원 문서 : {result_string}</span>
"""
st.markdown(desc_markdown, unsafe_allow_html=True)

# 세션 상태 초기화: 채팅 기록 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요! 질문을 입력해주세요."}]

# 기존 채팅 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("여기에 질문을 입력하세요..."):
    # 1) 사용자 메시지를 세션 상태에 추가하고 화면에 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Agent 응답 생성
    with st.spinner("Agent가 생각 중입니다..."):
        agent_response = get_agent_response(prompt)

    # 3) Agent 응답을 세션 상태에 추가하고 화면에 표시
    st.session_state.messages.append({"role": "assistant", "content": agent_response})
    with st.chat_message("assistant"):
        st.markdown(agent_response)