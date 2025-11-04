import streamlit as st
import requests
import os
import uuid
from dotenv import load_dotenv

with st.sidebar:
    st.subheader("Slack DM 옵션")
    use_dm = st.checkbox("답변을 Slack DM으로도 보내기", value=False)
    slack_user_id = st.text_input("Slack User ID (U...)", value="")
    slack_email = st.text_input("Slack Email", value="")

# ========== tools를 참조하지 못하여 추가
import sys
from pathlib import Path
# 1. 현재 파일의 부모(web)의 부모(new_src)를 프로젝트 루트로 지정합니다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 2. 이 경로를 Python 모듈 탐색 경로에 추가합니다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from new_src.tools import DEFAULT_DOCS
# =================================

# 챗봇 세션이 시작될 때 고유 ID 생성 (탭이 새로 열릴 때마다 1번 실행)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    print(f"[REQ ID: {st.session_state.session_id[:8]}] - session start")

load_dotenv()
# FastAPI 서버의 주소 설정
FASTAPI_URL = os.environ.get("FASTAPI_URL")

# FastAPI Agent API 호출 함수
def get_agent_response(user_input: str):
    """
    FastAPI /agent 엔드포인트에 요청을 보내고,
    (옵션) Slack DM 전송을 위해 slack_user_id / slack_email을 함께 전달합니다.
    """
    endpoint = f"{FASTAPI_URL}/agent"

    # 기본 payload
    payload = {
        "query": user_input,
        "session_id": st.session_state.session_id,
    }

    # DM 옵션이 켜져 있을 때만 DM 관련 필드를 포함
    if use_dm:
        if slack_user_id:
            payload["slack_user_id"] = slack_user_id
        if slack_email:
            payload["slack_email"] = slack_email

    try:
        # ✅ payload를 일관되게 사용 (중복/재요청 제거)
        resp = requests.post(endpoint, json=payload, timeout=60)

        if resp.status_code == 200:
            data = resp.json()
            # FastAPI의 응답 스키마에 맞춰 안전하게 접근
            return data.get("response", ""), data.get("file_path")
        else:
            # 서버가 에러를 반환한 경우 메시지 표시
            return (f"Agent 호출 실패: 상태 코드 {resp.status_code}\n"
                    f"응답: {resp.text}"), None

    except requests.exceptions.Timeout:
        return "요청이 타임아웃되었습니다. 서버 상태를 확인해 주세요.", None
    except requests.exceptions.ConnectionError:
        return "FastAPI 서버에 연결할 수 없습니다. 서버(8000번 포트) 실행 여부를 확인해 주세요.", None
    except Exception as e:
        return f"요청 중 예기치 않은 오류가 발생했습니다: {e}", None


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
    st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요! 질문을 입력해주세요.", "file_path": ""}]


for message in st.session_state.messages:
    # 1. 채팅 메시지 출력 (아이콘은 여기서 한 번만 그려집니다)
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 
        
        # 2. 파일 다운로드 버튼 표시 (오직 'assistant' 메시지에 대해)
        file_path = message.get("file_path", "")
        
        if message["role"] == "assistant" and file_path and os.path.exists(file_path):
            
            filename = os.path.basename(file_path)
            download_url = f"{FASTAPI_URL}/download/{filename}"
            
            # UI 출력
            st.markdown("---")
            st.info(f"💾 **파일 저장 완료:** `{filename}`")

            st.markdown(
                f'<a href="{download_url}" target="_blank" download="{filename}">'
                f'<button style="background-color: #4CAF50; color: white; padding: 10px 24px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; width: 100%;">'
                f'⬇️ 파일 다운로드 ({filename})'
                f'</button></a>',
                unsafe_allow_html=True
            )
        
# ----------------------------------------------------
# 5. 사용자 입력 처리 및 세션 상태 업데이트
# ----------------------------------------------------
if prompt := st.chat_input("여기에 질문을 입력하세요..."):
    # 1. 사용자 메시지를 세션 상태에 추가
    st.session_state.messages.append({"role": "user", "content": prompt, "file_path": ""}) 

    # 2. 사용자가 입력한 메시지를 스피너가 돌기 전에 즉시 화면에 표시합니다.
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Agent 응답 생성
    with st.spinner("Agent가 생각 중입니다..."):
        # 응답 텍스트와 파일 경로를 받습니다.
        agent_response_content, agent_file_path = get_agent_response(prompt)
    
    # 4. 새로운 Assistant 메시지를 세션에 추가
    # 이 메시지에 파일 경로 데이터를 저장합니다.
    st.session_state.messages.append({
        "role": "assistant", 
        "content": agent_response_content, 
        "file_path": agent_file_path 
    })
    
    # 5. UI를 새로고침하여 새로 추가된 메시지와 버튼을 표시
    st.rerun()