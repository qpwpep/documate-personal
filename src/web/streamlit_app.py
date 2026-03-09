import streamlit as st
import requests
import os
import sys
import uuid
import logging

from src.domain_docs import DEFAULT_DOCS
from src.logging_utils import configure_logging, log_event
from src.runtime_encoding import ensure_utf8_stdio
from src.settings import get_settings

ensure_utf8_stdio()
configure_logging()
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _warn_if_utf8_mode_disabled_once() -> None:
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


_warn_if_utf8_mode_disabled_once()

with st.sidebar:
    st.subheader("Slack 전송")
    slack_user_id = st.text_input("User ID (Uxxxxx)", value="")
    slack_email = st.text_input("Email (optional)", value="")
    slack_channel_id = st.text_input("Channel ID (C/G/Dxxxxx, optional)", value="")

from pathlib import Path

# 챗봇 세션이 시작될 때 고유 ID 생성 (탭이 새로 열릴 때마다 1번 실행)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    log_event(logger, logging.INFO, "streamlit_session_start", session_id=st.session_state.session_id[:8])

# ============ 파일 업로드 관련 초기 설정
UPLOADS_DIR = Path("uploads")

if 'uploaded_file_name' not in st.session_state:
    st.session_state['uploaded_file_name'] = None

SESSION_PATH = UPLOADS_DIR / st.session_state['session_id']
SESSION_PATH.mkdir(parents=True, exist_ok=True)
# =================================

SETTINGS = get_settings()
FASTAPI_URL = SETTINGS.fastapi_url

# FastAPI Agent API 호출 함수
def get_agent_response(user_input: str):
    """
    FastAPI /agent 엔드포인트에 요청을 보내고,
    (옵션) Slack DM 전송을 위해 slack_user_id / slack_email을 함께 전달합니다.
    """
    endpoint = f"{FASTAPI_URL}/agent"
    try:
        payload = {
            "query": user_input,
            "session_id": st.session_state.session_id,
        }

        # ✅ 값이 있으면 항상 payload에 포함 (전송은 모델이 요청 받을 때만)
        if slack_user_id:
            payload["slack_user_id"] = slack_user_id
        if slack_email:
            payload["slack_email"] = slack_email
        if slack_channel_id:
            payload["slack_channel_id"] = slack_channel_id

        # ✅ 업로드 경로 전달 (기존 순서 버그 수정: path 만든 뒤 넣기)
        if st.session_state.get("uploaded_file_name"):
            path = SESSION_PATH / st.session_state["uploaded_file_name"]
            payload["upload_file_path"] = path.as_posix()

        resp = requests.post(endpoint, json=payload, timeout=60)

        if resp.status_code == 200:
            data = resp.json()
            # FastAPI의 응답 스키마에 맞춰 안전하게 접근
            response_payload = data.get("response") or {}
            if isinstance(response_payload, dict):
                answer = str(response_payload.get("answer", "") or "")
                evidence = response_payload.get("evidence")
                evidence_items = evidence if isinstance(evidence, list) else []
            else:
                answer = str(response_payload)
                evidence_items = []
            return answer, data.get("file_path"), evidence_items
        else:
            # 서버가 에러를 반환한 경우 메시지 표시
            return (f"Agent 호출 실패: 상태 코드 {resp.status_code}\n"
                    f"응답: {resp.text}"), None, []

    except requests.exceptions.Timeout:
        return "요청이 타임아웃되었습니다. 서버 상태를 확인해 주세요.", None, []
    except requests.exceptions.ConnectionError:
        return "FastAPI 서버에 연결할 수 없습니다. 서버(8000번 포트) 실행 여부를 확인해 주세요.", None, []
    except Exception as e:
        return f"요청 중 예기치 않은 오류가 발생했습니다: {e}", None, []


# Streamlit 챗봇 UI 구성
st.set_page_config(page_title="Agent 챗봇 UI", layout="wide")
st.title("📚 DocuMate: 공식 문서를 기반으로 학습을 돕는 AI 챗봇")

docs_list = [f"`{key}`" for key in list(DEFAULT_DOCS.keys())]
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
        unsafe_allow_html=True
    )

# ----------------------------------------------------
# 파일 업로드
# ----------------------------------------------------
# Streamlit 업로드 핸들러
def handle_upload(uploaded_file):
    """파일 저장(업로드), 세션 상태 업데이트를 처리합니다."""
    
    if uploaded_file.name == st.session_state['uploaded_file_name']:
        # st.info("같은 파일이 이미 업로드되어 있습니다. RAG를 다시 구축하지 않습니다.")
        return

    # 세션 폴더에 파일 저장
    file_path_on_disk = SESSION_PATH / uploaded_file.name
    try:
        # 이전에 업로드된 파일이 있다면 삭제 (단일 파일 유지)
        if st.session_state['uploaded_file_name']:
            old_path = SESSION_PATH / st.session_state['uploaded_file_name']
            if old_path.exists():
                os.remove(old_path)

        # 새 파일 저장
        with open(file_path_on_disk, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state['uploaded_file_name'] = uploaded_file.name
        
    except ValueError as ve:
        st.session_state['uploaded_file_name'] = None
        st.error(f"파일 업로드 실패 (내용 오류): {ve}")
    except Exception as e:
        st.session_state['uploaded_file_name'] = None
        st.error(f"파일 업로드 실패 : {e}")

# ----------------------------------------------------
# 채팅 UI
# ----------------------------------------------------
# 세션 상태 초기화: 채팅 기록 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "안녕하세요! 질문을 입력해주세요.", "file_path": "", "evidence": []}
    ]


for message in st.session_state.messages:
    # 1. 채팅 메시지 출력 (아이콘은 여기서 한 번만 그려집니다)
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 
        evidence_items = message.get("evidence") or []
        if message["role"] == "assistant" and evidence_items:
            with st.expander("근거 보기"):
                for item in evidence_items:
                    if not isinstance(item, dict):
                        continue
                    kind = str(item.get("kind", "") or "").strip()
                    source = str(item.get("url_or_path", "") or "").strip()
                    title = str(item.get("title", "") or "").strip()
                    if title:
                        st.markdown(f"- `{kind}`: **{title}** ({source})")
                    else:
                        st.markdown(f"- `{kind}`: {source}")
        
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
    st.session_state.messages.append({"role": "user", "content": prompt, "file_path": "", "evidence": []})

    # 2. 사용자가 입력한 메시지를 스피너가 돌기 전에 즉시 화면에 표시합니다.
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Agent 응답 생성
    with st.spinner("Agent가 생각 중입니다..."):
        # 응답 텍스트와 파일 경로를 받습니다.
        agent_response_content, agent_file_path, agent_evidence = get_agent_response(prompt)
    
    # 4. 새로운 Assistant 메시지를 세션에 추가
    # 이 메시지에 파일 경로 데이터를 저장합니다.
    st.session_state.messages.append({
        "role": "assistant", 
        "content": agent_response_content, 
        "file_path": agent_file_path,
        "evidence": agent_evidence,
    })
    
    # 5. UI를 새로고침하여 새로 추가된 메시지와 버튼을 표시
    st.rerun()

# 업로드 버튼 위젯 생성
uploaded_file = st.file_uploader(
        label="파일 업로드 (.py, .ipynb 등 챗봇에게 질문할 때 사용할 파일을 업로드 하세요.)",
        type=['ipynb', 'py'],
        width=450,
    )

# 조건문으로 연결 (파일이 업로드되었을 때만 함수 실행)
if uploaded_file is not None:
    
    # 현재 세션 상태를 확인하여 중복 실행 방지
    if uploaded_file.name != st.session_state.get('uploaded_file_name'):
        
        # 파일 객체를 인수로 전달하며 RAG 초기화 함수 호출
        handle_upload(uploaded_file)

elif "uploaded_file_name" in st.session_state:
    # 파일 삭제 처리
    file_name = st.session_state["uploaded_file_name"]
    if file_name:
        try:
            old_path = SESSION_PATH / st.session_state['uploaded_file_name']
            if old_path.exists():
                os.remove(old_path)
                # st.info(f"이전 파일 '{st.session_state['uploaded_file_name']}'을(를) 삭제했습니다.")
                
        except FileNotFoundError:
            pass
    del st.session_state["uploaded_file_name"]
