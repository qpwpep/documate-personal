import os
import uuid
import logging

from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import FileResponse
from slack_sdk.web import WebClient
from slack_sdk.errors import SlackApiError

from .schemas import AgentRequest, AgentResponse
from ..agent_manager import AgentFlowManager
from ..util.util import get_save_text_output_dir

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_DEFAULT_DM_EMAIL = os.getenv("SLACK_DEFAULT_DM_EMAIL")
SLACK_DEFAULT_USER_ID = os.getenv("SLACK_DEFAULT_USER_ID")

slack_client = WebClient(token=SLACK_BOT_TOKEN) if SLACK_BOT_TOKEN else None

logger = logging.getLogger("uvicorn")
app = FastAPI()

def _resolve_user_id(user_id: str | None, email: str | None) -> str | None:
    """우선순위: 요청값(user_id/email) → .env → None"""
    if user_id:
        return user_id
    if email and slack_client:
        try:
            r = slack_client.users_lookupByEmail(email=email)
            return r["user"]["id"]
        except SlackApiError as e:
            logger.error(f"users.lookupByEmail failed: {e}")
    # .env fallback
    if SLACK_DEFAULT_USER_ID:
        return SLACK_DEFAULT_USER_ID
    if SLACK_DEFAULT_DM_EMAIL and slack_client:
        try:
            r = slack_client.users_lookupByEmail(email=SLACK_DEFAULT_DM_EMAIL)
            return r["user"]["id"]
        except SlackApiError as e:
            logger.error(f"fallback lookupByEmail failed: {e}")
    return None

def _open_dm_channel(user_id: str) -> str | None:
    """
    Slack DM 채널(Dxxxx)을 연 뒤 채널 ID를 반환합니다.
    """
    if not slack_client:
        return None
    try:
        # Slack Web API: conversations.open(users=Uxxxx)
        r = slack_client.conversations_open(users=user_id)
        return r["channel"]["id"]  # Dxxxx...
    except SlackApiError as e:
        logger.error(f"conversations.open failed for {user_id}: {e.response.get('error')}")
        return None


def send_slack_message(
    text: str,
    user_id: str | None = None,
    email: str | None = None,
    channel_id: str | None = None,
    target: str = "auto",  # auto / dm / channel / group
) -> None:
    """
    Slack 메시지 전송 (DM/채널/그룹 안전 판별)
    - DM: 반드시 Dxxxx 채널로 전송 (user Uxxxx → conversations.open → Dxxxx)
    - Channel: Cxxxx (public), Gxxxx (private)
    """
    if not slack_client:
        logger.warning("⚠️ SLACK_BOT_TOKEN 미설정: 메시지 전송 스킵")
        return

    resolved_id = None
    target_type = "Unknown"

    # 0) 명시 채널 우선 (C/G/D 모두 허용)
    if channel_id:
        resolved_id = channel_id
        if resolved_id.startswith("D"):
            target_type = "DM"
        elif resolved_id.startswith("C"):
            target_type = "Public Channel"
        elif resolved_id.startswith("G"):
            target_type = "Private Channel"
        else:
            target_type = "Unknown Channel"

    # 1) user 또는 email 이면 → 반드시 DM 채널 열기
    if not resolved_id and (user_id or email):
        uid = _resolve_user_id(user_id, email)  # Uxxxx (또는 이미 Uxxxx일 수 있음)
        if uid and uid.startswith("U"):
            dm_id = _open_dm_channel(uid)  # Dxxxx
            if dm_id:
                resolved_id = dm_id
                target_type = "DM"

    # 2) .env fallback (기본 DM 대상)
    if not resolved_id:
        uid = _resolve_user_id(None, None)
        if uid and uid.startswith("U"):
            dm_id = _open_dm_channel(uid)
            if dm_id:
                resolved_id = dm_id
                target_type = "DM"

    if not resolved_id:
        logger.warning("⚠️ Slack 대상 ID를 찾지 못해 메시지 전송 스킵")
        return

    # 3) target 강제 옵션 (채널/그룹 강제 시 유효성 체크)
    if target == "channel" and not resolved_id.startswith(("C", "G")):
        logger.warning("⚠️ target='channel'이지만 DM ID가 선택됨. C/G 채널 ID를 channel_id로 직접 넘기세요.")
    if target == "group" and not resolved_id.startswith("G"):
        logger.warning("⚠️ target='group'이지만 선택된 ID가 G가 아님. G 채널 ID를 channel_id로 넘기세요.")
    if target == "dm" and not resolved_id.startswith("D"):
        logger.warning("⚠️ target='dm'이지만 DM 채널이 아님. user_id/email로 다시 호출하세요.")
        # 가능하면 강제로 DM 전환 시도 (C/G/D가 아닌 경우)
        # 여기서는 경고만 하고 진행하지 않음.

    # 4) 실제 전송
    try:
        slack_client.chat_postMessage(channel=resolved_id, text=text)
        logger.info(f"✅ Slack {target_type} 전송 성공 → {resolved_id}")
    except SlackApiError as e:
        logger.error(f"❌ Slack {target_type} 전송 실패 ({resolved_id}): {e.response.get('error')}")


# 인메모리 세션 저장소 (Global Cache) 정의
# Key: session_id (str), Value: AgentFlowManager 인스턴스
active_agents: dict[str, 'AgentFlowManager'] = {}

@app.get("/")
async def root():
    return {"message": "Hello World"}


# [미들웨어] 모든 http 요청(ex. 쿼리 요청)에 고유 ID 할당 (로그 추적용)
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    logger.info(f"[REQ ID: {request_id[:8]}] - Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"[REQ ID: {request_id[:8]}] - Finished request with status {response.status_code}")
    return response


#AgentFlowManager 인스턴스를 가져오거나 생성
def _get_or_create_agent(session_id: str) -> AgentFlowManager:
    """세션 ID에 해당하는 AgentManager 인스턴스를 반환합니다. 없으면 새로 생성합니다."""
    
    if session_id not in active_agents:
        # AgentManager가 없으면 새로 생성 후 저장
        agent = AgentFlowManager()
        active_agents[session_id] = agent
        logger.info(f"✅ AgentManager 저장됨: {session_id[:8]}")
    else:
        # AgentManager가 이미 있으면 기존 인스턴스를 가져옴
        agent = active_agents[session_id]
        logger.info(f"♻️ 기존 AgentManager 재사용: {session_id[:8]}")
        
    return agent


# http://localhost:8000/agent
@app.post("/agent", response_model=AgentResponse)
async def run_agent_api(
    request: Request,
    request_data: AgentRequest,
):
    # 미들웨어에서 할당한 request 고유 ID를 가져옵니다.
    request_id = request.state.request_id[:8]
    
    user_query = request_data.query
    session_id = request_data.session_id

    # session_id 기준으로 하나의 agent_manager를 생성하여 사용
    agent_manager = _get_or_create_agent(session_id)

    # Agent 객체의 메모리 주소와 요청 ID를 로그에 출력
    logger.info(f"Session ID: {session_id[:8]} | [REQ ID: {request_id}] | Agent Object ID: {id(agent_manager)} | Query: '{user_query[:20]}...'")

    # agent_manager > run_agent_flow 메서드를 호출
    agent_answer = agent_manager.run_agent_flow(user_query)
    answer = agent_answer.get("message")
    file_path = agent_answer.get("filepath", "")

    logger.info(f"agent_answer : {agent_answer}")
    
    logger.info(f"answer : {answer}")
    logger.info(f"filepath : {file_path}")

    response = AgentResponse(
        response=answer,
        trace=f"Session ID: {session_id}, Request ID: {request_id}, Agent ID: {id(agent_manager)}",
        file_path=file_path,
    )
    return response


@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    지정된 파일 이름으로 서버에 저장된 파일을 다운로드합니다.
    """
    # 보안: 파일 경로 조작 공격(Path Traversal) 방지를 위해 
    # os.path.join을 사용하여 파일 이름을 안전하게 처리합니다.
    output_dir = get_save_text_output_dir()
    file_path = os.path.join(output_dir, filename)

    # 1. 파일 경로가 의도된 디렉토리 내부에 있는지 최종적으로 확인 (보안 강화)
    if not os.path.realpath(file_path).startswith(os.path.realpath(output_dir)):
         raise HTTPException(status_code=403, detail="Forbidden: Invalid file path")

    # 2. 파일 존재 여부 확인
    if not os.path.exists(file_path):
        # 파일이 없으면 404 에러 반환
        print(f"[ERROR] Download request failed: File not found at {file_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    # 3. FileResponse로 파일 스트리밍
    # FileResponse는 Content-Disposition 헤더를 자동으로 추가하여 
    # 브라우저가 파일을 다운로드하도록 유도합니다.
    # media_type은 .txt 파일이므로 'text/plain'을 사용합니다.
    return FileResponse(
        path=file_path,
        filename=filename, # 사용자에게 표시될 파일 이름
        media_type='text/plain'
    )