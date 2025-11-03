from fastapi import FastAPI, Depends, Request
import uuid
import logging

from .schemas import AgentRequest, AgentResponse
from ..agent_manager import AgentFlowManager
from ..graph_builder import build_agent_graph


logger = logging.getLogger("uvicorn")
app = FastAPI()

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


# 의존성 함수 정의: 요청 시마다 새로운 Agent 객체 생성 (새로운 브라우저에서 요청 시 동일한 객체를 사용하지 않기 위함)
def get_new_agent_manager():
    """요청이 들어올 때마다 새로운 AgentFlowManager 객체를 생성 및 반환합니다."""
    return AgentFlowManager(app.state.langgraph)


@app.on_event("startup")
async def startup_event():
    """서버 시작 시점에 build_agent_graph를 실행하여 단 한 번 할당"""
    
    app.state.langgraph = build_agent_graph()
    logger.info(f"[startup] LangGraph 할당 완료. ID: {id(app.state.langgraph)}")


# http://localhost:8000/agent
@app.post("/agent", response_model=AgentResponse)
async def run_agent_api(
    request: Request,
    request_data: AgentRequest,
    agent_manager: AgentFlowManager = Depends(get_new_agent_manager) # 의존성 주입(DI)
):
    # 미들웨어에서 할당한 request 고유 ID를 가져옵니다.
    request_id = request.state.request_id[:8]
    
    user_query = request_data.query
    session_id = request_data.session_id

    # Agent 객체의 메모리 주소와 요청 ID를 로그에 출력
    logger.info(f"Session ID: {session_id[:8]} | [REQ ID: {request_id}] | Agent Object ID: {id(agent_manager)} | Query: '{user_query[:20]}...'")

    # 주입받은 agent_manager > run_agent_flow 메서드를 호출
    agent_answer = agent_manager.run_agent_flow(user_query, session_id=session_id)    
    
    return AgentResponse(
        response=agent_answer,
        trace=f"Session ID: {session_id}, Request ID: {request_id}, Agent ID: {id(agent_manager)}"
    )