import os
import uuid
import logging

from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import FileResponse

from .schemas import AgentRequest, AgentResponse
from ..agent_manager import AgentFlowManager
from ..graph_builder import build_agent_graph

from ..util.util import get_save_text_output_dir

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

    logger.info(f"agent_answer : {agent_answer}")

    answer = agent_answer.get("message")
    file_path = agent_answer.get("filepath", "")
    
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