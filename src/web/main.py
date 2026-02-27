import logging
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from langchain_core.messages import SystemMessage

from .schemas import AgentRequest, AgentResponse
from ..agent_manager import AgentFlowManager
from ..util.util import get_save_text_output_dir

logger = logging.getLogger("uvicorn")
app = FastAPI()
ALLOWED_UPLOAD_SUFFIXES = {".py", ".ipynb"}

# In-memory per-session agent cache.
active_agents: dict[str, AgentFlowManager] = {}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    logger.info(f"[REQ ID: {request_id[:8]}] - Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"[REQ ID: {request_id[:8]}] - Finished request with status {response.status_code}")
    return response


def _get_or_create_agent(session_id: str) -> AgentFlowManager:
    """Return an AgentFlowManager for the session, creating one if needed."""
    if session_id not in active_agents:
        agent = AgentFlowManager()
        active_agents[session_id] = agent
        logger.info(f"AgentManager created: {session_id[:8]}")
    else:
        agent = active_agents[session_id]
        logger.info(f"AgentManager reused: {session_id[:8]}")

    return agent


def _validate_upload_file_path(upload_file_path: str | None, session_id: str) -> str | None:
    """
    Validate client-provided upload path and return a normalized absolute path.
    The file must exist under uploads/<session_id>/ and be one of allowed types.
    """
    if not upload_file_path:
        return None

    try:
        session_upload_dir = (Path("uploads") / session_id).resolve()
        candidate_path = Path(upload_file_path).expanduser().resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid upload_file_path") from exc

    if session_upload_dir not in candidate_path.parents:
        raise HTTPException(status_code=400, detail="Invalid upload file location")

    if candidate_path.suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported upload file type")

    if not candidate_path.is_file():
        raise HTTPException(status_code=400, detail="Upload file not found")

    return str(candidate_path)


@app.post("/agent", response_model=AgentResponse)
async def run_agent_api(
    request: Request,
    request_data: AgentRequest,
):
    request_id = request.state.request_id[:8]

    user_query = request_data.query
    session_id = request_data.session_id
    upload_file_path = _validate_upload_file_path(request_data.upload_file_path, session_id)
    logger.info(f"[upload_file_path]: {upload_file_path}")

    agent_manager = _get_or_create_agent(session_id)

    # Inject Slack destination hints into the message stream (no auto-send here).
    slack_hints = []
    if request_data.slack_channel_id:
        slack_hints.append(f"channel_id={request_data.slack_channel_id}")
    if request_data.slack_user_id:
        slack_hints.append(f"user_id={request_data.slack_user_id}")
    if request_data.slack_email:
        slack_hints.append(f"email={request_data.slack_email}")

    if slack_hints:
        hint_text = (
            "[Slack Destinations]\n"
            + "\n".join(slack_hints)
            + "\n(When the user asks to send to Slack, call slack_notify with these values.)"
        )
        agent_manager.messages.append(SystemMessage(content=hint_text))

    logger.info(
        f"Session ID: {session_id[:8]} | [REQ ID: {request_id}] | "
        f"Agent Object ID: {id(agent_manager)} | Query: '{user_query[:20]}...'"
    )

    agent_answer = agent_manager.run_agent_flow(user_query, upload_file_path)

    answer = agent_answer.get("message")
    file_path = agent_answer.get("filepath", "")

    logger.info(f"agent_answer: {agent_answer}")
    logger.info(f"answer: {answer}")
    logger.info(f"filepath: {file_path}")

    response = AgentResponse(
        response=answer,
        trace=f"Session ID: {session_id}, Request ID: {request_id}, Agent ID: {id(agent_manager)}",
        file_path=file_path,
    )
    return response


@app.get("/download/{filename}")
async def download_file(filename: str):
    output_dir = get_save_text_output_dir()
    file_path = os.path.join(output_dir, filename)

    if not os.path.realpath(file_path).startswith(os.path.realpath(output_dir)):
        raise HTTPException(status_code=403, detail="Forbidden: Invalid file path")

    if not os.path.exists(file_path):
        logger.error(f"Download request failed: File not found at {file_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/plain",
    )
