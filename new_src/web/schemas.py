from pydantic import BaseModel

# 입력 모델
class AgentRequest(BaseModel):
    query: str
    session_id: str # 클라이언트가 생성하여 전송해야 하는 고유 세션 ID (멀티턴 처리 시 사용)
    slack_user_id: str | None = None      # ← 추가 (선택)
    slack_email: str | None = None        # ← 추가 (선택)

# 출력 모델
class AgentResponse(BaseModel):
    response: str
    trace: str # 출력 확인용

    file_path: str | None = None # 파일이 생성된 경우, 해당 파일 경로
