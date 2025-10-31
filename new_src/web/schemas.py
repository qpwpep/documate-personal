from pydantic import BaseModel

# 입력 모델
class AgentRequest(BaseModel):
    query: str
    session_id: str # 클라이언트가 생성하여 전송해야 하는 고유 세션 ID (멀티턴 처리 시 사용)

# 출력 모델
class AgentResponse(BaseModel):
    response: str
    trace: str # 출력 확인용