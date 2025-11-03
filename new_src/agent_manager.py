from typing import List, Any, Dict, Optional
from langchain_core.messages import AIMessage

# 전역 상태 캐시 (웹 멀티턴용)
# Key: session_id (str), Value: messages (List[Any])
# FastAPI 서버 프로세스가 실행되는 동안만 상태를 유지하는 인메모리 캐시
SESSION_CACHE: Dict[str, List[Any]] = {}

# Agent Logic 클래스 정의 (build_graph, 멀티턴 messages 관리)
class AgentFlowManager:
    """
    LangGraph 기반 Agent의 상태(messages)와 실행 로직을 관리하는 클래스
    """
    def __init__(self, graph):
        # graph는 FastAPI startup에서 생성된 객체를 계속 전달 받는다.
        self.graph = graph
        # 멀티턴 상태 저장을 위한 변수
        self.messages: List[Any] = []

    def run_agent_flow(self, user_input: str, session_id: Optional[str] = None) -> str:
        
        # 상태 로드: CLI와 웹 모드 분기 처리
        if session_id:
            # 웹 모드: SESSION_CACHE에서 해당 ID의 상태를 로드 (없으면 []로 초기화)
            current_messages = SESSION_CACHE.get(session_id, [])
        else:
            # CLI 모드: 객체 내부의 self.messages 상태를 사용
            current_messages = self.messages
        
        # 종료 명령어 처리 (CLI와 웹 모두에서 세션 초기화 기능)
        if user_input.lower() in {"exit", "종료", "quit", "q"}:
            if session_id:
                SESSION_CACHE[session_id] = [] # 웹 세션 초기화
            else:
                self.messages = [] # CLI 세션 초기화
            return "챗봇 세션이 초기화되었습니다. 다시 시작합니다."

        try:
            # LangGraph에 멀티턴 상태 전달
            state = {
                "user_input": user_input,
                "messages": current_messages  # 이전 대화 상태 전달
            }

            # LangGraph 실행
            response = self.graph.invoke(state)
        
            # 결과 메시지 업데이트
            updated_messages = response["messages"]

            # 상태 저장: CLI와 웹 모드 분기 처리
            if session_id:
                # 웹 모드: 갱신된 상태를 캐시에 저장
                SESSION_CACHE[session_id] = updated_messages
            else:
                # CLI 모드: 객체 내부에 저장
                self.messages = updated_messages
            
            # 최종 AI 메시지를 찾아서 반환 (CLI의 마지막 print 부분)
            final_answer = ""
            for m in reversed(updated_messages):
                if isinstance(m, AIMessage):
                    final_answer = m.content
                    break
            
            return final_answer
        
        except Exception as e:
            print(f"Agent 실행 중 오류 발생: {e}")
            return f"Agent 실행 중 오류가 발생했습니다: {e}"