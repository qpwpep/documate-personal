import os
import json
from typing import List, Any, Dict, Optional
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

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

    def run_agent_flow(self, user_input: str, session_id: Optional[str] = None) -> dict:
        
        # 상태 로드: CLI와 웹 모드 분기 처리
        if session_id:
            # 웹 모드: SESSION_CACHE에서 해당 ID의 상태를 로드 (없으면 []로 초기화)
            session_state = SESSION_CACHE.get(session_id, {})
            current_messages = session_state.get("messages", [])
            retriever = session_state.get("retriever")
        else:
            # CLI 모드: 객체 내부의 self.messages 상태를 사용
            current_messages = self.messages
            retriever = None
        
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
                "messages": current_messages, # 이전 대화 상태 전달
                "retriever": retriever,
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
            
            final_answer = ""
            file_path = ""

            # 메시지 목록을 역순으로 탐색
            for m in reversed(updated_messages):

                # 사용자 입력 메시지를 찾으면 더이상 이전 메시지를 찾지 않고 종료
                if isinstance(m, HumanMessage):
                    break
                
                # 1. 최종 AIMessage 추출 (가장 최근 답변)
                # 아직 최종 답변을 찾지 못했을 때만 실행
                if not final_answer and isinstance(m, AIMessage):
                    final_answer = m.content
                    
                # 2. 가장 최근의 save_text ToolMessage에서 파일 경로 찾기
                # 아직 파일 경로를 찾지 못했고, 해당 메시지가 save_text의 ToolMessage일 때만 실행
                elif not file_path and isinstance(m, ToolMessage) and m.name == "save_text":
                    try:
                        # 툴 실행 결과(JSON 문자열)를 파싱합니다.
                        tool_result_dict: Dict[str, Any] = json.loads(m.content)
                        extracted_path = tool_result_dict.get("file_path")
                        
                        # 파일 경로가 추출되었을 때, 실제 파일이 존재하는지 확인
                        if extracted_path and os.path.exists(extracted_path):
                            file_path = extracted_path
                        else:
                            # 경로가 추출되었으나 파일이 없으면 빈 문자열로 리셋
                            file_path = ""

                        if extracted_path:
                            file_path = extracted_path
                            
                    except json.JSONDecodeError:
                        continue
                
                # 3. 중단 조건: 필요한 두 정보(최종 답변, 파일 경로)를 모두 찾았다면 루프를 즉시 종료
                if final_answer and file_path:
                    break 

            return {"message": final_answer, "filepath": file_path, "response": response}
        
        except Exception as e:
            print(f"Agent 실행 중 오류 발생: {e}")
            return f"Agent 실행 중 오류가 발생했습니다: {e}"