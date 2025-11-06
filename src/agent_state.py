from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

class AgentState(TypedDict, total=False):
    """LangGraph가 유지할 전체 상태 정의"""
    messages: Annotated[list[AnyMessage], add_messages] # 전체 대화 이력
    user_input: str                                     # 현재 사용자 입력 (멀티턴 입력 전용)
    final_answer: Optional[str]                         # 모델이 생성한 최종 응답
    search_needed: Optional[bool]                       # 검색 판단 결과 (툴 선택용)
    search_result: Optional[str]                        # 외부 검색 결과 저장용
    retriever: Optional[Any]                            # 임시 Chroma 리트리버 저장용