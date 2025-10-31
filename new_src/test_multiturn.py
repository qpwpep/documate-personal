import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from .make_graph import build_graph
from .llm import VERBOSE

def run_multiturn_test():
    load_dotenv()

    graph = build_graph()

    print("\n🧪 멀티턴 테스트 시작...\n")

    test_inputs = [
        "파이썬 클래스가 뭐야?",
        "그럼 상속은?",
        "super()는 어떻게 써?",
        "예외 처리는?",
        "마지막으로 데코레이터는?"
    ]

    # chat_history 상태 유지
    state = {"messages": []}

    for idx, user_input in enumerate(test_inputs):
        print(f"\n🧵 {idx+1}번째 질문: {user_input}")
    
        # 👇 LangGraph가 요구하는 입력 형식 유지
        state["user_input"] = user_input

        # LangGraph 실행
        state = graph.invoke(state)

        # 마지막 AI 메시지만 출력
        last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)

        if last_ai:
            print(f"🤖 AI 응답: {last_ai.content}")
        else:
            print("❌ AI 응답 없음!")

    # 전체 대화 이력 확인
    print("\n📜 최종 대화 히스토리:")
    for msg in state["messages"]:
        role = "👤 User" if isinstance(msg, HumanMessage) else "🤖 AI"
        print(f"{role}: {msg.content}")

if __name__ == "__main__":
    run_multiturn_test()