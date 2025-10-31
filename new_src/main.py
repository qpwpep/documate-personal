print(">>> running main from:", __file__)

import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from .make_graph import build_graph
from .llm import VERBOSE
from .tools import save_text_to_file

def maybe_save_mermaid_png(graph):
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_bytes)
        print("Saved graph to graph.png")
    except Exception:
        # Might fail if mermaid/graphviz backends aren't installed
        pass

def run_cli():
    from langchain_core.messages import AIMessage
    load_dotenv()
    graph = build_graph()
    maybe_save_mermaid_png(graph)
    
    # 멀티턴을 위한 전체 메시지 저장 변수
    messages = []

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            
            # LangGraph에 멀티턴 상태 전달
            state = {
                "user_input": user_input,
                "messages": messages  # 이전 대화 전달
            }

            response = graph.invoke(state)
            
            # 결과 메시지 업데이트
            messages = response["messages"]

            if VERBOSE:
                for msg in response["messages"]:
                    try:
                        msg.pretty_print()
                    except Exception:
                        print(repr(msg))
            else:
                # Print the last AI message
                for m in reversed(messages):
                    if isinstance(m, AIMessage):
                        print(m.content)
                        break
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print("Error:", e)
            break

if __name__ == "__main__":
    run_cli()
