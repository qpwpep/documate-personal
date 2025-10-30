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

    history = []  # ✅ keep conversation across turns

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break

            # send history + this turn
            response = graph.invoke({"messages": history + [HumanMessage(content=user_input)]})

            msgs = response.get("messages", [])

            # Always show final answer
            last_ai = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
            if last_ai:
                print("\n--- Final Answer ---")
                print(last_ai.content)
            else:
                print("[warn] No AIMessage found in response.")

            # ✅ carry forward for the next turn
            history = msgs

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print("Error:", e)
            break
        
if __name__ == "__main__":
    run_cli()
