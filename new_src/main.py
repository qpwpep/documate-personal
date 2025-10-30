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
    load_dotenv()
    graph = build_graph()
    maybe_save_mermaid_png(graph)

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break

            response = graph.invoke({"messages": [HumanMessage(content=user_input)]})

            if VERBOSE:
                for msg in response["messages"]:
                    try:
                        msg.pretty_print()
                    except Exception:
                        print(repr(msg))
            
            # --- Always prompt to save the last AI answer (works in both verbose/non-verbose) ---
            last_ai = next(
                (m for m in reversed(response["messages"]) if isinstance(m, AIMessage)), None)
            if last_ai:
                if not VERBOSE:
                    print(last_ai.content)

                choice = input("Save this response to a .txt file? (y/N): ").strip().lower()
                if choice in {"y", "yes"}:
                    try:
                        result = save_text_to_file(last_ai.content, filename_prefix="response")
                        print(result)
                    except Exception as e:
                        print("Save failed:", e)
            else:
                print("[warn] No AIMessage found in response.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print("Error:", e)
            break
        
if __name__ == "__main__":
    run_cli()
