print(">>> running main from:", __file__)

import sys
import os
import argparse
import subprocess
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from .make_graph import build_graph
from .llm import VERBOSE
from .tools import save_text_to_file
from .util.util import get_project_root_path

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

def run_web_service(mode: str):
    """
    start_services.sh 쉘 스크립트를 실행하여 FastAPI와 Streamlit 서비스를 시작합니다.
    """
    script_name = ""
    if mode == "startweb":
        script_name = "script/start_services.sh"
    elif mode == "stopweb":
        script_name = "script/stop_services.sh"
    else:
        print("실행할 수 없는 mode 입니다.")
        return

    root_path = get_project_root_path()
    print(root_path)

    # 쉘 스크립트 파일 경로 설정 (현재 디렉토리 기준)
    script_path = os.path.join(root_path, script_name)
    print(script_path)
    # 1. 스크립트 파일 존재 여부 확인
    if not os.path.exists(script_path):
        print(f"오류: 실행할 스크립트 파일 '{script_name}'을 찾을 수 없습니다.")
        print("스크립트 파일을 이 파이썬 파일과 같은 위치에 저장했는지 확인해주세요.")
        sys.exit(1)
        
    # 2. 실행 권한 확인 (중요!)
    if not os.access(script_path, os.X_OK):
        # 권한 부여
        os.chmod(script_path, 0o755)

        if not os.access(script_path, os.X_OK):
            print(f"오류: 스크립트 파일 '{script_name}'에 실행 권한이 없습니다.")
            print(f"'chmod +x {script_name}' 명령어를 실행하여 권한을 부여해주세요.")
            sys.exit(1)

    print("=" * 40)
    print(f"'{script_name}' 스크립트를 실행하여 웹 서비스를 시작합니다...")
    print("=" * 40)
    
    try:
        # subprocess.run을 사용하여 외부 쉘 스크립트를 실행합니다.
        # check=True: 스크립트 실행이 실패하면 예외를 발생시킵니다.
        # 이 스크립트가 백그라운드(&)로 프로세스를 띄우므로, 파이썬은 바로 다음 코드를 실행합니다.
        _ = subprocess.run([script_path], check=True, text=True)

    except subprocess.CalledProcessError as e:
        print(f"스크립트 실행 중 오류가 발생했습니다. 반환 코드: {e.returncode}")
        print(f"에러 메시지:\n{e.stderr}")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")


if __name__ == "__main__":

    """
    CLI 기반 실행 명령어(기본):
        uv run python -m new_src.main
    WebService 실행 명령어:
        uv run python -m new_src.main --mode startweb (웹서비스 시작)
        uv run python -m new_src.main --mode stopweb (웹서비스 종료)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", 
                        type=str, 
                        default="cli", 
                        choices=['cli', 'startweb', 'stopweb'], 
                        help="실행 모드를 지정합니다: 'web' (웹 서비스 시작), 'cli' (cli 실행)")
    
    args = parser.parse_args()

    if args.mode == 'startweb' or args.mode == 'stopweb':
        # 'web' 모드 선택 시, run_web_services 함수 호출
        run_web_service(args.mode)
    else:
        run_cli()
