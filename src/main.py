# main.py
import argparse
from dotenv import load_dotenv

# 로컬/상대 임포트 환경에 맞춰 수정해 주세요
try:
    from .make_graph import build_graph
except Exception:
    from make_graph import build_graph




def run(query: str, model: str = "gpt-4.1-mini", max_results: int = 5):
    app = build_graph(model=model, max_results=max_results)
    state = {
        "messages": [],
        "query": query,
        "gathered_urls": [],
        "used_web": False,
    }
    final = app.invoke(state)
    last = final["messages"][-1]
    print("\n================ 최종 답변 ================\n")
    print(getattr(last, "content", last))

    # --- provenance 출력 ---
    provenance = "WEB(웹검색 기반)" if final.get("used_web") else "LLM(자체 지식 기반)"
    print("\n---------------- PROVENANCE ----------------")
    print(f"[PROVENANCE] {provenance}")
    urls = final.get("gathered_urls") or []
    if urls:
        print("[URLS]")
        for u in urls:
            print(f"- {u}")




def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--max_results", type=int, default=5)
    args = parser.parse_args()
    run(query=args.query, model=args.model, max_results=args.max_results)




if __name__ == "__main__":
    main()