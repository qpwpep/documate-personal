from .make_graph import build_graph

def build_agent_graph():
    """LangGraph 에이전트 그래프를 구성하고 반환하는 함수입니다."""
    graph_object = build_graph()
    return graph_object