from .llm import build_llm_registry
from .make_graph import build_graph
from .node import State, add_user_message, make_chatbot_node, make_summarize_node
from .settings import AppSettings, get_settings
from .tools import build_tool_registry


def build_agent_graph(settings: AppSettings | None = None):
    app_settings = settings or get_settings()

    tool_registry = build_tool_registry(app_settings)
    llm_registry = build_llm_registry(app_settings, tool_registry.all_tools)

    summarize_node = make_summarize_node(
        llm_summarizer=llm_registry.llm_summarizer,
        verbose=llm_registry.verbose,
        max_turns=6,
    )
    chatbot_node = make_chatbot_node(
        llm_with_tools=llm_registry.llm_with_tools,
        verbose=llm_registry.verbose,
        max_turns=6,
    )

    graph_object = build_graph(
        state_type=State,
        add_user_node=add_user_message,
        summarize_node=summarize_node,
        chatbot_node=chatbot_node,
        tool_list=tool_registry.all_tools,
    )
    return graph_object
