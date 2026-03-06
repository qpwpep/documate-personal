from .llm import build_llm_registry
from .make_graph import build_graph
from .node import (
    State,
    add_user_message,
    make_action_postprocess_node,
    make_planner_node,
    make_retrieve_dispatch_node,
    make_summarize_node,
    make_synthesize_node,
    make_validate_evidence_node,
)
from .settings import AppSettings, get_settings
from .tools import build_tool_registry


def build_agent_graph(settings: AppSettings | None = None):
    app_settings = settings or get_settings()

    tool_registry = build_tool_registry(app_settings)
    llm_registry = build_llm_registry(app_settings)

    summarize_node = make_summarize_node(
        llm_summarizer=llm_registry.llm_summarizer,
        verbose=llm_registry.verbose,
        max_turns=6,
    )
    planner_node = make_planner_node(
        llm_planner=llm_registry.llm_planner,
        verbose=llm_registry.verbose,
        max_turns=6,
    )
    retrieve_dispatch_node = make_retrieve_dispatch_node(
        tavily_search_tool=tool_registry.tavily_search_tool,
        upload_search_tool=tool_registry.upload_search_tool,
        rag_search_tool=tool_registry.rag_search_tool,
        verbose=llm_registry.verbose,
    )
    synthesize_node = make_synthesize_node(
        llm_synthesizer=llm_registry.llm_synthesizer,
        verbose=llm_registry.verbose,
        max_turns=6,
    )
    validate_evidence_node = make_validate_evidence_node(verbose=llm_registry.verbose)
    action_postprocess_node = make_action_postprocess_node(
        save_text_tool=tool_registry.save_text_tool,
        slack_notify_tool=tool_registry.slack_notify_tool,
        verbose=llm_registry.verbose,
    )

    graph_object = build_graph(
        state_type=State,
        add_user_node=add_user_message,
        summarize_node=summarize_node,
        planner_node=planner_node,
        retrieve_dispatch_node=retrieve_dispatch_node,
        synthesize_node=synthesize_node,
        validate_evidence_node=validate_evidence_node,
        action_postprocess_node=action_postprocess_node,
        summary_max_turns=6,
    )
    return graph_object
