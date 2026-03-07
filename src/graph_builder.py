from .llm import build_llm_registry
from .make_graph import build_graph
from .nodes.actions import make_action_postprocess_node
from .nodes.planner import make_planner_node
from .nodes.retrieval import make_retrieve_dispatch_node
from .nodes.session import add_user_message, make_summarize_node
from .nodes.state import State
from .nodes.synthesis import make_synthesize_node
from .nodes.validation import make_validate_evidence_node
from .settings import AppSettings, get_settings
from .tools import build_tool_registry


def build_agent_graph(settings: AppSettings | None = None):
    app_settings = settings or get_settings()
    has_default_slack_destination = bool(
        app_settings.slack_default_user_id or app_settings.slack_default_dm_email
    )

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
        has_default_slack_destination=has_default_slack_destination,
    )
    validate_evidence_node = make_validate_evidence_node(verbose=llm_registry.verbose)
    action_postprocess_node = make_action_postprocess_node(
        save_text_tool=tool_registry.save_text_tool,
        slack_notify_tool=tool_registry.slack_notify_tool,
        verbose=llm_registry.verbose,
        has_default_slack_destination=has_default_slack_destination,
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
