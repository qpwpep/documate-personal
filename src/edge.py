from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END

def wire_tool_edges(graph_builder, tool_node_name: str = "tools", chatbot_node_name: str = "chatbot", tavily_tool=None):
    """Attach tool execution node and conditional edges to the builder."""
    if tavily_tool is None:
        raise ValueError("tavily_tool must be provided")

    # Add a ToolNode that can execute external tools
    tool_node = ToolNode(tools=[tavily_tool])
    graph_builder.add_node(tool_node_name, tool_node)

    # Conditional edge from chatbot -> either tools or END depending on tool-calls
    graph_builder.add_conditional_edges(
        chatbot_node_name,
        tools_condition,
        {tool_node_name: tool_node_name, END: END},
    )

    # After tools run, go back to chatbot
    graph_builder.add_edge(tool_node_name, chatbot_node_name)

    return graph_builder