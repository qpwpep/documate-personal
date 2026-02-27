import json
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .graph_builder import build_agent_graph
from .upload_helpers import build_temp_retriever


class AgentFlowManager:
    """Manages per-session LangGraph execution and message state."""

    def __init__(self):
        self.graph = build_agent_graph()
        self.messages: List[Any] = []
        self.retriever = None
        self.upload_file_path: Optional[str] = None

    def run_agent_flow(self, user_input: str, upload_file_path: Optional[str] = None) -> dict:
        current_messages = self.messages

        # Normalize special quit command handling to a dict response.
        if user_input.lower() in {"exit", "종료", "quit", "q"}:
            self.messages = []
            self.retriever = None
            self.upload_file_path = None
            return {
                "message": "채팅 세션이 초기화되었습니다. 다시 시작합니다.",
                "filepath": "",
                "response": None,
            }

        try:
            state = {
                "user_input": user_input,
                "messages": current_messages,
            }

            if upload_file_path is not None:
                # Rebuild retriever only when file path changes.
                if self.upload_file_path != upload_file_path:
                    self.upload_file_path = upload_file_path
                    self.retriever = build_temp_retriever(upload_file_path)

                if self.retriever is not None:
                    state["retriever"] = self.retriever
            else:
                self.retriever = None
                self.upload_file_path = None

            response = self.graph.invoke(state)

            updated_messages = response["messages"]
            self.messages = updated_messages

            final_answer = ""
            file_path = ""

            # Walk backwards until the current user turn.
            for message in reversed(updated_messages):
                if isinstance(message, HumanMessage):
                    break

                if not final_answer and isinstance(message, AIMessage):
                    final_answer = message.content
                elif (
                    not file_path
                    and isinstance(message, ToolMessage)
                    and message.name == "save_text"
                ):
                    try:
                        tool_result_dict: Dict[str, Any] = json.loads(message.content)
                        extracted_path = tool_result_dict.get("file_path")
                        if extracted_path and os.path.exists(extracted_path):
                            file_path = extracted_path
                    except json.JSONDecodeError:
                        continue

                if final_answer and file_path:
                    break

            return {"message": final_answer, "filepath": file_path, "response": response}

        except Exception as e:
            print(f"Agent 실행 중 오류 발생: {e}")
            return {"message": str(e), "filepath": "", "response": None}
