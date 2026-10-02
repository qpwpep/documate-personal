"""Small opt-in production-boundary checks; no benchmark, uploads, or delivery."""
from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage

from src.app.agent_manager import AgentFlowManager
from src.core.answer_schema import AnswerResponse, export_answer_text
from src.core.planner_schema import RetrievalPlanOutput
from src.infra.llm import build_llm_registry
from src.infra.llm_boundary import run_structured_call
from src.infra.settings import get_settings


pytestmark = pytest.mark.skipif(
    os.getenv("STRUCTURED_OUTPUT_LIVE_TEST") != "true",
    reason="Requires explicit STRUCTURED_OUTPUT_LIVE_TEST=true",
)


@pytest.mark.parametrize("query", [
    "pandas merge 사용법을 공식 문서 기준으로 간단히 설명해줘. 저장하거나 전송하지 마.",
    "NumPy reshape의 order 매개변수를 공식 문서 기준으로 간단히 설명해줘. 저장하거나 전송하지 마.",
])
def test_normal_question_completes_search_and_checked_answer(query, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    settings = get_settings().model_copy(update={"verbose": False})
    manager = AgentFlowManager(settings)
    result = manager.run_agent_flow(query)
    assert result["status"] == "completed", {
        "status": result["status"], "problem": result.get("problem"),
        "diagnostics": result.get("debug", {}).get("llm_diagnostics"),
        "retrieval": result.get("debug", {}).get("retrieval_diagnostics"),
        "planner": result.get("debug", {}).get("planner_diagnostics"),
        "validation": result.get("debug", {}).get("validation_events"),
    }
    answer = AnswerResponse.model_validate(result["response"])
    assert export_answer_text(answer).strip()
    assert answer.citations
    assert answer.actions == []
    assert "tavily_search" in result["debug"]["tool_calls"]
    assert not {"save_text", "slack_notify"}.intersection(result["debug"]["tool_calls"])
    assert not result["debug"]["llm_diagnostics"]
    print({"model": settings.chat_model, "status": result["status"],
           "citations": len(answer.citations), "tools": result["debug"]["tool_calls"],
           "llm_calls": len(result["debug"]["llm_calls"] or [])})


def test_retrieval_only_schema_is_accepted_by_configured_provider(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    registry = build_llm_registry(get_settings().model_copy(update={"verbose": False}))
    result = run_structured_call(
        registry.llm_planner_retry,
        [HumanMessage(content="No retrieval is necessary. Return use_retrieval=false and tasks=[].")],
        stage="planner", validate=RetrievalPlanOutput.model_validate,
    )
    assert result.use_retrieval is False
    assert result.tasks == []
