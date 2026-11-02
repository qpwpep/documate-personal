"""Opt-in provider regressions for supported-argument questions.

From the repository root in Git Bash:
    RUN_LIVE_API_AVAILABILITY=1 uv run --locked pytest -q tests/core/test_api_availability_live.py

These tests send synthetic questions and short source fixtures to the configured
planner and Decisions providers. They incur usage and do not mock model judgments.
No search, synthesis, file upload, or delivery action runs.
"""

from __future__ import annotations

import os

import pytest

from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts import PlannerState, ResponseState, RetrievalState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.infra.llm import LLMRegistry, build_llm_registry
from src.infra.settings import get_settings
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.synthesis_fixtures import synthesis_excerpt_limits


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_API_AVAILABILITY") != "1",
    reason="Set RUN_LIVE_API_AVAILABILITY=1 to call the configured planner and Decisions providers.",
)


@pytest.fixture(scope="module")
def live_registry() -> LLMRegistry:
    return build_llm_registry(get_settings().model_copy(update={"verbose": False}))


@pytest.mark.parametrize("query, option", [
    ("Python 3.9 공식 문서만 기준으로 Path.stat()에 follow_symlinks=False를 전달할 수 있는지 확인해 줘. 이후 버전의 기능을 섞지 마.", "follow_symlinks"),
    ("Python 3.9의 zip()에 strict=True 옵션이 있어? Python 3.9 공식 문서로만 확인해 줘.", "strict"),
])
def test_live_planner_does_not_presuppose_a_proposed_keyword_exists(query, option, live_registry):
    state = build_graph_state_input(user_input=query, current_turn_id="availability")

    output = make_planner_node(live_registry.llm_planner, False)(state)["planner"].output

    assert output.use_retrieval and output.tasks
    for task in output.tasks:
        assert task.requirement.version == "3.9"
        assert task.requirement.aspects == []
        assert set(task.requirement.evidence_needs) == {"signature", "explanation"}
    assert option in " ".join(task.query for task in output.tasks)


def test_live_planner_keeps_supported_setting_behavior_requirements(live_registry):
    query = "Path.stat(follow_symlinks=False)는 어떤 동작을 하는지 Python 공식 문서로 설명해 줘."
    state = build_graph_state_input(user_input=query, current_turn_id="setting-meaning")

    output = make_planner_node(live_registry.llm_planner, False)(state)["planner"].output

    assert output.use_retrieval and output.tasks
    task, = output.tasks
    assert task.requirement.version is None
    assert set(task.requirement.aspects) == {"follow_symlinks", "False"}
    assert {"parameters", "explanation"} <= set(task.requirement.evidence_needs)


@pytest.mark.parametrize("symbol, keyword, value, excerpt, source_version, supported_absence", [
    pytest.param("pathlib.Path.stat", "follow_symlinks", "False",
        "Path.stat()¶\n\nReturn information about this path.", "3.9", True, id="closed-stat"),
    pytest.param("builtins.zip", "strict", "True",
        "zip(*iterables)¶\n\nMake an iterator that aggregates elements from each of the iterables.",
        "3.9", True, id="positional-variadic-zip"),
    pytest.param("pathlib.Path.stat", "follow_symlinks", "False",
        "Path.stat(*, follow_", "3.9", False, id="clipped-declaration"),
    pytest.param("pathlib.Path.stat", "follow_symlinks", "False",
        "Path.stat(**kwargs)¶\n\nRead path metadata using the supplied keyword arguments.",
        "3.9", False, id="open-keywords"),
    pytest.param("pathlib.Path.stat", "follow_symlinks", "False",
        "Path.stat()¶\n\nReturn information about this path.", "3.8", False, id="different-version"),
    pytest.param("pathlib.Path.stat", "follow_symlinks", "False",
        "Path.stat reads information about a path.", "3.9", False, id="missing-declaration"),
])
def test_live_availability_requires_a_complete_matching_declaration(
    symbol, keyword, value, excerpt, source_version, supported_absence, live_registry,
):
    query = f"Python 3.9에서 {symbol}에 {keyword}={value}를 전달할 수 있는지 공식 문서로 확인해 줘."
    task = RetrievalTask(requirement_id="availability", route="docs", query=query, k=2,
        requirement={"library": "Python", "version": "3.9", "symbols": [symbol], "match": "symbol",
                     "evidence_needs": ["signature", "explanation"]})
    module = "functions" if symbol == "builtins.zip" else "pathlib"
    title = "Built-in Functions" if symbol == "builtins.zip" else "pathlib — Object-oriented filesystem paths"
    evidence = build_evidence(
        snapshot=build_snapshot(source_uri=f"https://docs.python.org/{source_version}/library/{module}.html",
            title=f"{title} — Python {source_version} documentation", media_type="text/html",
            source_type="official", content=excerpt, parser="official_html", parser_version="2"),
        element=DocumentElement(element_id="declaration", kind="paragraph", text=excerpt),
    )
    contract = RequestContract(request_id="availability", body={"kind": "compose", "instruction": query})
    state = build_graph_state_input(user_input=query, current_turn_id="availability",
        request_contract=contract, planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task])),
        retrieval=RetrievalState(hit_log=[SearchHit(evidence=evidence, requirement_id="availability", rank=1,
            score={"metric": "fixture", "direction": "higher"})]),
        retry=RetryState(max_retries=1, original_tasks=[task]))

    state.update(make_pre_synthesis_validation_node(False, decisions_client=live_registry.decisions_client,
        excerpt_limits=synthesis_excerpt_limits())(state))

    assert (not state["retry"].needs_retry) is supported_absence, "\n".join(state["debug"].validation_events)
    if not supported_absence:
        return
    prepared = state["retrieval"].prepared_evidence.normal
    assert not prepared.failed_requirement_ids
    assert prepared.evidence
    answer = text_document(f"Python 3.9의 {symbol}은 {keyword} 키워드 인자를 받지 않습니다.",
                           basis="source", refs=[item.id for item in prepared.evidence])
    state["response"] = ResponseState(result=finalize_answer(answer, prepared.evidence, retrieval_required=True),
        evidence_packet=prepared.evidence, evidence_requirement_map=prepared.requirement_map,
        synthesis_attempt=1, request_id=contract.request_id, contract_revision=contract.revision)

    state.update(make_post_synthesis_validation_node(False, decisions_client=live_registry.decisions_client)(state))

    assert not state["retry"].needs_retry, "\n".join(state["debug"].validation_events)
    assert state["response"].result.semantic_validation.passed
