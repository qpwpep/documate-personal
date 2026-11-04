"""Opt-in Decisions checks separating static source binding from execution facts.

Run from the repository root in Git Bash:
    RUN_LIVE_CODE_BINDING_SUFFICIENCY=1 uv run --locked pytest -q -s tests/core/test_code_binding_sufficiency_live.py

The default suite skips these paid provider calls. Source parsing and packet
preparation use owned code and the existing provider HTTP fake. Only the final
packet sufficiency questions call the configured Decisions provider.
No planner, embedding, synthesis, file delivery or notification API runs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from src.core.contracts import PlannerState, RetrievalState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import ComposeBody, ContractEvidence, ExtractBody, RequestContract
from src.infra.chunking import chunk_python_text
from src.infra.llm import LLMRegistry, build_llm_registry
from src.infra.settings import get_settings
from src.infra.tools.local_rag.requirements import resolve_source_requirement
from src.runtime.nodes.validation import make_pre_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary
from tests.synthesis_fixtures import synthesis_excerpt_limits


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_CODE_BINDING_SUFFICIENCY") != "1",
    reason="Set RUN_LIVE_CODE_BINDING_SUFFICIENCY=1 to call the configured Decisions provider.",
)

SOURCE = "import numpy as np\n\nif use_service:\n    np = service\nelse:\n    result = np.array([1, 2])\n"
STATIC_REQUEST = "첨부한 branch.py에서 실제 NumPy의 numpy.array를 호출하는 문장만 원문 그대로 인용하고 줄 번호를 알려줘. 다른 객체의 array 호출은 포함하지 마."
EXECUTION_REQUEST = "첨부 코드만으로 이 numpy.array 호출이 실제로 실행되어 결과를 반환했다는 사실을 증명해줘. 실행 여부의 추측이나 불확실성 안내로 대체하지 마."
CAPTURE_SOURCE = 'import numpy as np\n\nmatch service:\n    case {"client": np}:\n        result = np.array([3, 4])\n'
CAPTURE_REQUEST = "첨부 capture.py의 마지막 np.array 호출을 NumPy API 사용 사례로 볼 수 있어? match 문이 np에 어떤 영향을 주는지 코드 원문을 근거로 설명해줘."


@pytest.fixture(scope="module")
def live_registry() -> LLMRegistry:
    return build_llm_registry(get_settings().model_copy(update={"verbose": False}))


@pytest.mark.parametrize("request_text,task_query,expected_sufficient,file_id", [
    pytest.param(STATIC_REQUEST,
                 "branch.py에서 실제 NumPy numpy.array 호출 문장만 찾아 원문 그대로 인용하고 줄 번호를 확인한다. 다른 객체의 array 호출은 제외한다.",
                 True, None, id="static-binding"),
    pytest.param(EXECUTION_REQUEST, EXECUTION_REQUEST, False, None, id="actual-execution"),
    pytest.param(STATIC_REQUEST,
                 "첨부한 branch.py에서 실제 NumPy numpy.array를 호출하는 문장만 찾아 원문과 줄 번호를 확인한다.",
                 True, "31d84954596a4bc09118aa0de21a8057", id="file-scoped-browser-binding"),
    pytest.param(EXECUTION_REQUEST, EXECUTION_REQUEST, False,
                 "31d84954596a4bc09118aa0de21a8057", id="file-scoped-actual-execution"),
])
def test_live_source_call_identity_does_not_require_proof_of_execution(
    request_text: str, task_query: str, expected_sufficient: bool, file_id: str | None,
    live_registry: LLMRegistry, tmp_path: Path,
) -> None:
    task = RetrievalTask(route="upload", requirement_id="numpy_array_calls", query=task_query, k=5,
                         requirement={"symbols": ["numpy.array"], "match": "symbol",
                                      "file_ids": [file_id] if file_id is not None else []})
    document = chunk_python_text(path="branch.py", text=SOURCE, source_content=SOURCE.encode("utf-8"),
                                 source_uri=f"upload:///binding/{file_id or 'branch.py'}", chunk_size=2000, chunk_overlap=0)
    if file_id is not None:
        document.parsed.elements[0].metadata["file_id"] = file_id
    retrieved = resolve_source_requirement(requirement=task.requirement, source_documents=[document.parsed])
    contract = RequestContract(
        body=ExtractBody(instruction=request_text, requirement_ids=[task.requirement_id], evidence_ids=["e1"]),
        evidence=[ContractEvidence(id="e1", turn_id="current", quote=request_text,
                                   scope="current_request", interpretation="instruction")],
    )
    state = build_graph_state_input(
        user_input=request_text, current_turn_id="current", request_contract=contract,
        messages=[HumanMessage(content=request_text, id="current")],
        planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task])),
        retrieval=RetrievalState(hit_log=[hit.model_copy(update={"requirement_id": task.requirement_id})
                                         for hit in retrieved.hits]),
        retry=RetryState(max_retries=0, max_content_repairs=0),
    )
    boundary = DecisionsBoundary()
    state.update(make_pre_synthesis_validation_node(
        False, decisions_client=boundary, excerpt_limits=synthesis_excerpt_limits(),
    )(state))
    packet_call, = [call for call in boundary.requests if "requirements" in json.loads(call["input"])]
    selected, = json.loads(packet_call["input"])["sources"].values()
    assert selected["file_id"] == file_id
    result = live_registry.decisions_client.evaluate(
        input=packet_call["input"], questions=packet_call["questions"], stage="pre_synthesis_validation",
    )
    observation = {"raw_input": packet_call["input"], "input": json.loads(packet_call["input"]),
                   "questions": packet_call["questions"],
                   "probabilities": result.probabilities, "refused_names": sorted(result.refused_names)}
    record = tmp_path / "decisions.json"
    record.write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"expected_sufficient": expected_sufficient, "probabilities": result.probabilities,
                      "record": str(record)}, ensure_ascii=False))

    assert result.refused_names == set()
    assert (result.probabilities["requirement_0_target_identified"] >= 0.8) is True, str(record)
    assert (result.probabilities["requirement_0_target_analysis_supported"] >= 0.8) is expected_sufficient, str(record)


@pytest.mark.parametrize("unselected_tail,expected_sufficient", [
    pytest.param("", True, id="last-call-with-prefix-omitted"),
    pytest.param("\nresult = np.array([9])\n", False, id="last-call-with-tail-omitted"),
])
@pytest.mark.parametrize("task_query", [
    pytest.param(CAPTURE_REQUEST, id="original-question"),
    pytest.param("첨부 capture.py에서 마지막 np.array 호출을 찾아 NumPy API 사용 사례인지 확인하고, "
                 "match 문이 np에 미치는 영향을 코드 원문에 근거해 설명", id="browser-find-call"),
    pytest.param("첨부 capture.py의 마지막 np.array 호출이 NumPy API 사용 사례인지 확인하고, "
                 "match 문이 np에 미치는 영향을 코드 원문과 함께 설명", id="browser-check-call"),
])
def test_live_last_call_requires_the_end_of_the_file_not_its_unrelated_prefix(
    unselected_tail: str, expected_sufficient: bool, task_query: str,
    live_registry: LLMRegistry, tmp_path: Path,
) -> None:
    source = CAPTURE_SOURCE + unselected_tail
    document = chunk_python_text(path="capture.py", text=source, source_content=source.encode("utf-8"),
                                 source_uri="upload:///binding/capture.py", chunk_size=2000, chunk_overlap=0)
    evidence = build_evidence(snapshot=document.parsed.snapshot, element=document.parsed.elements[0],
                              start=20, end=len(CAPTURE_SOURCE))
    task = RetrievalTask(route="upload", requirement_id="last_call", query=task_query, k=5,
                         requirement={"symbols": ["np.array"], "match": "symbol"})
    contract = RequestContract(
        body=ComposeBody(instruction=CAPTURE_REQUEST, evidence_ids=["e1"]),
        evidence=[ContractEvidence(id="e1", turn_id="current", quote=CAPTURE_REQUEST,
                                   scope="current_request", interpretation="instruction")],
    )
    state = build_graph_state_input(
        user_input=CAPTURE_REQUEST, current_turn_id="current", request_contract=contract,
        messages=[HumanMessage(content=CAPTURE_REQUEST, id="current")],
        planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task])),
        retrieval=RetrievalState(hit_log=[SearchHit(evidence=evidence, requirement_id=task.requirement_id,
            rank=1, score=RetrievalScore(metric="source_symbol", direction="higher"))]),
        retry=RetryState(max_retries=0, max_content_repairs=0),
    )
    boundary = DecisionsBoundary()
    state.update(make_pre_synthesis_validation_node(
        False, decisions_client=boundary, excerpt_limits=synthesis_excerpt_limits(),
    )(state))
    packet_call, = [call for call in boundary.requests if "requirements" in json.loads(call["input"])]
    payload = json.loads(packet_call["input"])
    selected, = payload["sources"].values()
    assert selected["excerpt"] == CAPTURE_SOURCE[20:]
    assert "operation" not in payload["tasks"][task.requirement_id]
    result = live_registry.decisions_client.evaluate(
        input=packet_call["input"], questions=packet_call["questions"], stage="pre_synthesis_validation",
    )
    record = tmp_path / "decisions.json"
    record.write_text(json.dumps({"raw_input": packet_call["input"], "input": payload,
        "questions": packet_call["questions"],
        "probabilities": result.probabilities, "refused_names": sorted(result.refused_names)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"expected_sufficient": expected_sufficient, "probabilities": result.probabilities,
                      "record": str(record)}, ensure_ascii=False))
    assert not result.refused_names
    assert (result.probabilities["requirement_0_target_identified"] >= 0.8) is expected_sufficient, str(record)
    assert (result.probabilities["requirement_0_target_analysis_supported"] >= 0.8) is True, str(record)


@pytest.mark.parametrize("selected_end", [15, 16], ids=["assignment-without-final-lf", "assignment-with-final-lf"])
def test_live_named_assignment_lookup_does_not_require_whole_file_coverage(
    selected_end: int, live_registry: LLMRegistry, tmp_path: Path,
) -> None:
    source = "RETRY_LIMIT = 3\n"
    query = "settings.py에서 RETRY_LIMIT의 값 또는 할당을 확인해 답변"
    document = chunk_python_text(path="settings.py", text=source, source_content=source.encode("utf-8"),
                                 source_uri="upload:///binding/settings.py", chunk_size=2000, chunk_overlap=0)
    evidence = build_evidence(snapshot=document.parsed.snapshot, element=document.parsed.elements[0], end=selected_end)
    task = RetrievalTask(route="upload", requirement_id="retry_limit", query=query, k=5,
                         requirement={"symbols": ["RETRY_LIMIT"], "match": "symbol"})
    state = build_graph_state_input(
        user_input=query, current_turn_id="current", request_contract=RequestContract(),
        messages=[HumanMessage(content=query, id="current")],
        planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task])),
        retrieval=RetrievalState(hit_log=[SearchHit(evidence=evidence, requirement_id=task.requirement_id,
            rank=1, score=RetrievalScore(metric="source_symbol", direction="higher"))]),
        retry=RetryState(max_retries=0, max_content_repairs=0),
    )
    boundary = DecisionsBoundary()
    state.update(make_pre_synthesis_validation_node(
        False, decisions_client=boundary, excerpt_limits=synthesis_excerpt_limits(),
    )(state))
    packet_call, = [call for call in boundary.requests if "requirements" in json.loads(call["input"])]
    payload = json.loads(packet_call["input"])
    selected, = payload["sources"].values()
    assert selected["excerpt"] == source[:selected_end]
    assert selected["element_extent"]["omitted_suffix"] is (selected_end < len(source))
    result = live_registry.decisions_client.evaluate(
        input=packet_call["input"], questions=packet_call["questions"], stage="pre_synthesis_validation",
    )
    record = tmp_path / "decisions.json"
    record.write_text(json.dumps({"raw_input": packet_call["input"], "input": payload,
        "questions": packet_call["questions"],
        "probabilities": result.probabilities, "refused_names": sorted(result.refused_names)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"expected_sufficient": True, "probabilities": result.probabilities,
                      "record": str(record)}, ensure_ascii=False))
    assert not result.refused_names
    assert result.probabilities["requirement_0_target_identified"] >= 0.8, str(record)
    assert result.probabilities["requirement_0_target_analysis_supported"] >= 0.8, str(record)
