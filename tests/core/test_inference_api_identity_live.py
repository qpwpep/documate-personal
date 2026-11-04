"""Opt-in grounding checks for code bindings versus API-looking syntax.

From the repository root in Git Bash:
    RUN_LIVE_INFERENCE_API_IDENTITY=1 uv run --locked pytest -q -s tests/core/test_inference_api_identity_live.py

Five fixed candidates retain the observed four-unit answer. The public
post-validation node constructs each isolated unit input and predicate through
the provider HTTP fake. Each unchanged source-grounding call reaches the
configured Decisions provider once: four calls per case, twenty in total. Body
and requirement judgments remain HTTP fakes, and returned source judgments are
replayed without another API call. The default suite skips these paid calls.
No planner, retrieval, synthesis,
save or notification API runs. Diagnostic JSON is written to pytest temporary
directories and printed for the caller's local log.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import pytest
from langchain_core.messages import HumanMessage

from src.core.answer_schema import AnswerDocument, finalize_answer
from src.core.answer_schema.source_locations import SOURCE_LOCATION_PLACEHOLDER
from src.core.contracts import GraphState, PlannerState, ResponseState, RetrievalState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import ComposeBody, ContractEvidence, RequestContract
from src.infra.chunking import chunk_python_text
from src.infra.llm import LLMRegistry, build_llm_registry
from src.infra.llm_boundary import capture_llm_diagnostics, request_call_budget
from src.infra.settings import get_settings
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_INFERENCE_API_IDENTITY") != "1",
    reason="Set RUN_LIVE_INFERENCE_API_IDENTITY=1 to call the configured Decisions provider.",
)

CAPTURE_SOURCE = (
    "import numpy as np\n\nmatch service:\n"
    '    case {"client": np}:\n        result = np.array([3, 4])\n'
)
BRANCH_SOURCE = "import numpy as np\n\nif use_service:\n    np = service\nelse:\n    result = np.array([1, 2])\n"
CAPTURE_FILE_ID = "947c3a44f3f8447eafa331b3f05f6888"
BRANCH_FILE_ID = "edc20f6301114f69be937e8ce6ca8530"
CAPTURE_URI = "upload:///8c5bb892-e13d-4570-ae58-c9671ea1b2e3/947c3a44f3f8447eafa331b3f05f6888"
CAPTURE_REQUEST = (
    "첨부 capture.py의 마지막 np.array 호출을 NumPy API 사용 사례로 볼 수 있어? "
    "match 문이 np에 어떤 영향을 주는지 코드 원문을 근거로 설명해줘."
)
AFFIRMATIVE_TEXT = (
    "따라서 이 코드 조각에서 `np.array([3, 4])`는 `np`를 통해 `array`를 호출하는 형태의 "
    "NumPy API 사용 사례로 볼 수 있습니다."
)
CLAIMS = {
    "actual_bad": AFFIRMATIVE_TEXT + (
        " 다만 이 발췌만으로 `service`의 실제 값이나 `np`가 NumPy를 가리키는지는 확인할 수 없으므로, "
        "실제 실행 시 NumPy API 호출이라고 단정할 수는 없습니다."
    ),
    "affirmative_only": AFFIRMATIVE_TEXT,
    "correct_unknown": (
        '`case {"client": np}`는 매치된 매핑의 client 값을 이름 np에 바인딩합니다. '
        "따라서 이 호출은 그 값의 array 속성을 호출하며, 그 값이 NumPy 모듈인지는 원문에서 확인되지 않습니다. "
        "이 자료만으로 NumPy API 사용 사례라고 확정할 수 없습니다."
    ),
    "static_binding": (
        "else 분기에서는 if 분기의 np = service 대입을 지나지 않으므로, import numpy as np가 만든 "
        "NumPy 별칭 np를 통해 array를 호출합니다. 이는 해당 경로의 정적 이름 관계이며 "
        "이 분기가 실제로 실행됐다는 뜻은 아닙니다."
    ),
    "explicit_conditional": (
        "service의 client 값이 NumPy 모듈이라는 추가 가정을 두면 이 호출은 NumPy의 array를 호출합니다. "
        "그러나 원문은 그 가정을 입증하지 않으므로 이 호출을 NumPy 사용 사례라고 확정할 수 없습니다."
    ),
}
Case = Literal["actual_bad", "affirmative_only", "correct_unknown", "static_binding", "explicit_conditional"]


def _state(case: Case) -> GraphState:
    if case == "static_binding":
        title, source, start, end = "branch.py", BRANCH_SOURCE, 0, 89
        file_id = BRANCH_FILE_ID
        uri = f"upload:///inference-api-identity/{file_id}"
        request = "첨부 branch.py의 else 분기 np.array 호출이 NumPy API 사용 사례인지 코드 원문을 근거로 설명해줘."
        description = (
            "첨부된 `branch.py`에는 `import numpy as np`와 `if use_service:` 분기의 `np = service`, "
            "그리고 `else:` 분기의 `result = np.array([1, 2])`가 있습니다."
        )
        explanation = (
            "`np = service` 대입은 if 분기에만 있으며 else 분기는 그 대입을 수행하지 않습니다. "
            "이 원문만으로 해당 분기가 실제로 실행되었는지는 확인할 수 없습니다."
        )
    else:
        title, source, start, end = "capture.py", CAPTURE_SOURCE, 20, 94
        file_id = CAPTURE_FILE_ID
        uri, request = CAPTURE_URI, CAPTURE_REQUEST
        description = (
            '첨부된 `capture.py`의 확인 가능한 코드에는 `match service:`의 패턴 `case {"client": np}:`와 '
            "그 안의 `result = np.array([3, 4])`가 있습니다."
        )
        explanation = (
            '패턴은 `service`가 `"client"` 키를 가진 딕셔너리에 매치되면 그 키의 값을 이름 `np`에 바인딩합니다. '
            "그러므로 매치된 경우 `np.array([3, 4])`는 바인딩된 값을 대상으로 호출하며, "
            "그 값이 NumPy 모듈인지 여부는 발췌에 나타나 있지 않습니다."
        )
    parsed = chunk_python_text(
        path=title, text=source, source_content=source.encode("utf-8"), source_uri=uri,
        chunk_size=2000, chunk_overlap=0,
    ).parsed
    parsed.elements[0].metadata["file_id"] = file_id
    evidence = build_evidence(snapshot=parsed.snapshot, element=parsed.elements[0], start=start, end=end)
    task = RetrievalTask(
        route="upload", requirement_id="array-api-identity", query=request, k=5,
        requirement={"symbols": ["np.array"], "match": "symbol", "file_ids": [file_id]},
    )
    contract = RequestContract(
        body=ComposeBody(instruction=request, evidence_ids=["instruction"]),
        evidence=[ContractEvidence(id="instruction", turn_id="current", quote=request,
                                   scope="current_request", interpretation="instruction")],
    )
    document = AnswerDocument.model_validate({"blocks": [
        {"type": "paragraph", "content": [
            {"text": description, "basis": "source", "refs": [evidence.id]},
            {"text": SOURCE_LOCATION_PLACEHOLDER, "basis": "source", "refs": [evidence.id],
             "source_location": {"ref": evidence.id, "start": start, "end": end}},
        ]},
        {"type": "paragraph", "content": [
            {"text": CLAIMS[case], "basis": "inference", "refs": [evidence.id]},
            {"text": explanation, "basis": "inference", "refs": [evidence.id]},
        ]},
    ]})
    result = finalize_answer(document, [evidence], retrieval_required=True)
    assert result.content.blocks[0].content[1].text == (
        "branch.py:1–6" if case == "static_binding" else "capture.py:3–5"
    )
    return build_graph_state_input(
        user_input=request, current_turn_id="current", request_contract=contract,
        messages=[HumanMessage(content=request, id="current")],
        planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task])),
        retrieval=RetrievalState(hit_log=[SearchHit(
            evidence=evidence, requirement_id=task.requirement_id, rank=1,
            score=RetrievalScore(metric="source_symbol", direction="higher"),
        )]),
        retry=RetryState(max_retries=0, max_content_repairs=1),
        response=ResponseState(
            result=result, evidence_packet=[evidence],
            evidence_requirement_map={evidence.id: [task.requirement_id]}, synthesis_attempt=1,
            request_id=contract.request_id, contract_revision=contract.revision,
        ),
    )


@pytest.fixture(scope="module")
def live_registry() -> LLMRegistry:
    return build_llm_registry(get_settings().model_copy(update={"verbose": False}))


@pytest.mark.parametrize("case", [
    "actual_bad", "affirmative_only", "correct_unknown", "static_binding", "explicit_conditional",
])
def test_live_inference_requires_bindings_for_api_attribution(
    case: Case, live_registry: LLMRegistry, tmp_path: Path,
) -> None:
    state = _state(case)
    original = state["response"].result.model_copy(deep=True)
    capture = DecisionsBoundary()
    make_post_synthesis_validation_node(False, decisions_client=capture, pass_threshold=0.8)(state)
    source_calls = [call for call in capture.requests if set(json.loads(call["input"])) == {"units"}]
    assert len(source_calls) == 4
    units = []
    for call in source_calls:
        unit, = json.loads(call["input"])["units"]
        question, = call["questions"]
        assert question["name"] == unit["name"]
        units.append(unit)
    assert [unit["name"] for unit in units] == [f"grounded_{i}" for i in range(4)]
    assert [unit["unit_id"] for unit in units] == [
        "b0.content.0", "b0.content.1", "b1.content.0", "b1.content.1",
    ]
    assert [unit["basis"] for unit in units] == ["source", "source", "inference", "inference"]
    assert units[2]["text"] == CLAIMS[case]
    expected_excerpt = BRANCH_SOURCE if case == "static_binding" else CAPTURE_SOURCE[20:94]
    projected_source = units[0]["cited_sources"]
    for unit in units:
        assert unit["refs"] == ["e1"]
        assert unit["cited_sources"] == projected_source
        source, = unit["cited_sources"]
        assert source["id"] == "e1"
        assert source["file_id"] == (BRANCH_FILE_ID if case == "static_binding" else CAPTURE_FILE_ID)
        assert source["excerpt"] == expected_excerpt
        assert source["selection"] == {
            "start": 0 if case == "static_binding" else 20,
            "end": 89 if case == "static_binding" else 94, "cell_ids": [],
        }
        assert source["is_partial"] is (case != "static_binding")
        if case != "static_binding":
            assert source["uri"] == CAPTURE_URI
            assert source["snapshot_id"] == "snapshot:8047b4827243661442eed399015d4eda0912095012fce20cc3877348593190ee"
            assert source["element_extent"] == {
                "kind": "python_file", "text_length": 94, "omitted_prefix": True, "omitted_suffix": False,
            }

    probabilities = {}
    refused_names = set()
    source_observations = []
    with capture_llm_diagnostics():
        request_call_budget("post_synthesis_validation").max_calls = len(source_calls)
        for call in source_calls:
            judgment = live_registry.decisions_client.evaluate(
                input=call["input"], questions=call["questions"], stage="post_synthesis_validation",
            )
            probabilities.update(judgment.probabilities)
            refused_names.update(judgment.refused_names)
            source_observations.append({
                "input": call["input"], "questions": call["questions"],
                "probabilities": judgment.probabilities, "refused_names": sorted(judgment.refused_names),
            })
    replay = DecisionsBoundary(probabilities=probabilities, refused_names=refused_names)
    updates = make_post_synthesis_validation_node(False, decisions_client=replay, pass_threshold=0.8)(state)
    replay_source_calls = [call for call in replay.requests if set(json.loads(call["input"])) == {"units"}]
    for actual, captured in zip(replay_source_calls, source_calls, strict=True):
        assert actual["input"] == captured["input"]
        assert actual["questions"] == captured["questions"]
    expected_supported = case not in {"actual_bad", "affirmative_only"}
    observation = {
        "case": case, "expected_supported": expected_supported,
        "source_calls": source_observations,
        "probabilities": probabilities, "refused_names": sorted(refused_names),
        "node_needs_retry": updates["retry"].needs_retry,
        "node_response_kind": updates.get("response", state["response"]).kind,
        "validation_events": updates["debug"].validation_events,
    }
    record = tmp_path / "decisions.json"
    record.write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**observation, "record": str(record)}, ensure_ascii=False))

    assert refused_names == set(), str(record)
    assert (probabilities["grounded_2"] >= 0.8) is expected_supported, str(record)
    assert all(probabilities[f"grounded_{i}"] >= 0.8 for i in (0, 1, 3)), str(record)
    assert updates["retry"].needs_retry is not expected_supported, str(record)
    assert state["response"].result == original
    if expected_supported:
        assert updates["response"].kind == "answer", str(record)
        assert updates["response"].result.content == original.content
        assert updates["response"].result.content_hash == original.content_hash
        assert updates["response"].result.citations == original.citations
        assert updates["response"].result.semantic_validation.passed, str(record)
    else:
        assert "b1.content.0" in updates["retry"].retrieval_feedback
