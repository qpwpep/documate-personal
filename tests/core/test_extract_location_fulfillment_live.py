"""Opt-in request fulfillment checks for exact source excerpts and their locations.

From the repository root in Git Bash:
    RUN_LIVE_EXTRACT_LOCATION_FULFILLMENT=1 uv run --locked pytest -q -s tests/core/test_extract_location_fulfillment_live.py

Five fixed candidates retain the observed branch.py query, complete 89-character
source and extraction instruction. The public post-validation node constructs
each request through the existing provider HTTP fake. Only the unchanged body
call containing request_fulfilled reaches the configured Decisions provider,
once per case. Its judgments are replayed through the public node without an
additional API call. The default suite skips these five paid calls. No planner,
retrieval, synthesis, save or notification API runs. Diagnostic JSON is written
inside pytest's temporary directories and printed for the caller's local log.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import pytest

from src.core.answer_schema import AnswerDocument, export_answer_text, finalize_answer, iter_content_units
from src.core.answer_schema.source_locations import SOURCE_LOCATION_PLACEHOLDER
from src.core.contracts import GraphState
from src.infra.llm import LLMRegistry, build_llm_registry
from src.infra.llm_boundary import capture_llm_diagnostics, request_call_budget
from src.infra.settings import get_settings
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core import test_post_source_quote_live
from tests.core.test_decisions_validation import DecisionsBoundary


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_EXTRACT_LOCATION_FULFILLMENT") != "1",
    reason="Set RUN_LIVE_EXTRACT_LOCATION_FULFILLMENT=1 to call the configured Decisions provider.",
)

BODY_INSTRUCTION = (
    "첨부한 branch.py에서 실제 NumPy의 numpy.array 호출 문장만 원문 그대로 인용하고 줄 번호를 제시한다. "
    "다른 객체의 array 호출은 제외한다."
)
CODE_QUOTE = "    result = np.array([1, 2])"
Case = Literal["location_first", "code_first", "missing_quote", "missing_location", "altered_quote"]


def _state(case: Case) -> GraphState:
    state = test_post_source_quote_live._state("exact-source-and-location")
    contract = state["runtime"].request_contract
    state["runtime"] = state["runtime"].model_copy(update={
        "request_contract": contract.model_copy(update={
            "body": contract.body.model_copy(update={"instruction": BODY_INSTRUCTION}),
        }),
    })
    evidence, = state["response"].evidence_packet
    quote = {
        "type": "code", "language": "python", "inline": False,
        "content": {"text": "    result = np.array([9])" if case == "altered_quote" else CODE_QUOTE,
                    "basis": "source", "refs": [evidence.id], "source_location": None},
    }
    location = {
        "type": "paragraph", "content": [{
            "text": SOURCE_LOCATION_PLACEHOLDER, "basis": "source", "refs": [evidence.id],
            "source_location": {"ref": evidence.id, "start": 59, "end": 89},
        }],
    }
    if case == "missing_quote":
        blocks = [location]
    elif case == "missing_location":
        blocks = [quote]
    elif case == "location_first":
        blocks = [location, quote]
    else:
        blocks = [quote, location]
    result = finalize_answer(AnswerDocument.model_validate({"blocks": blocks}), [evidence], retrieval_required=True)
    state["response"] = state["response"].model_copy(update={"result": result})
    displayed = export_answer_text(result)
    assert SOURCE_LOCATION_PLACEHOLDER not in displayed
    assert ("branch.py:6" in displayed) is (case != "missing_location")
    assert (CODE_QUOTE in displayed) is (case not in {"missing_quote", "altered_quote"})
    return state


@pytest.fixture(scope="module")
def live_registry() -> LLMRegistry:
    return build_llm_registry(get_settings().model_copy(update={"verbose": False}))


@pytest.mark.parametrize("case", [
    "location_first", "code_first", "missing_quote", "missing_location", "altered_quote",
])
def test_live_extract_requires_the_unchanged_quote_and_its_location_in_either_order(
    case: Case, live_registry: LLMRegistry, tmp_path: Path,
) -> None:
    state = _state(case)
    original = state["response"].result.model_copy(deep=True)
    capture = DecisionsBoundary()
    make_post_synthesis_validation_node(False, decisions_client=capture, pass_threshold=0.8)(state)
    body_call, = [call for call in capture.requests
                  if any(question["name"] == "request_fulfilled" for question in call["questions"])]
    assert [question["name"] for question in body_call["questions"]] == ["request_fulfilled"]
    payload = json.loads(body_call["input"])
    assert payload["latest_user_request"] == test_post_source_quote_live.REQUEST
    assert payload["confirmed_request"]["body"]["instruction"] == BODY_INSTRUCTION
    displayed = AnswerDocument.model_validate(payload["rendered_response"]["body"])
    assert [block.model_dump(exclude={"content"}) for block in displayed.blocks] == [
        block.model_dump(exclude={"content"}) for block in original.content.blocks
    ]
    for (actual_path, actual), (original_path, original_unit) in zip(
        iter_content_units(displayed), iter_content_units(original.content), strict=True,
    ):
        assert actual_path == original_path
        assert actual.model_dump(exclude={"refs", "source_location"}) == original_unit.model_dump(
            exclude={"refs", "source_location"},
        )
        assert actual.refs == ["e1"]
        if original_unit.source_location is not None:
            assert actual.source_location.model_dump() == {"ref": "e1", "start": 59, "end": 89}
        else:
            assert actual.source_location is None
    assert len(payload["units"]) == (1 if case in {"missing_quote", "missing_location"} else 2)
    for unit in payload["units"]:
        source, = unit["cited_sources"]
        assert source["excerpt"] == test_post_source_quote_live.SOURCE
        assert source["selection"] == {"start": 0, "end": 89, "cell_ids": []}
        assert source["is_partial"] is False
        assert unit["refs"] == [source["id"]]
        if unit["source_location"] is not None:
            assert unit["text"] == "branch.py:6"
            assert unit["source_location"] == {"ref": source["id"], "start": 59, "end": 89}

    with capture_llm_diagnostics():
        request_call_budget("post_synthesis_validation").max_calls = 1
        judgment = live_registry.decisions_client.evaluate(
            input=body_call["input"], questions=body_call["questions"], stage="post_synthesis_validation",
        )
    replay = DecisionsBoundary(probabilities=judgment.probabilities, refused_names=judgment.refused_names)
    updates = make_post_synthesis_validation_node(False, decisions_client=replay, pass_threshold=0.8)(state)
    replay_call, = [call for call in replay.requests
                    if any(question["name"] == "request_fulfilled" for question in call["questions"])]
    assert replay_call["input"] == body_call["input"]
    assert replay_call["questions"] == body_call["questions"]
    expected_fulfilled = case in {"location_first", "code_first"}
    observation = {
        "case": case, "expected_fulfilled": expected_fulfilled,
        "input": payload, "questions": body_call["questions"],
        "probabilities": judgment.probabilities, "refused_names": sorted(judgment.refused_names),
        "node_needs_retry": updates["retry"].needs_retry,
        "node_response_kind": updates.get("response", state["response"]).kind,
        "validation_events": updates["debug"].validation_events,
    }
    record = tmp_path / "decisions.json"
    record.write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**observation, "record": str(record)}, ensure_ascii=False))

    assert judgment.refused_names == set(), str(record)
    assert (judgment.probabilities["request_fulfilled"] >= 0.8) is expected_fulfilled, str(record)
    assert updates["retry"].needs_retry is not expected_fulfilled, str(record)
    if expected_fulfilled:
        assert updates["response"].kind == "answer", str(record)
        assert updates["response"].result.content == original.content
        assert updates["response"].result.citations == original.citations
        assert updates["response"].result.semantic_validation.passed, str(record)
    else:
        assert "semantic_request_fulfillment" in updates["retry"].retrieval_feedback
