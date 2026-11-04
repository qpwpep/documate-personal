"""Target identification and supported analysis independently gate source approval."""
import json

import pytest

from src.core.contracts.graph_state import DecisionCacheEntry
from src.core.decisions import decision_fingerprint
from src.core.request_bodies import ExtractBody
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from src.runtime.nodes.validation.assessment import assess_validation
from src.runtime.nodes.validation.semantic import assess_post_semantics
from src.runtime.nodes.validation.snapshot import collect_validation_snapshot
from tests.core.test_decisions_prepared_packets import distinct_packet_state
from tests.core.test_decisions_validation import DecisionsBoundary, _state
from tests.synthesis_fixtures import synthesis_excerpt_limits


@pytest.mark.parametrize("stage", ["pre", "post"])
@pytest.mark.parametrize("target,analysis,refused,failed", [
    (0.8, 0.8, None, False),
    (0.79, 0.99, None, True),
    (0.99, 0.79, None, True),
    (0.99, 0.99, "target_identified", True),
    (0.99, 0.99, "target_analysis_supported", True),
])
def test_each_requirement_dimension_can_block_preparation_and_delivery(stage, target, analysis, refused, failed):
    state = _state()
    probabilities = {f"{base}_{dimension}": probability
                     for base in ("requirement_0", "compact_requirement_0")
                     for dimension, probability in (("target_identified", target), ("target_analysis_supported", analysis))}
    boundary = DecisionsBoundary(probabilities, refused_names=(
        [f"requirement_0_{refused}", f"compact_requirement_0_{refused}"] if refused else []
    ))
    if stage == "pre":
        node = make_pre_synthesis_validation_node(False, decisions_client=boundary,
                                                  excerpt_limits=synthesis_excerpt_limits())
    else:
        node = make_post_synthesis_validation_node(False, decisions_client=boundary)

    state.update(node(state))

    assert state["retry"].needs_retry is failed
    assert state["retry"].failed_requirement_ids == (["path_exists"] if failed else [])
    questions = [question for call in boundary.requests if "requirements" in json.loads(call["input"])
                 for question in call["questions"]]
    assert {question["name"] for question in questions} == {
        "requirement_0_target_identified", "requirement_0_target_analysis_supported",
    }
    if stage == "pre":
        assert state["retrieval"].prepared_evidence.normal.failed_requirement_ids == (["path_exists"] if failed else [])


@pytest.mark.parametrize("refused", [None, "target_identified", "target_analysis_supported"])
def test_post_requirement_report_preserves_each_actual_probability_and_refusal(refused):
    snapshot, _ = collect_validation_snapshot(_state(), max_turns=6)
    assessment = assess_validation(snapshot)
    boundary = DecisionsBoundary({"requirement_0_target_identified": 0.91,
                                  "requirement_0_target_analysis_supported": 0.83},
                                 refused_names=[f"requirement_0_{refused}"] if refused else [])

    assess_post_semantics(snapshot, assessment, decisions_client=boundary, threshold=0.8, attempt=1)

    report = assessment.checked_result.semantic_validation
    checks = {check.name: check for check in report.checks if check.name.startswith("requirement_")}
    assert set(checks) == {"requirement_target_identified:path_exists", "requirement_target_analysis_supported:path_exists"}
    for dimension, probability in (("target_identified", 0.91), ("target_analysis_supported", 0.83)):
        check = checks[f"requirement_{dimension}:path_exists"]
        assert check.probability == (None if refused == dimension else probability)
        assert check.status == ("refused" if refused == dimension else "evaluated")
    assert report.passed is (refused is None)


def test_requirement_dimensions_reuse_identical_packets_and_cache_each_criterion_separately(monkeypatch):
    from src.runtime.nodes.validation import prepared_semantics

    state = _state()
    boundary = DecisionsBoundary()
    node = make_pre_synthesis_validation_node(False, decisions_client=boundary,
                                              excerpt_limits=synthesis_excerpt_limits())
    state.update(node(state))
    requirement_calls = [call for call in boundary.requests if "requirements" in json.loads(call["input"])]
    assert len(requirement_calls) == 1
    assert len(requirement_calls[0]["questions"]) == 2

    previous_count = len(boundary.requests)
    state.update(node(state))
    assert len(boundary.requests) == previous_count

    monkeypatch.setattr(prepared_semantics, "REQUIREMENT_CRITERIA", tuple(
        (name, instructions + " Check the required explanation." if name == "target_analysis_supported" else instructions)
        for name, instructions in prepared_semantics.REQUIREMENT_CRITERIA
    ))
    state.update(node(state))

    refreshed, = boundary.requests[previous_count:]
    assert [question["name"] for question in refreshed["questions"]] == ["requirement_0_target_analysis_supported"]
    original_entry, = json.loads(requirement_calls[0]["input"])["requirements"]
    refreshed_entry, = json.loads(refreshed["input"])["requirements"]
    assert refreshed_entry == original_entry
    assert original_entry["name"] == "requirement_0"
    assert state["retrieval"].prepared_evidence.normal.failed_requirement_ids == []


def test_supported_analysis_cannot_borrow_target_identification_from_another_packet(distinct_packet_state):
    boundary = DecisionsBoundary({
        "requirement_0_target_identified": 0.1,
        "requirement_0_target_analysis_supported": 0.99,
        "compact_requirement_0_target_identified": 0.99,
        "compact_requirement_0_target_analysis_supported": 0.1,
    })
    state = distinct_packet_state

    state.update(make_pre_synthesis_validation_node(False, decisions_client=boundary,
        excerpt_limits=synthesis_excerpt_limits())(state))

    prepared = state["retrieval"].prepared_evidence
    assert prepared.normal.failed_requirement_ids == ["path_exists"]
    assert prepared.compact is None
    assert state["retry"].needs_retry
    assert state["retry"].failed_requirement_ids == ["path_exists"]


def test_partial_support_cache_does_not_rename_the_shared_source_entry(monkeypatch):
    from src.runtime.nodes.validation import prepared_semantics

    state = _state()
    state["planner"].output.tasks[0].requirement.aspects = ["return value"]
    boundary = DecisionsBoundary()
    node = make_pre_synthesis_validation_node(False, decisions_client=boundary,
                                              excerpt_limits=synthesis_excerpt_limits())
    state.update(node(state))
    original = next(call for call in boundary.requests if "evidence" in json.loads(call["input"]))
    previous_count = len(boundary.requests)
    monkeypatch.setattr(prepared_semantics, "SUPPORT_INSTRUCTIONS",
                        prepared_semantics.SUPPORT_INSTRUCTIONS + " Preserve the requested aspect scope.")

    state.update(node(state))

    refreshed = next(call for call in boundary.requests[previous_count:] if "evidence" in json.loads(call["input"]))
    assert [question["name"] for question in refreshed["questions"]] == ["evidence_0_support_1"]
    original_entry, = json.loads(original["input"])["evidence"]
    refreshed_entry, = json.loads(refreshed["input"])["evidence"]
    assert refreshed_entry == original_entry
    assert original_entry["name"] == "evidence_0"


def test_neutral_entries_reassess_cached_judgments_from_the_old_question_named_input():
    boundary = DecisionsBoundary()
    node = make_pre_synthesis_validation_node(False, decisions_client=boundary,
                                              excerpt_limits=synthesis_excerpt_limits())
    node(_state())
    old_cache = []
    for call in boundary.requests:
        payload = json.loads(call["input"])
        collection = "evidence" if "evidence" in payload else "requirements"
        entry, = payload[collection]
        data = {"task": payload["tasks"][entry["task_id"]]}
        if collection == "evidence":
            data["source"] = payload["sources"][entry["source_id"]]
        else:
            data["sources"] = [payload["sources"][identity] for identity in entry["source_ids"]]
        for question in call["questions"]:
            # Persisted judgments used this pre-fix identity, without the model-facing entry shape.
            old_cache.append(DecisionCacheEntry(fingerprint=decision_fingerprint({
                "model": boundary.model_name, "entry": data,
                "instructions": question["instructions"].replace(
                    f"entry named {entry['name']}", "entry named <entry>",
                ),
            }), probability=0.01))
    state = _state()
    state["retrieval"].decision_cache = old_cache
    previous_count = len(boundary.requests)

    state.update(node(state))

    assert {question["name"] for call in boundary.requests[previous_count:]
            for question in call["questions"]} == {
        "evidence_0", "requirement_0_target_identified", "requirement_0_target_analysis_supported",
    }
    assert state["retrieval"].prepared_evidence.normal.failed_requirement_ids == []


@pytest.mark.parametrize("stage", ["pre", "post"])
def test_requirement_input_uses_the_same_json_format_as_source_and_body_checks(stage):
    boundary = DecisionsBoundary()
    node = (make_pre_synthesis_validation_node(False, decisions_client=boundary,
                                               excerpt_limits=synthesis_excerpt_limits()) if stage == "pre"
            else make_post_synthesis_validation_node(False, decisions_client=boundary))

    node(_state())

    call, = [call for call in boundary.requests if "requirements" in json.loads(call["input"])]
    assert call["input"] == json.dumps(json.loads(call["input"]), ensure_ascii=False)


@pytest.mark.parametrize("bound_extract", [False, True])
def test_pre_and_post_send_identical_requirement_inputs_for_the_same_selected_sources(bound_extract):
    pre = DecisionsBoundary()
    post = DecisionsBoundary()
    state = _state()
    if bound_extract:
        state["runtime"] = state["runtime"].model_copy(update={
            "request_contract": state["runtime"].request_contract.model_copy(update={
                "body": ExtractBody(requirement_ids=("path_exists",)),
            }),
        })

    make_pre_synthesis_validation_node(False, decisions_client=pre,
        excerpt_limits=synthesis_excerpt_limits())(state)
    make_post_synthesis_validation_node(False, decisions_client=post)(state)

    pre_call, = [call for call in pre.requests if "requirements" in json.loads(call["input"])]
    post_call, = [call for call in post.requests if "requirements" in json.loads(call["input"])]
    assert pre_call["input"] == post_call["input"]
    assert pre_call["questions"] == post_call["questions"]
    assert {question["name"] for question in pre_call["questions"]} == {
        "requirement_0_target_identified", "requirement_0_target_analysis_supported",
    }


def test_changed_model_input_serialization_invalidates_cached_judgments(monkeypatch):
    from src.runtime.nodes.validation import prepared_semantics

    state = _state()
    boundary = DecisionsBoundary()
    node = make_pre_synthesis_validation_node(False, decisions_client=boundary,
                                              excerpt_limits=synthesis_excerpt_limits())
    state.update(node(state))
    previous_count = len(boundary.requests)
    encode = prepared_semantics.encode_decision_input

    def reformatted_input(payload):
        return json.dumps(json.loads(encode(payload)), ensure_ascii=False, indent=2)

    monkeypatch.setattr(prepared_semantics, "encode_decision_input", reformatted_input)

    state.update(node(state))

    refreshed = boundary.requests[previous_count:]
    assert {question["name"] for call in refreshed for question in call["questions"]} == {
        "evidence_0", "requirement_0_target_identified", "requirement_0_target_analysis_supported",
    }
    assert all(call["input"] == json.dumps(json.loads(call["input"]), ensure_ascii=False, indent=2)
               for call in refreshed)
    assert state["retrieval"].prepared_evidence.normal.failed_requirement_ids == []
