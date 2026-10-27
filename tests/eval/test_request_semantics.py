"""Request semantics and attempt coverage are mandatory grading evidence."""
from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_openai import ChatOpenAI

from src.core.planner_schema import InitialPlannerOutput
from src.eval.request_contract_eval import load_cases, load_policy, make_live_observer
from src.eval.request_contract_scoring import score_observation, summarize_results
from src.eval.request_contract_semantics import (
    BodySemanticAssessor, SemanticChecks, semantic_input_hash,
)
from src.infra.llm_boundary import bind_structured_output


def _case(query="Explain the cache eviction policy", *, kind="compose"):
    return {
        "id": "semantic-regression",
        "cohort": "known",
        "query": query,
        "context": {"history": [], "previous_answer": None, "pending": None},
        "expected": {
            "actions": {"save_text": "not_requested", "slack_notify": "not_requested"},
            "body": {"kind": kind, "source": None},
            "requirements": {"required": []},
            "preferences": [],
            "disposition": "answer",
            "blocker_slots": [],
            "state": {"relation": "new", "target_request": "none"},
            "critical_context": {"canceled_request": False},
            "recipient": {"state": "omitted"},
        },
    }


def _policy():
    return {
        "normalization": {"constraint_aliases": {}},
        "case_count": 1, "known_case_count": 1, "repeats": 3,
        "qualification": {"minimum_overall_rate": 0.95, "minimum_contract_acceptance_rate": 0.99},
        "policy_id": "semantic-regression", "scope_limit": "test",
    }


def _checks(**failures):
    return {
        name: {"status": "fail" if name in failures else "pass",
               "reason": failures.get(name, "The requested meaning is preserved.")}
        for name in ("subject", "operation", "target_language", "constraints")
    }


def _observation(case, *, instruction=None):
    contract = {
        "request_id": "fresh-request", "revision": 1, "relation": "new", "target_request_id": None,
        "body": {"kind": case["expected"]["body"]["kind"], "instruction": instruction or case["query"]},
        "actions": {name: {"intent": intent} for name, intent in case["expected"]["actions"].items()},
        "answer": {"content": [], "format": [], "preferences": []},
        "slack_recipient": {"state": "omitted"}, "missing_info": [],
    }
    observed = {
        "canonical": contract, "contract_accepted": True, "disposition": "answer",
        "product_execution": {"status": "completed", "failure_origin": None},
        "context": {"current_turn_id": "u1", "user_turns": {"u1": case["query"]}},
        "raw_candidates": [{"attempt": 1, "status": "available", "candidate": deepcopy(contract)}],
        "body_semantics": {"status": "evaluated", "checks": _checks()},
    }
    observed["body_semantics"]["input_hash"] = semantic_input_hash(case, observed)
    return observed


@pytest.mark.parametrize(
    ("query", "instruction", "criterion"),
    [
        ("Explain the cache eviction policy", "Describe the history of jazz.", "subject"),
        ("Translate the selected phrase into Korean", "Translate the selected phrase into French.", "target_language"),
        ("Summarize the uploaded article", "Critique the uploaded article and add your opinion.", "operation"),
        ("Explain the dependency graph", "Explain only dependencies of Python packages released in 2025.", "constraints"),
    ],
)
def test_semantic_mismatch_cannot_pass_structurally_valid_contract(query, instruction, criterion):
    case = _case(query)
    observed = _observation(case, instruction=instruction)
    observed["body_semantics"]["checks"] = _checks(**{criterion: "The planned meaning differs from the user request."})
    score = score_observation(case, observed, _policy())
    assert score["overall"] is False
    assert score["status"] == "product_failure"
    assert score["dimensions"]["body_semantics"]["passed"] is False


@pytest.mark.parametrize("assessment", [None, {"status": "unavailable"}, {"status": "invalid"}, {"status": "not_applicable"}])
def test_missing_or_invalid_semantic_evaluation_cannot_be_success(assessment):
    case = _case()
    observed = _observation(case)
    observed["body_semantics"] = assessment
    score = score_observation(case, observed, _policy())
    assert score["overall"] is False
    assert score["status"] in {"evaluation_failure", "observation_insufficient"}
    assert score["dimensions"]["body_semantics"]["passed"] is False


def test_semantic_evidence_passes_without_literal_keyword_matching():
    case = _case("Explain least recently used cache eviction")
    observed = _observation(case, instruction="Describe how LRU removes an entry based on its last access time.")
    assert score_observation(case, observed, _policy())["overall"] is True


@pytest.mark.parametrize(("case_id", "description"), [
    ("quotation_en", "Translate the selected phrase into Korean."),
    ("quotation_en", "Render the selected phrase in Korean."),
    ("beginner_preference", "Explain for a beginner."),
    ("beginner_preference", "Assume no prior knowledge and define unfamiliar terms."),
])
def test_equivalent_free_form_conditions_use_the_same_semantic_evidence(case_id, description):
    case = next(case for case in load_cases() if case["id"] == case_id)
    observed = _observation(case)
    if case["expected"]["body"]["source"] is not None:
        selected = case["expected"]["body"]["source"]["text"]
        start = case["query"].index(selected)
        observed["canonical"]["body"]["source"] = {
            "turn_id": "u1", "text": selected, "start": start, "end": start + len(selected),
        }
        observed["canonical"]["answer"]["content"] = [
            {"kind": "custom", "mode": "required", "description": description},
        ]
    else:
        observed["canonical"]["answer"]["preferences"] = [description]
    observed["body_semantics"]["input_hash"] = semantic_input_hash(case, observed)

    assert score_observation(case, observed, load_policy())["overall"] is True


def test_free_form_conditions_on_a_copy_still_require_valid_semantic_evidence():
    case = _case("Copy the selected answer", kind="copy_answer")
    observed = _observation(case)
    observed["canonical"]["answer"]["content"] = [
        {"kind": "custom", "mode": "required", "description": "Keep the named terms unchanged."},
    ]
    observed["body_semantics"] = None
    score = score_observation(case, observed, _policy())
    assert score["overall"] is False
    assert score["status"] == "observation_insufficient"
    assert score["dimensions"]["body_semantics"]["status"] == "unavailable"


def test_semantic_pass_does_not_authorize_an_unrequested_hard_format():
    case = _case("Explain the cache eviction policy")
    observed = _observation(case)
    observed["canonical"]["answer"]["format"] = [{"kind": "ordered_list", "mode": "required"}]
    observed["body_semantics"]["input_hash"] = semantic_input_hash(case, observed)
    score = score_observation(case, observed, _policy())
    assert score["overall"] is False
    assert score["dimensions"]["requirements"]["errors"] == [
        "unrequested_hard_constraint:('ordered_list', 'required', None)",
    ]


@pytest.mark.parametrize(("group", "condition"), [
    ("content", {"kind": "custom", "mode": "required", "description": "Add a conclusion in French."}),
    ("preferences", "Assume the reader already knows the terminology."),
])
def test_unrequested_content_or_lost_preferences_fail_the_general_meaning_check(group, condition):
    case = _case("Explain the cache policy for a reader without prior knowledge")
    case["expected"]["preferences"] = ["no prior knowledge"]
    observed = _observation(case)
    observed["canonical"]["answer"][group] = [condition]
    observed["body_semantics"]["checks"] = _checks(
        constraints="The condition changes the requested language or audience level.",
    )
    observed["body_semantics"]["input_hash"] = semantic_input_hash(case, observed)
    score = score_observation(case, observed, _policy())
    assert score["overall"] is False
    assert score["status"] == "product_failure"
    assert score["dimensions"]["body_semantics"]["errors"] == ["semantic_mismatch:constraints"]


def test_unauthorized_action_in_repair_is_critical_even_when_product_rejects_it():
    case = _case()
    case["expected"]["critical_context"]["canceled_request"] = True
    observed = _observation(case)
    unauthorized = deepcopy(observed["canonical"])
    unauthorized["actions"]["slack_notify"]["intent"] = "requested"
    observed["raw_candidates"] = [
        {"attempt": 1, "status": "missing_contract", "candidate": None},
        {"attempt": 2, "status": "available", "candidate": unauthorized},
    ]
    observed["canonical"] = None
    observed["contract_accepted"] = False
    score = score_observation(case, observed, _policy())
    assert {"unauthorized_intent_escalation", "canceled_action_revival"} <= set(score["critical_failures"])
    assert score["critical_attempts"] == [
        {"attempt": 2, "actions": ["slack_notify"], "failures": ["unauthorized_intent_escalation", "canceled_action_revival"]}
    ]


def test_unobserved_attempt_cannot_count_as_zero_risk_success():
    case = _case()
    observed = _observation(case)
    observed["raw_candidates"].append({"attempt": 2, "status": "unavailable", "candidate": None})
    score = score_observation(case, observed, _policy())
    assert score["overall"] is False
    assert score["status"] == "observation_insufficient"


def test_summary_distinguishes_product_failure_from_evaluation_and_observation_failure():
    case = _case()
    mismatched = _observation(case)
    mismatched["body_semantics"]["checks"] = _checks(subject="Another subject.")
    unavailable = _observation(case)
    unavailable["body_semantics"] = {"status": "unavailable"}
    invalid = _observation(case)
    invalid["body_semantics"] = {"status": "invalid"}
    results = [
        {"case_id": case["id"], "score": score_observation(case, observed, _policy())}
        for observed in (mismatched, unavailable, invalid)
    ]
    summary = summarize_results(results, [case], _policy(), repeats=3)
    assert summary["qualified"] is False
    assert summary["status_counts"] == {
        "product_failure": 1, "observation_insufficient": 1, "evaluation_failure": 1,
    }
    assert summary["scoring_contract_version"] == "request-contract-semantics-v3"
    assert {row["score"]["scoring_contract_version"] for row in results} == {
        summary["scoring_contract_version"],
    }


def _http_assessor(handler):
    # Real ChatOpenAI serialization and provider-response parsing; only HTTP is fake.
    model = ChatOpenAI(
        model="semantic-test", api_key="test-key", base_url="https://semantic.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        use_responses_api=False, max_retries=0,
    ).bind(response_format={
        "type": "json_schema",
        "json_schema": {"name": "request_body_semantics", "strict": True, "schema": SemanticChecks.model_json_schema()},
    })
    return BodySemanticAssessor(model, definition={"model": "semantic-test", "digest": "test-definition"})


def _response(content):
    return httpx.Response(200, json={
        "id": "semantic-completion", "object": "chat.completion", "created": 1,
        "model": "semantic-test-v1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    })


def test_http_assessor_preserves_original_request_and_rejects_unrelated_plan():
    case = _case("Explain the cache eviction policy")
    observed = _observation(case, instruction="Describe the history of jazz.")
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return _response(json.dumps(_checks(subject="The plan replaces cache eviction with music history.")))

    observed["body_semantics"] = _http_assessor(respond)(case, observed)
    score = score_observation(case, observed, _policy())
    assert score["status"] == "product_failure"
    assert score["dimensions"]["body_semantics"]["errors"] == ["semantic_mismatch:subject"]
    payload = json.loads(requests[0]["messages"][1]["content"])
    assert payload["user_query"] == case["query"]
    assert payload["candidate"]["body"]["instruction"] == "Describe the history of jazz."
    assert payload["oracle"] == case["expected"]
    assert "id" not in payload  # Fixture identifiers never influence the assessor.
    audit = observed["body_semantics"]
    assert audit["request"][1]["content"] == requests[0]["messages"][1]["content"]
    assert audit["raw_response"]["response_metadata"]["model_name"] == "semantic-test-v1"
    assert audit["raw_response"]["usage_metadata"]["total_tokens"] == 30
    assert audit["input_hash"] == semantic_input_hash(case, observed)


@pytest.mark.parametrize("content", [
    "not JSON",
    json.dumps({"subject": {"status": "pass", "reason": "Correct."}}),
    json.dumps({**_checks(), "subject": {"status": "not_applicable", "reason": "Skip topic."}}),
    json.dumps({**_checks(), "operation": {"status": True, "reason": "Correct."}}),
    json.dumps({**_checks(), "constraints": {"status": "pass", "reason": "  "}}),
])
def test_invalid_provider_assessment_is_evaluation_failure(content):
    case = _case()
    observed = _observation(case)
    observed["body_semantics"] = _http_assessor(lambda request: _response(content))(case, observed)
    score = score_observation(case, observed, _policy())
    assert observed["body_semantics"]["status"] == "invalid"
    assert score["status"] == "evaluation_failure"
    assert score["overall"] is False


def test_provider_failure_is_observation_insufficient_not_product_mismatch():
    case = _case()
    observed = _observation(case)
    observed["body_semantics"] = _http_assessor(
        lambda request: httpx.Response(503, json={"error": {"message": "unavailable", "type": "server_error"}})
    )(case, observed)
    score = score_observation(case, observed, _policy())
    assert observed["body_semantics"]["status"] == "unavailable"
    assert score["status"] == "observation_insufficient"
    assert score["failure_categories"] == ["observation_insufficient"]


def test_assessment_cannot_be_reused_for_a_different_candidate():
    case = _case()
    observed = _observation(case)
    observed["canonical"]["body"]["instruction"] = "Discuss a different subject."
    score = score_observation(case, observed, _policy())
    assert score["overall"] is False
    assert score["status"] == "evaluation_failure"


def test_raw_invalid_shape_does_not_hide_an_embedded_unauthorized_action():
    case = _case()
    observed = _observation(case)
    candidate = {"actions": {"save_text": {"intent": "requested"}}}
    observed["raw_candidates"].append({"attempt": 2, "status": "invalid_contract", "candidate": candidate})
    assert score_observation(case, observed, _policy())["critical_failures"] == ["unauthorized_intent_escalation"]


def test_successful_repair_of_missing_contract_keeps_all_attempts_observed():
    case = _case()
    observed = _observation(case)
    observed["raw_candidates"] = [
        {"attempt": 1, "status": "missing_contract", "candidate": None},
        {"attempt": 2, "status": "available", "candidate": deepcopy(observed["canonical"])},
    ]
    score = score_observation(case, observed, _policy())
    assert score["status"] == "passed"
    assert score["dimensions"]["attempt_coverage"]["passed"] is True


def test_repair_cannot_assert_zero_raw_action_risk_for_unparseable_earlier_attempt():
    case = _case()
    observed = _observation(case)
    observed["raw_candidates"] = [
        {"attempt": 1, "status": "invalid_json", "candidate": None},
        {"attempt": 2, "status": "available", "candidate": deepcopy(observed["canonical"])},
    ]
    score = score_observation(case, observed, _policy())
    assert score["status"] == "observation_insufficient"
    assert score["overall"] is False
    assert score["dimensions"]["attempt_coverage"]["passed"] is False


def test_duplicate_semantic_keys_cannot_overwrite_a_failed_check():
    case = _case()
    observed = _observation(case)
    response = json.dumps(_checks())
    response = '{"subject":{"status":"fail","reason":"Wrong subject."},' + response[1:]
    observed["body_semantics"] = _http_assessor(lambda request: _response(response))(case, observed)
    assert observed["body_semantics"]["status"] == "invalid"
    assert score_observation(case, observed, _policy())["status"] == "evaluation_failure"


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_incomplete_or_filtered_provider_result_cannot_pass_with_valid_json(finish_reason):
    case = _case()
    observed = _observation(case)

    def respond(request):
        payload = json.loads(_response(json.dumps(_checks())).content)
        payload["choices"][0]["finish_reason"] = finish_reason
        return httpx.Response(200, json=payload)

    observed["body_semantics"] = _http_assessor(respond)(case, observed)
    assert observed["body_semantics"]["status"] == "invalid"
    assert score_observation(case, observed, _policy())["overall"] is False


def test_provider_refusal_cannot_pass_with_valid_json():
    case = _case()
    observed = _observation(case)

    def respond(request):
        payload = json.loads(_response(json.dumps(_checks())).content)
        payload["choices"][0]["message"]["refusal"] = "I cannot evaluate this request."
        return httpx.Response(200, json=payload)

    observed["body_semantics"] = _http_assessor(respond)(case, observed)
    assert observed["body_semantics"]["status"] == "invalid"
    assert score_observation(case, observed, _policy())["overall"] is False


@pytest.mark.parametrize(
    ("case_id", "body", "criterion"),
    [
        ("mention_en", {"kind": "compose", "instruction": "Explain how to bake a cake."}, "subject"),
        ("quotation_en", {
            "kind": "transform_input",
            "source": {"turn_id": "u1", "quote": "send this to Slack"},
            "instruction": "Translate into French.",
        }, "target_language"),
    ],
)
def test_real_observer_and_scorer_reject_wrong_subject_or_translation_with_http_fakes(monkeypatch, case_id, body, criterion):
    # Read one frozen oracle as test input. No batch evaluator or benchmark CLI runs.
    case = next(case for case in load_cases() if case["id"] == case_id)
    plan = InitialPlannerOutput.model_validate({
        "use_retrieval": False, "tasks": [],
        "request_contract": {"body": body, "slack_recipient": {"state": "omitted"}},
    }).model_dump(mode="json")
    with httpx.Client(transport=httpx.MockTransport(lambda request: _response(json.dumps(plan)))) as client:
        planner = bind_structured_output(ChatOpenAI(
            model="planner-test", api_key="test-key", base_url="https://planner.invalid/v1",
            http_client=client, use_responses_api=False, max_retries=0,
        ), InitialPlannerOutput, name="PlannerOutput")
        monkeypatch.setattr("src.infra.llm.build_llm_registry", lambda settings: SimpleNamespace(llm_planner=planner))
        assessor = _http_assessor(lambda request: _response(json.dumps(_checks(**{
            criterion: "The user request and planned instruction disagree.",
        }))))
        observation = make_live_observer(
            SimpleNamespace(openai_api_key="test-key"), body_semantic_assessor=assessor,
        )(case)
    score = score_observation(case, observation, _policy())
    assert observation["contract_accepted"] is True
    assert score["dimensions"]["body"]["passed"] is True
    assert score["dimensions"]["attempt_coverage"]["passed"] is True
    assert score["dimensions"]["body_semantics"]["errors"] == [f"semantic_mismatch:{criterion}"]
    assert score["status"] == "product_failure"
    assert score["overall"] is False


def test_incomplete_evaluation_is_not_absorbed_by_allowed_product_failure_rate():
    case = _case()
    observed = _observation(case)
    passed = score_observation(case, observed, _policy())
    observed["body_semantics"] = {"status": "unavailable"}
    incomplete = score_observation(case, observed, _policy())
    cases = [{**case, "id": f"synthetic-{i}", "cohort": "extension"} for i in range(100)]
    rows = [{"case_id": item["id"], "score": passed} for item in cases for _ in range(3)]
    rows[-1]["score"] = incomplete
    policy = {**_policy(), "case_count": 100, "known_case_count": 0}
    summary = summarize_results(rows, cases, policy, repeats=3)
    assert summary["gates"]["overall_at_least_95_percent"] is True
    assert summary["gates"]["contract_acceptance_at_least_99_percent"] is True
    assert summary["gates"]["complete_evaluation"] is False
    assert summary["qualified"] is False


@pytest.mark.parametrize("failure_boundary", ["registry", "case_state"])
def test_evaluation_setup_failure_is_not_a_product_failure(monkeypatch, failure_boundary):
    case = _case()

    def invalid_configuration(*args, **kwargs):
        raise ValueError("invalid evaluator model settings")

    target = (
        "src.infra.llm.build_llm_registry"
        if failure_boundary == "registry" else "src.eval.request_contract_eval.build_case_state"
    )
    monkeypatch.setattr(target, invalid_configuration)
    observation = make_live_observer(SimpleNamespace(openai_api_key="test-key"))(case)
    score = score_observation(case, observation, _policy())
    assert observation["product_execution"] == {
        "status": "not_started", "failure_origin": "evaluation_setup",
    }
    assert observation["model_calls"] == []
    assert score["status"] == "evaluation_failure"
    assert score["failure_categories"] == ["evaluation_failure"]
    assert score["dimensions"]["validity"]["passed"] is None
    assert score["dimensions"]["body"]["status"] == "not_evaluated"
    assert score["critical_assessment"] == "not_evaluated"
    assert score["overall"] is False


def test_unexpected_observer_exception_does_not_assert_a_product_failure():
    from src.eval import request_contract_eval

    def unavailable_observer(case):
        raise RuntimeError("observation serialization failed")

    case = _case()
    observation = request_contract_eval.observe_case(case, unavailable_observer)
    score = score_observation(case, observation, _policy())
    assert observation["product_execution"] == {"status": "unknown", "failure_origin": "observer"}
    assert score["status"] == "evaluation_failure"
    assert "product_failure" not in score["failure_categories"]
    assert score["dimensions"]["validity"]["status"] == "not_evaluated"
    assert score["overall"] is False


def test_actual_product_call_failure_is_still_a_product_failure(monkeypatch):
    case = _case()
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        401, json={"error": {"message": "Invalid credential", "type": "authentication_error"}},
    ))) as client:
        planner = bind_structured_output(ChatOpenAI(
            model="planner-test", api_key="test-key", base_url="https://planner.invalid/v1",
            http_client=client, use_responses_api=False, max_retries=0,
        ), InitialPlannerOutput, name="PlannerOutput")
        monkeypatch.setattr("src.infra.llm.build_llm_registry", lambda settings: SimpleNamespace(llm_planner=planner))
        observation = make_live_observer(SimpleNamespace(openai_api_key="test-key"))(case)
    score = score_observation(case, observation, _policy())
    assert observation["product_execution"] == {"status": "failed", "failure_origin": "product"}
    assert len(observation["model_calls"]) == 1
    assert score["status"] == "product_failure"
    assert score["failure_categories"] == ["product_failure", "observation_insufficient"]
    assert score["dimensions"]["validity"]["passed"] is False


def test_setup_failure_summary_keeps_product_dimensions_unobserved(monkeypatch):
    def invalid_configuration(settings):
        raise ValueError("invalid evaluator model settings")

    monkeypatch.setattr("src.infra.llm.build_llm_registry", invalid_configuration)
    case = _case()
    observation = make_live_observer(SimpleNamespace(openai_api_key="test-key"))(case)
    score = score_observation(case, observation, _policy())
    summary = summarize_results([{"case_id": case["id"], "score": score}], [case], _policy(), repeats=1)
    assert summary["status_counts"] == {"evaluation_failure": 1}
    assert summary["dimension_failures"] == {"evaluation_execution": 1}
    assert summary["dimensions_not_evaluated"]["validity"] == 1
    assert summary["gates"]["zero_critical_failures"] is False
    assert summary["gates"]["complete_evaluation"] is False
