import json

import httpx
from langchain_core.messages import HumanMessage
from openai import OpenAI

from src.core.contracts import DebugState, PlannerState, ResponseState, RetrievalState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input, normalize_graph_update
from src.core.planner_schema import PlannerOutput, RetrievalRewriteOutput
from src.core.request_contracts import RequestContract, UserTurnSnapshot, WireRequestContract
from src.infra.decisions import DecisionsClient


def make_decisions_client(probabilities: dict[str, float] | None = None) -> DecisionsClient:
    """Keep SDK parsing, budgets and usage real while replacing provider HTTP."""
    probabilities = dict(probabilities or {})

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/decisions"
        payload = json.loads(request.content)
        return httpx.Response(200, headers={"x-request-id": "req_test_decisions"}, json={
            "model": "gpt-6-luna",
            "answers": [
                {"type": "predicate", "name": question["name"],
                 "probability": probabilities.get(question["name"], 0.99)}
                for question in payload["questions"]
            ],
            "usage": {"input_tokens": 32, "output_tokens": 0, "total_tokens": 32},
        })

    sdk = OpenAI(api_key="test-key", base_url="https://provider.test/v1",
                 http_client=httpx.Client(transport=httpx.MockTransport(respond)), max_retries=0)
    return DecisionsClient(sdk)


def build_test_state(payload: dict):
    raw = dict(payload)
    user_input = str(raw.pop("user_input", "") or "")
    messages = raw.pop("messages", []) or []
    users = [message for message in messages if isinstance(message, HumanMessage)]
    current_turn_id = raw.pop("current_turn_id", None) or (users[-1].id if users and users[-1].id else f"user:{max(0, len(users)-1)}")
    user_turns = raw.pop("user_turns", None)
    if user_turns is None:
        user_turns = tuple(UserTurnSnapshot(turn_id=message.id or f"user:{index}", text=str(message.content)) for index, message in enumerate(users))
        if not user_turns:
            user_turns = (UserTurnSnapshot(turn_id=current_turn_id, text=user_input),)
    state = build_graph_state_input(
        user_input=user_input, current_turn_id=current_turn_id, user_turns=user_turns,
        messages=messages,
        retriever=raw.pop("retriever", None),
        session_metadata=raw.pop("session_metadata", None),
        memory_summary=raw.pop("memory_summary", None),
        previous_response=raw.pop("previous_response", None),
        request_contract=raw.pop("request_contract", None),
        pending_action=raw.pop("pending_action", None),
    )
    if "planner" in raw:
        state["planner"] = PlannerState.model_validate(raw.pop("planner"))
    elif any(key in raw for key in ("planner_output", "planner_status", "planner_diagnostics", "guided_followup")):
        state["planner"] = PlannerState(
            output=raw.pop("planner_output", None) or PlannerState().output,
            status=raw.pop("planner_status", "llm") or "llm",
            diagnostics=raw.pop("planner_diagnostics", None) or PlannerState().diagnostics,
            guided_followup=raw.pop("guided_followup", None),
        )
    if "retrieval" in raw:
        state["retrieval"] = RetrievalState.model_validate(raw.pop("retrieval"))
    elif "retrieved_hits" in raw:
        state["retrieval"] = RetrievalState(hit_log=raw.pop("retrieved_hits"))
    if "retry" in raw:
        state["retry"] = RetryState.model_validate(raw.pop("retry"))
    elif "retry_context" in raw or "needs_retry" in raw:
        retry = dict(raw.pop("retry_context", {}) or {})
        if "needs_retry" in raw:
            retry["needs_retry"] = raw.pop("needs_retry")
        state["retry"] = RetryState.model_validate(retry)
    if "response" in raw:
        state["response"] = ResponseState.model_validate(raw.pop("response"))
    elif any(key in raw for key in ("result", "evidence_packet", "synthesis_attempt")):
        state["response"] = ResponseState(
            result=raw.pop("result", None) or ResponseState().result,
            evidence_packet=raw.pop("evidence_packet", []),
            synthesis_attempt=int(raw.pop("synthesis_attempt", 0) or 0),
        )
    debug = raw.pop("debug", None)
    if debug is None:
        debug = {key: raw.pop(key) for key in list(raw) if key in DebugState.model_fields}
    if debug:
        state["debug"] = DebugState.model_validate(debug)
    if raw:
        raise ValueError(f"Unknown test state fields: {sorted(raw)}")
    return normalize_graph_update(state)


class _CapturePlannerLLM:
    def __init__(self, planner_output):
        # Retrieval-only fixtures still supply an explicit neutral interpretation.
        # Missing-contract tests use an unadapted model boundary of their own.
        if isinstance(planner_output, PlannerOutput) and planner_output.request_contract is None:
            planner_output.request_contract = WireRequestContract(slack_recipient={"state": "omitted"})
        elif isinstance(planner_output, PlannerOutput) and isinstance(planner_output.request_contract, RequestContract):
            planner_output.request_contract = planner_output.request_contract.to_wire()
        elif isinstance(planner_output, dict) and "rewrites" not in planner_output:
            planner_output = dict(planner_output)
            planner_output.setdefault("request_contract", WireRequestContract(slack_recipient={"state": "omitted"}).model_dump(mode="json"))
        self.planner_output = (planner_output.model_dump(mode="json")
                               if isinstance(planner_output, (PlannerOutput, RetrievalRewriteOutput)) else planner_output)
        self.call_count = 0

    def invoke(self, messages, *, timeout=None):
        self.call_count += 1
        return self.planner_output
