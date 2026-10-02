import unittest
import pytest

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.core.conversation_memory import ConversationMemoryPolicy
from src.core.contracts import GraphState, PlannerState, ResponseState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.runtime.make_graph import build_graph
from src.runtime.graph_builder import _instrument_stage_node
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.session import add_user_message
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from src.core.planner_schema import PlannerOutput, RetrievalPlanOutput, RetrievalTask
from src.core.request_contracts import RequestContract

from .helpers import (
    _CapturePlannerLLM,
    _tool_payload,
)
from tests.synthesis_fixtures import synthesis_excerpt_limits


def _official_hit(*, uri, title, excerpt, score):
    snapshot = build_snapshot(
        source_uri=uri, title=title, media_type="text/html", source_type="official",
        content=excerpt, parser="fixture", parser_version="1",
    )
    evidence = build_evidence(snapshot=snapshot, element=DocumentElement(element_id="body", kind="paragraph", text=excerpt))
    return SearchHit(evidence=evidence, score=RetrievalScore(metric="relevance", raw=score, direction="higher"), rank=1).model_dump(mode="json")


def _response(text, *, state=None, attempt=1):
    evidence = [SearchHit.model_validate(hit).evidence for hit in get_retrieval_state(state).hit_log] if state else []
    document = text_document(text, basis="source" if evidence else "interaction", refs=[item.id for item in evidence])
    contract = state["runtime"].request_contract if state else None
    return ResponseState(result=finalize_answer(document, evidence), evidence_packet=evidence, synthesis_attempt=attempt,
                         request_id=contract.request_id if contract else None, contract_revision=contract.revision if contract else 0)


def _neutral_planner_update(state, plan=None):
    contract = state["runtime"].request_contract or RequestContract()
    plan = plan or PlannerOutput(use_retrieval=False, tasks=[])
    return {
        "runtime": state["runtime"].model_copy(update={"request_contract": contract}),
        "planner": PlannerState(output=plan.model_copy(update={"request_contract": contract.to_wire()})),
    }


def _run_graph(graph, initial_state):
    visited = []
    state = initial_state
    for mode, update in graph.stream(initial_state, stream_mode=["updates", "values"]):
        if mode == "updates":
            visited.extend(update)
        else:
            state = update
    branch_sources = {
        "add_user_message", "planner", "pre_synthesis_validation", "post_synthesis_validation",
    }
    expected = [
        (source, visited[index + 1])
        for index, source in enumerate(visited) if source in branch_sources
    ]
    decisions = state["route_decisions"]
    assert [(decision.source, decision.target) for decision in decisions] == expected
    assert [decision.sequence for decision in decisions] == list(range(1, len(expected) + 1))
    return state, visited


class GraphRoutingTest(unittest.TestCase):
    pass






if __name__ == "__main__":
    unittest.main()






def _assert_repair_flow(routes, defect, max_retries, persistent, *, instrumented=True):
    import json

    from src.core.answer_schema import AnswerDocument
    from src.runtime.nodes.synthesis import make_synthesize_node
    from tests.core.test_answer_validation import _evidence, _hit

    hits = {
        route: _hit(_evidence(source_type="official" if route == "docs" else "upload")).model_dump(mode="json")
        for route in routes
    }
    searches = {"docs": 0, "upload": 0}
    packets = []

    def search(route, query):
        searches[route] += 1
        return _tool_payload(
            [hits[route]], tool="tavily_search" if route == "docs" else "upload_search",
            route=route, status="success", message="", query=query,
        )

    class RepairingLLM:
        def with_structured_output(self, *args, **kwargs):
            return self

        def invoke(self, messages):
            raw = next(str(message.content) for message in messages if str(message.content).startswith("[Evidence Packet]"))
            packet = json.loads(raw.split("\n", 2)[2])
            packets.append(packet)
            refs = [item["id"] for item in packet]
            broken = len(packets) == 1 or persistent
            if broken and defect == "references":
                refs = ["unknown-ref"]
            if broken and defect == "coverage":
                refs = refs[:1]
            if defect == "code" and not broken:
                document = {"blocks": [{
                    "type": "code", "language": "python",
                    "content": {"text": "retries = 5", "basis": "example", "refs": refs},
                }]}
            else:
                document = {"blocks": [{
                    "type": "paragraph",
                    "content": [{"text": "수정 전 본문" if broken else "확인한 설명", "basis": "source", "refs": refs}],
                }]}
            return AnswerDocument.model_validate(document).model_dump(mode="json")

    request = "업로드 코드 예시를 설명해줘" if defect == "code" else (
        "공식 문서와 업로드 자료를 비교해줘" if len(routes) == 2 else "공식 문서를 설명해줘"
    )
    contract = RequestContract.model_validate({
        "answer": {"content": [{"kind": "code_example", "mode": "required", "evidence_ids": ["code"]}]},
        "evidence": [{"id": "code", "turn_id": "request", "quote": request,
                      "scope": "answer.content.code_example", "interpretation": "instruction"}],
    }) if defect == "code" else RequestContract()
    plan = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route=route, query=route, k=3) for route in routes], request_contract=contract.to_wire())
    planner_llm = _CapturePlannerLLM(plan)
    contracts = []
    synthesize = make_synthesize_node(RepairingLLM(), verbose=False, excerpt_limits=synthesis_excerpt_limits())

    def synthesize_with_contract_observation(state):
        contracts.append(state["runtime"].request_contract)
        return synthesize(state)

    def observe(stage, node):
        return _instrument_stage_node(stage, node) if instrumented else node

    graph = build_graph(
        state_type=GraphState, add_user_node=add_user_message,
        summarize_node=lambda state: {},
        planner_node=observe("planner", make_planner_node(planner_llm, verbose=False)),
        retrieve_dispatch_node=make_retrieve_dispatch_node(
            lambda query, **kwargs: search("docs", query),
            lambda query, k, retriever=None, **kwargs: search("upload", query),
            verbose=False,
        ),
        synthesize_node=synthesize_with_contract_observation,
        pre_synthesis_validation_node=observe(
            "pre_synthesis_validation", make_pre_synthesis_validation_node(False),
        ),
        post_synthesis_validation_node=observe(
            "post_synthesis_validation", make_post_synthesis_validation_node(False),
        ),
        action_postprocess_node=lambda state: {},
        memory_policy=ConversationMemoryPolicy(),
    )

    initial_state = build_graph_state_input(
        user_input=request, current_turn_id="request", messages=[], retriever=object() if "upload" in routes else None,
        retry={"max_retries": max_retries},
    )
    state, visited = _run_graph(graph, initial_state)

    expected_attempts = 1 if max_retries == 0 else 2
    assert len(packets) == expected_attempts, (routes, defect, max_retries)
    assert all(packet == packets[0] for packet in packets)
    assert len(contracts) == expected_attempts
    assert all(contract == contracts[0] for contract in contracts)
    assert planner_llm.call_count == 1
    assert searches == {route: int(route in routes) for route in searches}
    result = state["response"].result
    assert all(check.reference_status != "missing" for check in result.checks)
    assert [message.content for message in state["messages"] if isinstance(message, AIMessage)] == [export_answer_text(result)]
    if persistent:
        assert any(issue.code == "source_excerpt_fallback" for issue in result.issues)
        assert "수정 전 본문" not in export_answer_text(result)
    else:
        assert not any(issue.code == "source_excerpt_fallback" for issue in result.issues)
    return state, visited
