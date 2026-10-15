"""Incomplete retrieval preserves grounded natural answers without delivery."""
import json

import pytest

from src.core.answer_schema import export_answer_text
from src.core.contracts import RetryState
from src.core.contracts.debug import RetrievalDiagnostic
from src.core.contracts.retrieval import RetrievalResult
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.runtime.agent_runtime.response_assembler import ResponseAssembler
from src.runtime.make_graph import _pre_synthesis_route
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.retry import build_retry_update
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.core.helpers import build_test_state
from tests.core.test_actions_nodes import _contract
from tests.core.test_answer_validation import _evidence, _hit
from tests.synthesis_fixtures import synthesis_excerpt_limits


class NaturalPartialModel:
    def __init__(self):
        self.calls = 0
        self.messages = []

    def with_structured_output(self, *args, **kwargs):
        return self

    def invoke(self, messages, *, timeout=None):
        self.calls += 1
        self.messages = messages
        packet_text = next(str(message.content) for message in messages
                           if str(message.content).startswith("[Evidence Packet]"))
        packet = json.loads(packet_text[packet_text.index("[", len("[Evidence Packet]")):])
        return {"blocks": [{"type": "paragraph", "content": [
            {"text": "The available documentation explains axis selection.", "basis": "source", "refs": [packet[0]["id"]]},
            {"text": "The requested uploaded evidence remains unavailable, so that part cannot be compared.", "basis": "interaction", "refs": []},
        ]}]}


@pytest.mark.parametrize("failure", ["partial", "error", "unavailable"])
def test_incomplete_retrieval_returns_a_natural_cited_partial_without_delivery(failure):
    tasks = [RetrievalTask(route=route, query="axis", k=2) for route in ("docs", "upload")]
    contract = _contract(save="requested", slack="requested")
    state = build_test_state({
        "user_input": "Compare the docs and uploaded code, then save and send the answer",
        "request_contract": contract, "planner_output": PlannerOutput(use_retrieval=True, tasks=tasks),
        "retry": RetryState(max_retries=0, original_tasks=tasks),
    })
    calls = []

    def docs(**kwargs):
        calls.append("docs")
        return RetrievalResult(hits=[_hit(_evidence("axis selects the dimension", source_type="official"))],
                               diagnostics=RetrievalDiagnostic(status="success", answerability="covered"))

    def upload(**kwargs):
        calls.append("upload")
        hits = [_hit(_evidence("Only part of the uploaded source is available"))] if failure == "partial" else []
        return RetrievalResult(hits=hits, diagnostics=RetrievalDiagnostic(
            status="success" if failure == "partial" else failure,
            answerability="partial" if hits else "missing", message="source unavailable",
        ))

    state.update(make_retrieve_dispatch_node(docs, upload, False)(state))
    state.update(make_pre_synthesis_validation_node(False)(state))

    assert _pre_synthesis_route(state)[0] == "synthesize"
    assert state["retry"].terminal_failed_requirement_ids == [tasks[1].requirement_id]
    model = NaturalPartialModel()
    state.update(make_synthesize_node(model, excerpt_limits=synthesis_excerpt_limits())(state))
    state.update(make_post_synthesis_validation_node(False)(state))
    outcome = ResponseAssembler().assemble(response=state, debug_info={})
    delivered = []
    state.update(make_action_postprocess_node(
        lambda **kwargs: delivered.append("save"), lambda **kwargs: delivered.append("slack"), False,
    )(state))

    assert outcome["status"] == "partial"
    assert outcome["problem"]["stage"] == "retrieval"
    assert state["response"].kind == "failure"
    assert state["response"].result.citations
    assert "The available documentation explains axis selection." in export_answer_text(state["response"].result)
    assert "answer_incomplete" in {issue.code for issue in state["response"].result.issues}
    assert any("[Incomplete Retrieval]" in str(message.content) for message in model.messages)
    assert model.calls == 1
    assert sorted(calls) == ["docs", "upload"]
    assert delivered == []
    assert not state["runtime"].pending_action.body_prepared






@pytest.mark.parametrize("repair_succeeds", [False, True])
def test_graph_bounds_search_recovery_and_body_repair_independently(repair_succeeds):
    from src.core.contracts import GraphState
    from src.core.conversation_memory import ConversationMemoryPolicy
    from src.runtime.make_graph import build_graph
    from src.runtime.nodes.planner import make_planner_node
    from src.runtime.nodes.session import add_user_message
    from tests.core.helpers import _CapturePlannerLLM

    first = RetrievalTask(route="docs", query="narrow axis", k=2, requirement_id="first")
    second = RetrievalTask(route="docs", query="working axis", k=2, requirement_id="second")
    initial = _CapturePlannerLLM(PlannerOutput(use_retrieval=True, tasks=[first, second]))
    rewrite = _CapturePlannerLLM({"rewrites": [{"requirement_id": "first", "query": "broader axis", "k": 2}]})
    searches = []

    def search(*, query, **kwargs):
        searches.append(query)
        hits = [] if query == "narrow axis" else [_hit(_evidence(query + " selects the dimension", source_type="official"))]
        return RetrievalResult(hits=hits, diagnostics=RetrievalDiagnostic(
            status="success" if hits else "no_result", answerability="covered" if hits else "missing",
        ))

    class RepairModel(NaturalPartialModel):
        def invoke(self, messages, *, timeout=None):
            self.calls += 1
            packet_text = next(str(message.content) for message in messages
                               if str(message.content).startswith("[Evidence Packet]"))
            packet = json.loads(packet_text[packet_text.index("[", len("[Evidence Packet]")):])
            refs = [source["id"] for source in packet] if repair_succeeds and self.calls > 1 else [packet[0]["id"]]
            return {"blocks": [{"type": "paragraph", "content": [
                {"text": "The sources describe axis selection.", "basis": "source", "refs": refs},
            ]}]}

    model = RepairModel()
    graph = build_graph(
        GraphState, add_user_message, lambda state: {},
        make_planner_node(initial, False, llm_planner_retry=rewrite),
        make_retrieve_dispatch_node(search, None, False),
        make_synthesize_node(model, excerpt_limits=synthesis_excerpt_limits()),
        memory_policy=ConversationMemoryPolicy(),
        pre_synthesis_validation_node=make_pre_synthesis_validation_node(False),
        post_synthesis_validation_node=make_post_synthesis_validation_node(False),
        action_postprocess_node=make_action_postprocess_node(lambda **kwargs: None, lambda **kwargs: None, False),
    )

    state = graph.invoke(build_test_state({"user_input": "Compare both axis sources"}))
    outcome = ResponseAssembler().assemble(response=state, debug_info={})

    assert outcome["status"] == ("completed" if repair_succeeds else "partial")
    assert state["retry"].attempt == 1
    assert state["retry"].content_repair_attempt == 1
    assert model.calls == 2
    assert initial.call_count == 1
    assert rewrite.call_count == 1
    assert searches.count("narrow axis") == 1
    assert searches.count("broader axis") == 1
    assert searches.count("working axis") == 1
    assert len(state["response"].result.citations) == 2
    assert not state["retry"].needs_retry


@pytest.mark.parametrize("docs_need_rewrite", [False, True])
def test_graph_keeps_independent_docs_when_upload_retriever_is_missing(docs_need_rewrite):
    from src.core.contracts import GraphState
    from src.core.conversation_memory import ConversationMemoryPolicy
    from src.infra.tools.local_rag.tool import build_upload_search_tool
    from src.runtime.make_graph import build_graph
    from src.runtime.nodes.planner import make_planner_node
    from src.runtime.nodes.session import add_user_message
    from tests.core.helpers import _CapturePlannerLLM

    tasks = [RetrievalTask(route="docs", query="narrow axis" if docs_need_rewrite else "axis", k=2),
             RetrievalTask(route="upload", query="axis", k=2)]
    initial = _CapturePlannerLLM(PlannerOutput(use_retrieval=True, tasks=tasks))
    rewrite = _CapturePlannerLLM({"rewrites": [{"requirement_id": tasks[0].requirement_id, "query": "axis", "k": 2}]})
    model = NaturalPartialModel()
    searches = []

    def search(*, query, **kwargs):
        searches.append(query)
        if query == "narrow axis":
            return RetrievalResult(hits=[], diagnostics=RetrievalDiagnostic(status="no_result", answerability="missing"))
        return RetrievalResult(hits=[_hit(_evidence("axis selects the dimension", source_type="official"))],
                               diagnostics=RetrievalDiagnostic(status="success", answerability="covered"))

    graph = build_graph(
        GraphState, add_user_message, lambda state: {}, make_planner_node(initial, False, llm_planner_retry=rewrite),
        make_retrieve_dispatch_node(search, build_upload_search_tool(), False),
        make_synthesize_node(model, excerpt_limits=synthesis_excerpt_limits()),
        memory_policy=ConversationMemoryPolicy(),
        pre_synthesis_validation_node=make_pre_synthesis_validation_node(False),
        post_synthesis_validation_node=make_post_synthesis_validation_node(False),
        action_postprocess_node=make_action_postprocess_node(lambda **kwargs: None, lambda **kwargs: None, False),
    )

    state = graph.invoke(build_test_state({"user_input": "Compare official axis docs with my uploaded code"}))
    outcome = ResponseAssembler().assemble(response=state, debug_info={})

    assert outcome["status"] == "partial"
    assert [task.requirement_id for task in state["planner"].output.tasks] == [task.requirement_id for task in tasks]
    assert state["retry"].terminal_failed_requirement_ids == [tasks[1].requirement_id]
    assert searches == (["narrow axis", "axis"] if docs_need_rewrite else ["axis"])
    assert state["retry"].attempt == int(docs_need_rewrite)
    assert model.calls == 1
    assert len(state["response"].result.citations) == 1
    assert state["debug"].retrieval_diagnostics[-1].status == "unavailable"











def _exhausted_partial_recovery_state(guard=None):
    from src.core.answer_schema import AnswerDocument, finalize_answer
    from src.core.contracts import ResponseState

    missing = RetrievalTask(route="docs", query="official axis behavior", k=2)
    available = RetrievalTask(route="upload", query="uploaded axis behavior", k=2,
                              requirement={"aspects": ["copy"]} if guard == "all_terminal_uncited" else {})
    tasks = [missing, available]
    hit = _hit(_evidence("axis selects the dimension")).model_copy(update={"requirement_id": available.requirement_id})
    hits = [hit]
    units = [
        {"text": "The uploaded source explains axis selection.", "basis": "source", "refs": [hit.evidence.id]},
        {"text": "The official documentation is unavailable, so the sources cannot be compared.", "basis": "interaction", "refs": []},
        {"text": "An unsupported claim.", "basis": "source", "refs": ["missing"]},
    ]
    terminal_ids = [missing.requirement_id]
    answer = {}
    blocks = [{"type": "paragraph", "content": units}]
    if guard == "missing_success_requirement":
        other = RetrievalTask(route="upload", query="uploaded copy behavior", k=2)
        tasks.append(other)
        hits.append(_hit(_evidence("copy creates an independent result")).model_copy(
            update={"requirement_id": other.requirement_id},
        ))
        units[-1]["refs"].append(hits[-1].evidence.id)
    elif guard == "required_code_removed":
        answer = {"content": [{"kind": "code_example", "mode": "required", "evidence_ids": ["request"]}]}
        blocks.append({"type": "code", "language": "python", "content": {
            "text": "unavailable_example()", "basis": "source", "refs": ["missing"],
        }})
    elif guard == "forbidden_code_retained":
        answer = {"format": [{"kind": "code_block", "mode": "forbidden", "evidence_ids": ["request"]}]}
        blocks.append({"type": "code", "language": "python", "content": {
            "text": "axis = 0", "basis": "source", "refs": [hit.evidence.id],
        }})
    elif guard == "all_terminal_uncited":
        terminal_ids.append(available.requirement_id)
        units.pop(0)
    contract = _contract(save="requested", slack="requested", answer=answer)
    packet = [item.evidence for item in hits]
    return build_test_state({
        "user_input": "Compare the official docs and uploaded code, then save and send the answer",
        "request_contract": contract, "planner_output": PlannerOutput(use_retrieval=True, tasks=tasks),
        "retrieved_hits": hits,
        "retrieval_diagnostics": [
            RetrievalDiagnostic(route="docs", requirement_id=missing.requirement_id, status="error", answerability="missing"),
            *[RetrievalDiagnostic(route="upload", requirement_id=item.requirement_id, status="success",
                                  answerability="partial" if guard == "all_terminal_uncited" else "covered") for item in hits],
        ],
        "retry": RetryState(attempt=1, max_retries=1, content_repair_attempt=1,
                            terminal_failed_requirement_ids=terminal_ids, terminal_retrieval_reason="tool_error"),
        "response": ResponseState(
            result=finalize_answer(AnswerDocument.model_validate({"blocks": blocks}), packet, retrieval_required=True),
            evidence_packet=packet, evidence_requirement_map={item.evidence.id: [item.requirement_id] for item in hits},
            request_id=contract.request_id, contract_revision=contract.revision, synthesis_attempt=2,
        ),
    })




