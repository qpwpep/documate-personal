"""Incomplete retrieval preserves grounded natural answers without delivery."""
import hashlib
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.core.answer_schema import AnswerDocument, export_answer_text, finalize_answer, text_document
from src.core.contracts import PlannerState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.contracts.debug import RetrievalDiagnostic
from src.core.contracts.retrieval import RetrievalResult
from src.core.llm_errors import FailureCode, make_problem
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.core.uploads import UploadFileInfo
from src.infra.chunking import chunk_python_text
from src.infra.tools.local_rag import build_upload_search_tool
from src.runtime.agent_runtime.response_assembler import ResponseAssembler
from src.runtime.make_graph import _pre_synthesis_route
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from src.runtime.nodes.validation.policy import build_response_updates
from tests.core.helpers import build_test_state, make_decisions_client, mock_chat_model
from tests.core.test_pending_action_delivery import _contract
from tests.core.test_answer_validation import _evidence, _hit
from tests.core.test_decisions_validation import DecisionsBoundary, read_decision_input
from tests.synthesis_fixtures import synthesis_excerpt_limits


class NaturalPartialModel:
    def __init__(self):
        self.calls = 0
        self.messages = []


    def invoke(self, messages, *, timeout=None):
        self.calls += 1
        self.messages = messages
        packet_text = next(str(message.content) for message in messages
                           if str(message.content).startswith("[Evidence Packet]"))
        packet = json.loads(packet_text[packet_text.index("[", len("[Evidence Packet]")):])
        return AnswerDocument.model_validate({"blocks": [{"type": "paragraph", "content": [
            {"text": "The available documentation explains axis selection.", "basis": "source", "refs": [packet[0]["id"]]},
            {"text": "The requested uploaded evidence remains unavailable, so that part cannot be compared.", "basis": "interaction", "refs": []},
        ]}]})


@pytest.mark.parametrize("failure_code", ["evidence_insufficient", "provider_configuration", "model_output_invalid"])
def test_failed_result_does_not_publish_unvalidated_document(failure_code: FailureCode) -> None:
    problem = make_problem(failure_code, "retrieval")
    state = build_response_updates(
        finalize_answer(text_document("UNVALIDATED_PROVIDER_TEXT"), []),
        attempt=1, evidence_packet=[], kind="failure", problem=problem,
    )

    outcome = ResponseAssembler().assemble(response=state, debug_info={})

    assert outcome["status"] == "failed"
    assert outcome["response"] is None
    assert outcome["message"] == problem.message
    assert "UNVALIDATED_PROVIDER_TEXT" not in outcome["message"]


def test_incomplete_retrieval_returns_a_natural_cited_partial_without_delivery():
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
        return RetrievalResult(hits=[], diagnostics=RetrievalDiagnostic(
            status="error", answerability="missing", message="source unavailable",
        ))

    state.update(make_retrieve_dispatch_node(docs, upload, False)(state))
    state.update(make_pre_synthesis_validation_node(False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=make_decisions_client())(state))

    assert _pre_synthesis_route(state)[0] == "synthesize"
    assert state["retry"].terminal_failed_requirement_ids == [tasks[1].requirement_id]
    model = NaturalPartialModel()
    state.update(make_synthesize_node(mock_chat_model(model.invoke))(state))
    state.update(make_post_synthesis_validation_node(False, decisions_client=make_decisions_client())(state))
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


@pytest.mark.parametrize("with_history", [False, True])
@pytest.mark.parametrize("has_known_value, include_prefetch", [(False, False), (True, False), (True, True)],
                         ids=["no-requested-values", "one-known-one-missing", "two-known-one-missing"])
def test_mixed_upload_variables_keep_confirmed_value_and_identify_only_missing_items(
    tmp_path, with_history, has_known_value, include_prefetch,
):
    source = ("WORKER_COUNT = 6\n" if has_known_value else "") + "PREFETCH_LIMIT = 48\n"
    path = tmp_path / "worker_profile.py"
    path.write_bytes(source.encode("utf-8"))
    content = path.read_bytes()
    file_id = "a" * 32
    upload = UploadFileInfo(
        file_id=file_id, name=path.name, size_bytes=len(content),
        content_hash="sha256:" + hashlib.sha256(content).hexdigest(),
        source_uri=f"upload:///mixed-variables/{file_id}",
    )
    document = chunk_python_text(
        path=str(path), text=content.decode("utf-8"), source_content=content,
        source_uri=upload.source_uri, title=upload.name, chunk_size=800, chunk_overlap=120,
    )
    for element in document.parsed.elements:
        element.metadata["file_id"] = file_id
    requested_symbols = ["WORKER_COUNT", "PREFETCH_LIMIT", "FAILOVER_LIMIT"] if include_prefetch else [
        "WORKER_COUNT", "FAILOVER_LIMIT",
    ]
    tasks = [RetrievalTask(
        route="upload", query=f"worker_profile.py에서 {symbol}의 할당값을 알려줘.", k=3,
        requirement_id=symbol, requirement={"symbols": [symbol], "match": "symbol", "file_ids": [file_id]},
    ) for symbol in requested_symbols]
    question = f"첨부 코드의 {', '.join(requested_symbols)} 값을 알려줘."
    messages = ([HumanMessage(content="pandas merge 사용법을 설명해줘.", id="prior-user"),
                 AIMessage(content="pandas.merge는 두 DataFrame을 조인합니다.", id="prior-answer")]
                if with_history else [])
    messages.append(HumanMessage(content=question, id="current-user"))
    state = build_graph_state_input(
        user_input=question, current_turn_id="current-user", messages=messages,
        upload_files=(upload,), retriever=SimpleNamespace(source_documents=(document.parsed,)),
        request_contract=RequestContract(request_id="mixed-variables", body={"kind": "compose", "instruction": question}),
        planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=tasks)),
        retry=RetryState(max_retries=0, original_tasks=tasks),
    )
    # These are provider judgments, not a heuristic implementation of the rubric.
    # The missing requirement still fails when its exact source range is absent.
    decisions = make_decisions_client({"request_ready": 0.1, f"requirement_{len(tasks) - 1}": 0.0})

    state.update(make_retrieve_dispatch_node(None, build_upload_search_tool(), False)(state))
    state.update(make_pre_synthesis_validation_node(
        False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=decisions,
    )(state))

    if not has_known_value:
        outcome = ResponseAssembler().assemble(response=state, debug_info={})
        assert outcome["status"] == "failed"
        assert outcome["problem"]["code"] == "evidence_insufficient"
        assert outcome["problem"]["next_action"] == "supply_information"
        assert "worker_profile.py" in outcome["message"]
        assert all(symbol in outcome["message"] for symbol in requested_symbols)
        assert outcome["response"] is None
        assert not state["response"].result.citations
        assert state["retrieval"].prepared_evidence.normal.evidence == []
        return

    assert _pre_synthesis_route(state)[0] == "synthesize"
    assert state["retry"].terminal_failed_requirement_ids == ["FAILOVER_LIMIT"]
    packet = state["retrieval"].prepared_evidence.normal.evidence
    expected_values = [("WORKER_COUNT", 6)]
    if include_prefetch:
        expected_values.append(("PREFETCH_LIMIT", 48))
    expected_excerpts = [f"{symbol} = {value}" for symbol, value in expected_values]
    assert [item.excerpt for item in packet] == expected_excerpts
    assert [item.element.metadata["file_id"] for item in packet] == [file_id] * len(expected_values)
    answer = AnswerDocument.model_validate({"blocks": [
        {"type": "paragraph", "content": [{"text": f"{symbol}의 값은 {value}입니다.", "basis": "source",
                                           "refs": [evidence.id]}]}
        for (symbol, value), evidence in zip(expected_values, packet)
    ]})
    state.update(make_synthesize_node(mock_chat_model(lambda _messages: answer))(state))
    post_decisions = DecisionsBoundary()
    state.update(make_post_synthesis_validation_node(False, decisions_client=post_decisions)(state))
    outcome = ResponseAssembler().assemble(response=state, debug_info={})

    body_payload = read_decision_input(post_decisions.requests[0]["input"])
    gap, = body_payload["retrieval_gaps"]
    assert gap["task"]["requirement"]["symbols"] == ["FAILOVER_LIMIT"]
    assert gap["reason"] == "no_evidence"
    assert gap["files"] == [{"file_id": file_id, "name": path.name}]
    diagnostic, = gap["diagnostics"]
    assert diagnostic["status"] == "no_result"
    assert diagnostic["answerability"] == "missing"
    assert diagnostic["missing_requirements"] == [f"file:{file_id}:FAILOVER_LIMIT"]
    assert outcome["status"] == "partial"
    assert outcome["problem"]["code"] == "evidence_insufficient"
    result = state["response"].result
    assert all(f"{symbol}의 값은 {value}입니다." in export_answer_text(result)
               for symbol, value in expected_values)
    assert "FAILOVER_LIMIT" not in export_answer_text(result)
    assert [citation.evidence.excerpt for citation in result.citations] == expected_excerpts
    gaps = [issue.message for issue in result.issues if issue.code == "retrieved_evidence_incomplete"]
    assert len(gaps) == 1
    assert "FAILOVER_LIMIT" in gaps[0] and "worker_profile.py" in gaps[0]
    assert all(symbol not in gaps[0] for symbol, _ in expected_values)
    assert body_payload["retrieval_notice"] == gaps[0]
    assert body_payload["rendered_response"] == {
        "body": result.content.model_dump(mode="json"), "notices": gaps,
    }
    assert "displayed_document" not in body_payload
    assert gaps[0] in export_answer_text(result, include_sources=True)
    assert not state["retry"].needs_retry
    assert state["retry"].content_repair_attempt == 0
    assert state["response"].synthesis_attempt == 1


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

    packets = []

    class RepairModel(NaturalPartialModel):
        def invoke(self, messages, *, timeout=None):
            self.calls += 1
            packet_text = next(str(message.content) for message in messages
                               if str(message.content).startswith("[Evidence Packet]"))
            packet = json.loads(packet_text[packet_text.index("[", len("[Evidence Packet]")):])
            packets.append(packet)
            refs = [source["id"] for source in packet] if repair_succeeds and self.calls > 1 else [packet[0]["id"]]
            return AnswerDocument.model_validate({"blocks": [{"type": "paragraph", "content": [
                {"text": "The sources describe axis selection.", "basis": "source", "refs": refs},
            ]}]})

    model = RepairModel()
    graph = build_graph(
        GraphState, add_user_message, lambda state: {},
        make_planner_node(initial, False, llm_planner_retry=rewrite),
        make_retrieve_dispatch_node(search, None, False),
        make_synthesize_node(mock_chat_model(model.invoke)),
        memory_policy=ConversationMemoryPolicy(),
        pre_synthesis_validation_node=make_pre_synthesis_validation_node(False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=make_decisions_client()),
        post_synthesis_validation_node=make_post_synthesis_validation_node(False, decisions_client=make_decisions_client()),
        action_postprocess_node=make_action_postprocess_node(lambda **kwargs: None, lambda **kwargs: None, False),
    )

    state = graph.invoke(build_test_state({"user_input": "Compare both axis sources"}))
    outcome = ResponseAssembler().assemble(response=state, debug_info={})

    assert outcome["status"] == ("completed" if repair_succeeds else "partial")
    assert state["retry"].attempt == 1
    assert state["retry"].content_repair_attempt == 1
    assert model.calls == 2
    assert packets[0] == packets[1]
    assert initial.call_count == 1
    assert rewrite.call_count == 1
    assert searches.count("narrow axis") == 1
    assert searches.count("broader axis") == 1
    assert searches.count("working axis") == 1
    assert len(state["response"].result.citations) == 2
    assert not state["retry"].needs_retry


def test_graph_keeps_independent_docs_when_upload_retriever_is_missing():
    from src.core.contracts import GraphState
    from src.core.conversation_memory import ConversationMemoryPolicy
    from src.infra.tools.local_rag.tool import build_upload_search_tool
    from src.runtime.make_graph import build_graph
    from src.runtime.nodes.planner import make_planner_node
    from src.runtime.nodes.session import add_user_message
    from tests.core.helpers import _CapturePlannerLLM

    tasks = [RetrievalTask(route="docs", query="narrow axis", k=2),
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
        make_synthesize_node(mock_chat_model(model.invoke)),
        memory_policy=ConversationMemoryPolicy(),
        pre_synthesis_validation_node=make_pre_synthesis_validation_node(False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=make_decisions_client()),
        post_synthesis_validation_node=make_post_synthesis_validation_node(False, decisions_client=make_decisions_client()),
        action_postprocess_node=make_action_postprocess_node(lambda **kwargs: None, lambda **kwargs: None, False),
    )

    state = graph.invoke(build_test_state({"user_input": "Compare official axis docs with my uploaded code"}))
    outcome = ResponseAssembler().assemble(response=state, debug_info={})

    assert outcome["status"] == "partial"
    assert [task.requirement_id for task in state["planner"].output.tasks] == [task.requirement_id for task in tasks]
    assert state["retry"].terminal_failed_requirement_ids == [tasks[1].requirement_id]
    assert searches == ["narrow axis", "axis"]
    assert state["retry"].attempt == 1
    assert model.calls == 1
    assert len(state["response"].result.citations) == 1
    assert state["debug"].retrieval_diagnostics[-1].status == "unavailable"
