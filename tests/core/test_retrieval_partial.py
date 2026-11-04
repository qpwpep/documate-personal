"""Incomplete retrieval preserves grounded natural answers without delivery."""
import hashlib
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.core.answer_schema import AnswerDocument, export_answer_text, finalize_answer, text_document
from src.core.contracts import PlannerState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.llm_errors import FailureCode, make_problem
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.core.uploads import UploadFileInfo
from src.infra.chunking import chunk_python_text
from src.infra.tools.local_rag import build_upload_search_tool
from src.runtime.agent_runtime.response_assembler import ResponseAssembler
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from src.runtime.nodes.validation.policy import build_response_updates
from tests.core.helpers import make_decisions_client, mock_chat_model
from tests.core.test_decisions_validation import DecisionsBoundary, read_decision_input
from tests.synthesis_fixtures import synthesis_excerpt_limits


@pytest.mark.parametrize("failure_code", ["evidence_insufficient", "provider_configuration", "model_output_invalid"])
def test_failed_result_does_not_publish_unvalidated_document(failure_code: FailureCode) -> None:
    problem = make_problem(failure_code, "retrieval")
    state = build_response_updates(
        finalize_answer(text_document("UNVALIDATED_PROVIDER_TEXT"), []),
        attempt=1, evidence_packet=[], kind="failure", problem=problem,
    )

    outcome = ResponseAssembler().assemble(response=state)

    assert outcome.status == "failed"
    assert outcome.response is None
    assert outcome.message == problem.message
    assert "UNVALIDATED_PROVIDER_TEXT" not in outcome.message


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
    decisions = make_decisions_client({f"requirement_{len(tasks) - 1}_target_analysis_supported": 0.0})

    state.update(make_retrieve_dispatch_node(None, build_upload_search_tool(), False)(state))
    state.update(make_pre_synthesis_validation_node(
        False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=decisions,
    )(state))

    if not has_known_value:
        outcome = ResponseAssembler().assemble(response=state)
        assert outcome.status == "failed"
        assert outcome.problem.code == "evidence_insufficient"
        assert outcome.problem.next_action == "supply_information"
        assert "worker_profile.py" in outcome.message
        assert all(symbol in outcome.message for symbol in requested_symbols)
        assert outcome.response is None
        assert not state["response"].result.citations
        assert state["retrieval"].prepared_evidence.normal.evidence == []
        return

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
    outcome = ResponseAssembler().assemble(response=state)

    body_request, = [call for call in post_decisions.requests
                    if any(question["name"] == "request_fulfilled" for question in call["questions"])]
    body_payload = read_decision_input(body_request["input"])
    gap, = body_payload["retrieval_gaps"]
    assert gap["task"]["requirement"]["symbols"] == ["FAILOVER_LIMIT"]
    assert gap["reason"] == "no_evidence"
    assert gap["files"] == [{"file_id": file_id, "name": path.name}]
    diagnostic, = gap["diagnostics"]
    assert diagnostic["status"] == "no_result"
    assert diagnostic["answerability"] == "missing"
    assert diagnostic["missing_requirements"] == [f"file:{file_id}:FAILOVER_LIMIT"]
    assert outcome.status == "partial"
    assert outcome.problem.code == "evidence_insufficient"
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
        "body": AnswerDocument.model_validate({"blocks": [
            {"type": "paragraph", "content": [{"text": f"{symbol}의 값은 {value}입니다.", "basis": "source",
                                               "refs": [alias]}]}
            for (symbol, value), alias in zip(expected_values, ["e1", "e2"])
        ]}).model_dump(mode="json"),
        "notices": gaps,
    }
    assert "displayed_document" not in body_payload
    assert gaps[0] in export_answer_text(result, include_sources=True)
    assert not state["retry"].needs_retry
    assert state["retry"].content_repair_attempt == 0
    assert state["response"].synthesis_attempt == 1
