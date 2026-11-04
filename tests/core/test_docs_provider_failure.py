"""Provider authentication failures stop retrieval without changing the query."""
import json
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest

from src.core.answer_schema import AnswerDocument, export_answer_text
from src.core.contracts import PlannerState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.llm_errors import LLMCallError
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.core.uploads import UploadFileInfo
from src.infra.chunking import chunk_python_text
from src.infra.settings import AppSettings
from src.infra.tools.docs_search import client
from src.infra.tools.docs_search.tool import build_docs_search_tool
from src.infra.tools.local_rag import build_upload_search_tool
from src.runtime.agent_runtime.response_assembler import ResponseAssembler
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.core.helpers import build_test_state, make_decisions_client, mock_chat_model
from tests.synthesis_fixtures import synthesis_excerpt_limits


@pytest.fixture
def tavily_http(monkeypatch):
    replies = []
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            status, body = replies.pop(0)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(client, "TAVILY_SEARCH_API_URL", f"http://127.0.0.1:{server.server_port}/search")
    try:
        yield replies, received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("status, code, next_action", [
    (401, "provider_configuration", "fix_configuration"),
    (403, "provider_configuration", "fix_configuration"),
    (429, "provider_rate_limited", "retry_later"),
])
def test_provider_rejection_stops_before_replanning(tavily_http, status, code, next_action):
    replies, received = tavily_http
    # The status must survive even when the provider body is not valid JSON.
    replies.append((status, b"private-provider-detail and secret-api-key"))
    task = RetrievalTask(route="docs", query="pandas DataFrame.head", k=3, requirement_id="head",
                         requirement={"library": "pandas", "symbols": ["pandas.DataFrame.head"]})
    state = build_test_state({
        "request_contract": RequestContract(), "user_input": "head 사용법을 설명해줘",
        "planner_output": PlannerOutput(use_retrieval=True, tasks=[task]),
        "retry": RetryState(),
    })
    settings = AppSettings(_env_file=None, tavily_api_key="test", openai_api_key="test")
    state.update(make_retrieve_dispatch_node(build_docs_search_tool(settings), None, False)(state))
    validate = make_pre_synthesis_validation_node(
        False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=make_decisions_client(),
    )

    with pytest.raises(LLMCallError) as raised:
        validate(state)

    assert raised.value.problem.code == code
    assert raised.value.problem.stage == "retrieval"
    assert raised.value.problem.next_action == next_action
    assert not state["retry"].needs_retry
    assert len(received) == 1
    diagnostic = state["debug"].retrieval_diagnostics[-1]
    assert diagnostic.error_code == code
    assert "secret-api-key" not in diagnostic.message
    assert "private-provider-detail" not in diagnostic.message


def test_temporary_search_failure_remains_retryable(tavily_http):
    replies, received = tavily_http
    replies.append((503, b'{"detail":{"error":"private-provider-detail"}}'))
    task = RetrievalTask(route="docs", query="pandas DataFrame.head", k=3, requirement_id="head",
                         requirement={"library": "pandas"})
    state = build_test_state({
        "request_contract": RequestContract(), "user_input": "head 사용법을 설명해줘",
        "planner_output": PlannerOutput(use_retrieval=True, tasks=[task]),
    })
    settings = AppSettings(_env_file=None, tavily_api_key="test", openai_api_key="test")
    state.update(make_retrieve_dispatch_node(build_docs_search_tool(settings), None, False)(state))
    state.update(make_pre_synthesis_validation_node(
        False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=make_decisions_client(),
    )(state))

    assert state["retry"].needs_retry
    assert state["retry"].retry_reason == "tool_error"
    assert state["retry"].failed_requirement_ids == ["head"]
    assert len(received) == 1
    assert "private-provider-detail" not in state["debug"].retrieval_diagnostics[-1].message


@pytest.mark.parametrize("status, code, next_action", [
    (401, "provider_configuration", "fix_configuration"),
    (429, "provider_rate_limited", "retry_later"),
])
def test_rejected_docs_preserve_a_cited_upload_partial_without_replanning(
    tmp_path, tavily_http, status, code, next_action,
):
    replies, received = tavily_http
    replies.append((status, b"private-provider-detail"))
    source = b"WORKER_COUNT = 6\n"
    path = tmp_path / "worker_profile.py"
    path.write_bytes(source)
    file_id = "a" * 32
    upload = UploadFileInfo(
        file_id=file_id, name=path.name, size_bytes=len(source),
        content_hash="sha256:" + hashlib.sha256(source).hexdigest(),
        source_uri=f"upload:///provider-failure/{file_id}",
    )
    document = chunk_python_text(
        path=str(path), text=source.decode(), source_content=source,
        source_uri=upload.source_uri, title=upload.name, chunk_size=800, chunk_overlap=120,
    )
    for element in document.parsed.elements:
        element.metadata["file_id"] = file_id
    tasks = [
        RetrievalTask(route="docs", query="pandas DataFrame.head", k=3, requirement_id="head",
                      requirement={"library": "pandas", "symbols": ["pandas.DataFrame.head"]}),
        RetrievalTask(route="upload", query="worker_profile.py의 WORKER_COUNT 값", k=3, requirement_id="workers",
                      requirement={"symbols": ["WORKER_COUNT"], "match": "symbol", "file_ids": [file_id]}),
    ]
    question = "pandas head 사용법과 첨부 코드의 WORKER_COUNT 값을 알려줘."
    state = build_graph_state_input(
        user_input=question, current_turn_id="provider-failure", upload_files=(upload,),
        retriever=SimpleNamespace(source_documents=(document.parsed,)),
        request_contract=RequestContract(request_id="provider-failure", body={"kind": "compose", "instruction": question}),
        planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=tasks)),
        retry=RetryState(original_tasks=tasks),
    )
    settings = AppSettings(_env_file=None, tavily_api_key="test", openai_api_key="test")
    state.update(make_retrieve_dispatch_node(
        build_docs_search_tool(settings), build_upload_search_tool(), False,
    )(state))
    state.update(make_pre_synthesis_validation_node(
        False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=make_decisions_client(),
    )(state))

    assert not state["retry"].needs_retry
    assert state["retry"].terminal_failed_requirement_ids == ["head"]
    packet = state["retrieval"].prepared_evidence.normal.evidence
    assert [item.excerpt for item in packet] == ["WORKER_COUNT = 6"]
    answer = AnswerDocument.model_validate({"blocks": [{"type": "paragraph", "content": [
        {"text": "WORKER_COUNT의 값은 6입니다.", "basis": "source", "refs": [packet[0].id]},
        {"text": "공식 문서 검색 실패로 head 사용법은 확인하지 못했습니다.", "basis": "interaction", "refs": []},
    ]}]})
    state.update(make_synthesize_node(mock_chat_model(lambda _messages: answer))(state))
    state.update(make_post_synthesis_validation_node(False, decisions_client=make_decisions_client())(state))
    outcome = ResponseAssembler().assemble(response=state)

    assert outcome.status == "partial"
    assert outcome.problem.code == code
    assert outcome.problem.next_action == next_action
    assert "WORKER_COUNT의 값은 6입니다." in export_answer_text(state["response"].result)
    assert state["response"].result.citations
    assert "answer_incomplete" in {issue.code for issue in state["response"].result.issues}
    assert len(received) == 1
