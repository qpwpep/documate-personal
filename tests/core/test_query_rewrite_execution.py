"""Planner rewrites preserve search text through both request reuse layers."""
import json

import httpx
import pytest
import requests
from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.core.planner_schema import PlannerOutput, RetrievalRewriteOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.infra.decisions import DecisionsClient
from src.infra.llm_boundary import bind_structured_output
from src.infra.settings import AppSettings
from src.infra.tools.docs_search.tool import build_docs_search_tool
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.validation import make_pre_synthesis_validation_node
from tests.core.helpers import _CapturePlannerLLM, build_test_state, make_decisions_client


QUERY_REWRITES = [
    pytest.param("numpy NAN", "numpy nan", id="case"),
    pytest.param('numpy.reshape "x  y"', 'numpy.reshape "x y"', id="quoted-space"),
]


@pytest.fixture
def docs_http(monkeypatch):
    queries = []
    successful_queries = set()

    def response(url, data):
        value = requests.Response()
        value.status_code = 200
        value.url = url
        value._content = json.dumps(data).encode()
        value._content_consumed = True
        return value

    def post(url, **kwargs):
        query = kwargs["json"]["query"]
        queries.append(query)
        results = [{
            "url": "https://numpy.org/doc/2.4/reference/generated/numpy.reshape.html",
            "title": "numpy.reshape", "score": 0.9,
            "raw_content": "# numpy.reshape\nnumpy.reshape(a, shape)\nGives a new shape to an array.",
        }] if query in successful_queries else []
        return response(url, {"results": results})

    monkeypatch.setattr(requests, "post", post)
    yield queries, successful_queries


def _docs_tool():
    return build_docs_search_tool(AppSettings(_env_file=None, tavily_api_key="test", openai_api_key="test"))


@pytest.mark.parametrize("initial_query, rewritten_query", QUERY_REWRITES)
def test_planner_text_rewrite_reaches_http_and_recovers_missing_evidence(docs_http, initial_query, rewritten_query):
    queries, successful_queries = docs_http
    successful_queries.add(rewritten_query)
    task = RetrievalTask(route="docs", query=initial_query, k=2, requirement_id="reshape",
                         requirement={"library": "numpy", "symbols": ["numpy.reshape"], "match": "definition"})
    state = build_test_state({
        "request_contract": RequestContract(), "user_input": "numpy.reshape definition",
        "planner_output": PlannerOutput(use_retrieval=True, tasks=[task]),
    })
    dispatch = make_retrieve_dispatch_node(_docs_tool(), None, False)
    validate = make_pre_synthesis_validation_node(False, decisions_client=make_decisions_client())
    state.update(dispatch(state))
    assert queries[0] == initial_query
    state.update(validate(state))
    assert state["retry"].failed_requirement_ids == [task.requirement_id]
    state["retry"] = state["retry"].model_copy(update={"original_tasks": [task]})
    rewrite = _CapturePlannerLLM({"rewrites": [{
        "requirement_id": task.requirement_id, "query": rewritten_query, "k": task.k,
    }]})
    state.update(make_planner_node(None, False, llm_planner_retry=rewrite)(state))
    revised = state["planner"].output.tasks[0]
    assert revised.query == rewritten_query
    assert (revised.route, revised.requirement, revised.requirement_id) == (
        task.route, task.requirement, task.requirement_id,
    )
    previous_request_count = len(queries)
    state.update(dispatch(state))
    assert queries[previous_request_count:] == [rewritten_query]
    diagnostic = state["debug"].retrieval_diagnostics[-1]
    assert diagnostic.reused is False
    assert diagnostic.answerability == "covered"
    assert [request.query for request in diagnostic.provider_requests] == [rewritten_query]
    state.update(validate(state))
    assert not state["retry"].needs_retry
    assert not state["retry"].failed_requirement_ids


@pytest.mark.parametrize("same_uri", [True, False], ids=["new-snapshot-same-uri", "new-uri"])
@pytest.mark.parametrize("source_accepted", [True, False], ids=["incomplete-source", "rejected-source"])
def test_semantic_rejection_refreshes_covered_evidence_and_selects_new_source(monkeypatch, same_uri, source_accepted):
    initial_query = "numpy.reshape behavior"
    rewritten_query = "numpy.reshape compatible shape -1 dimension inference"
    initial_url = "https://numpy.org/doc/2.4/reference/generated/numpy.reshape.html"
    fresh_url = initial_url if same_uri else "https://numpy.org/doc/stable/reference/generated/numpy.reshape.html"
    initial_text = "# numpy.reshape\nnumpy.reshape(a, shape)\nGives a new shape to an array."
    fresh_text = initial_text + "\nOne shape dimension can be -1; its size is inferred from the array length and remaining dimensions."
    provider_queries = []
    decision_sources = []

    def tavily_post(url, **kwargs):
        payload = kwargs["json"]
        provider_queries.append((payload["query"], payload["max_results"]))
        fresh = payload["query"] == rewritten_query
        response = requests.Response()
        response.status_code = 200
        response.url = url
        response._content = json.dumps({"results": [{
            "url": fresh_url if fresh else initial_url,
            "title": "numpy.reshape", "score": 0.7 if fresh else 0.99,
            "raw_content": fresh_text if fresh else initial_text,
        }]}).encode()
        return response

    def openai_response(request):
        payload = json.loads(request.content)
        if request.url.path == "/v1/chat/completions":
            rewrite = {"rewrites": [{"requirement_id": "reshape", "query": rewritten_query, "k": 6}]}
            return httpx.Response(200, json={
                "id": "chatcmpl-rewrite", "object": "chat.completion", "created": 1, "model": "gpt-5.6-luna",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": json.dumps(rewrite),
                }}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            })
        assert request.url.path == "/v1/decisions"
        context = json.loads(payload["input"])
        if "evidence" in context:
            probabilities = {
                item["name"]: 0.99 if source_accepted or "One shape dimension can be -1" in item["source"]["excerpt"] else 0.1
                for item in context["evidence"]
            }
        else:
            excerpts = [source["excerpt"] for requirement in context["requirements"] for source in requirement["sources"]]
            decision_sources.append(excerpts)
            probability = 0.99 if any("One shape dimension can be -1" in text for text in excerpts) else 0.1
            probabilities = {question["name"]: probability for question in payload["questions"]}
        return httpx.Response(200, json={
            "model": "gpt-6-luna",
            "answers": [{"type": "predicate", "name": question["name"], "probability": probabilities[question["name"]]}
                        for question in payload["questions"]],
            "usage": {"input_tokens": 20, "output_tokens": 0, "total_tokens": 20},
        })

    monkeypatch.setattr(requests, "post", tavily_post)
    task = RetrievalTask(route="docs", query=initial_query, k=3, requirement_id="reshape",
                         requirement={"library": "numpy", "symbols": ["numpy.reshape"],
                                      "evidence_needs": ["explanation"], "match": "topic"})
    state = build_test_state({
        "request_contract": RequestContract(), "user_input": "numpy.reshape에서 -1 차원의 크기를 어떻게 정하는지 설명해줘",
        "planner_output": PlannerOutput(use_retrieval=True, tasks=[task]),
        "retry_context": {"original_tasks": [task]},
    })
    tool = _docs_tool()
    dispatch = make_retrieve_dispatch_node(tool, None, False)
    with httpx.Client(transport=httpx.MockTransport(openai_response)) as client:
        decisions = DecisionsClient(OpenAI(api_key="test-key", base_url="https://provider.test/v1",
                                          http_client=client, max_retries=0))
        planner_model = bind_structured_output(ChatOpenAI(
            model="gpt-5.6-luna", api_key="test-key", base_url="https://provider.test/v1",
            http_client=client, max_retries=0, use_responses_api=False,
        ), RetrievalRewriteOutput)
        validate = make_pre_synthesis_validation_node(False, decisions_client=decisions)
        state.update(dispatch(state))
        first = state["debug"].retrieval_diagnostics[-1]
        old_hit = state["retrieval"].hit_log[0]
        assert first.answerability == "covered"
        assert provider_queries == [(initial_query, 9)]

        # Ordinary coverage reuse remains available until validation requests refresh.
        cached = tool(rewritten_query, requirement=task.requirement, k=6,
                      previous_hits=[old_hit], completed_requests=first.completed_requests)
        assert cached.diagnostics.provider_requests == []
        assert cached.hits[0].evidence.id == old_hit.evidence.id

        state.update(validate(state))
        assert state["retry"].needs_retry
        assert state["retry"].retry_reason == "no_evidence"
        assert state["retry"].failed_requirement_ids == [task.requirement_id]
        assert state["retrieval"].evidence_verdicts[0].status == ("passed" if source_accepted else "failed")
        state.update(make_planner_node(None, False, llm_planner_retry=planner_model)(state))
        state.update(dispatch(state))

        assert provider_queries == [(initial_query, 9), (rewritten_query, 18)]
        diagnostic = state["debug"].retrieval_diagnostics[-1]
        assert not diagnostic.reused
        assert [request.query for request in diagnostic.provider_requests] == [rewritten_query]
        assert diagnostic.candidate_count == (2 if source_accepted else 1)
        fresh_hits = state["retrieval"].hit_log[state["retry"].hit_start_index:]
        assert any(hit.evidence.excerpt == fresh_text for hit in fresh_hits)
        assert any(hit.evidence.id == old_hit.evidence.id for hit in fresh_hits) == source_accepted
        assert any(hit.evidence.id == old_hit.evidence.id for hit in state["retrieval"].hit_log)
        fresh_hit = next(hit for hit in fresh_hits if hit.evidence.excerpt == fresh_text)
        assert fresh_hit.evidence.snapshot.source_uri == fresh_url
        assert fresh_hit.evidence.snapshot.snapshot_id != old_hit.evidence.snapshot.snapshot_id

        state.update(validate(state))
        assert not state["retry"].needs_retry
        assert not state["retry"].failed_requirement_ids
        assert decision_sources == (
            [[initial_text], [initial_text, fresh_text]] if source_accepted else [[], [fresh_text]]
        )


@pytest.mark.parametrize("same_query", [False, True], ids=["rewritten-query", "same-query"])
def test_semantic_rejection_refreshes_docs_alongside_another_requirement_timeout(monkeypatch, same_query):
    symbols = ["reshape", "stack", "concatenate"]
    initial_queries = {symbol: f"numpy.{symbol} behavior" for symbol in symbols}
    retry_queries = {
        symbol: initial_queries[symbol] if same_query else f"numpy.{symbol} detailed behavior"
        for symbol in symbols[:2]
    }
    initial_text = "# numpy.reshape\nnumpy.reshape(a, shape)\nGives a new shape to an array."
    fresh_text = initial_text + "\nOne shape dimension can be -1; its size is inferred from the array length and remaining dimensions."
    source_texts = {
        "reshape": initial_text,
        "stack": "# numpy.stack\nnumpy.stack(arrays, axis=0)\nJoin a sequence of arrays along a new axis.",
        "concatenate": "# numpy.concatenate\nnumpy.concatenate(arrays, axis=0)\nJoin a sequence of arrays along an existing axis.",
    }
    provider_queries = []
    source_requests = {symbol: 0 for symbol in symbols}
    decision_sources = []

    def tavily_post(url, **kwargs):
        query = kwargs["json"]["query"]
        provider_queries.append(query)
        symbol = next(symbol for symbol in symbols if f"numpy.{symbol}" in query)
        source_requests[symbol] += 1
        if symbol == "stack" and source_requests[symbol] == 1:
            raise requests.Timeout("temporary search failure")
        fresh = symbol == "reshape" and source_requests[symbol] > 1
        response = requests.Response()
        response.status_code = 200
        response.url = url
        response._content = json.dumps({"results": [{
            "url": f"https://numpy.org/doc/2.4/reference/generated/numpy.{symbol}.html",
            "title": f"numpy.{symbol}", "score": 0.7 if fresh else 0.99,
            "raw_content": fresh_text if fresh else source_texts[symbol],
        }]}).encode()
        return response

    def openai_response(request):
        payload = json.loads(request.content)
        if request.url.path == "/v1/chat/completions":
            rewrite = {"rewrites": [
                {"requirement_id": symbol, "query": query, "k": 3}
                for symbol, query in retry_queries.items()
            ]}
            return httpx.Response(200, json={
                "id": "chatcmpl-rewrite", "object": "chat.completion", "created": 1, "model": "gpt-5.6-luna",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": json.dumps(rewrite),
                }}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            })
        assert request.url.path == "/v1/decisions"
        context = json.loads(payload["input"])
        if "evidence" in context:
            # Each source is useful even when it cannot establish every fact
            # requested for that requirement. Coverage is a separate judgment.
            probabilities = {item["name"]: 0.99 for item in context["evidence"]}
        else:
            excerpts = {requirement["task"]["requirement_id"]: [source["excerpt"] for source in requirement["sources"]]
                        for requirement in context["requirements"]}
            decision_sources.append(excerpts)
            supported = {
                symbol: any("One shape dimension can be -1" in text for text in texts)
                if symbol == "reshape" else bool(texts)
                for symbol, texts in excerpts.items()
            }
            probabilities = {requirement["name"]: 0.99 if supported[requirement["task"]["requirement_id"]] else 0.1
                             for requirement in context["requirements"]}
            probabilities["request_ready"] = 0.99 if all(supported.values()) else 0.1
        return httpx.Response(200, json={
            "model": "gpt-6-luna",
            "answers": [{"type": "predicate", "name": question["name"], "probability": probabilities[question["name"]]}
                        for question in payload["questions"]],
            "usage": {"input_tokens": 20, "output_tokens": 0, "total_tokens": 20},
        })

    monkeypatch.setattr(requests, "post", tavily_post)
    tasks = [RetrievalTask(
        route="docs", query=initial_queries[symbol], k=3, requirement_id=symbol,
        requirement={"library": "numpy", "symbols": [f"numpy.{symbol}"], "evidence_needs": ["explanation"]},
    ) for symbol in symbols]
    state = build_test_state({
        "request_contract": RequestContract(),
        "user_input": "numpy.reshape의 -1 차원 추론을 설명하고 stack과 concatenate의 동작을 비교해줘",
        "planner_output": PlannerOutput(use_retrieval=True, tasks=tasks),
        "retry_context": {"original_tasks": tasks},
    })
    dispatch = make_retrieve_dispatch_node(_docs_tool(), None, False)
    with httpx.Client(transport=httpx.MockTransport(openai_response)) as client:
        decisions = DecisionsClient(OpenAI(api_key="test-key", base_url="https://provider.test/v1",
                                          http_client=client, max_retries=0))
        planner_model = bind_structured_output(ChatOpenAI(
            model="gpt-5.6-luna", api_key="test-key", base_url="https://provider.test/v1",
            http_client=client, max_retries=0, use_responses_api=False,
        ), RetrievalRewriteOutput)
        validate = make_pre_synthesis_validation_node(False, decisions_client=decisions)
        state.update(dispatch(state))
        initial_diagnostics = {item.requirement_id: item for item in state["debug"].retrieval_diagnostics}
        initial_hits = {hit.requirement_id: hit for hit in state["retrieval"].hit_log}
        assert initial_diagnostics["reshape"].answerability == "covered"
        assert initial_diagnostics["stack"].error_code == "RETRIEVAL_DOCS_TIMEOUT"
        assert initial_diagnostics["concatenate"].answerability == "covered"
        assert sorted(provider_queries) == sorted(initial_queries.values())

        state.update(validate(state))
        assert state["retry"].needs_retry
        assert state["retry"].retry_reason == "tool_error"
        assert state["retry"].failed_requirement_ids == ["reshape", "stack"]
        state.update(make_planner_node(None, False, llm_planner_retry=planner_model)(state))
        state.update(dispatch(state))

        assert sorted(provider_queries) == sorted([*initial_queries.values(), *retry_queries.values()])
        diagnostics = {item.requirement_id: item for item in state["debug"].retrieval_diagnostics[-3:]}
        for symbol, query in retry_queries.items():
            assert not diagnostics[symbol].reused
            assert [request.query for request in diagnostics[symbol].provider_requests] == [query]
        assert diagnostics["concatenate"].reused
        assert diagnostics["concatenate"].provider_requests == []
        retry_hits = {hit.requirement_id: hit for hit in state["retrieval"].hit_log[state["retry"].hit_start_index:]}
        assert retry_hits["reshape"].evidence.excerpt == fresh_text
        assert retry_hits["reshape"].evidence.snapshot.snapshot_id != initial_hits["reshape"].evidence.snapshot.snapshot_id
        assert retry_hits["concatenate"].evidence.id == initial_hits["concatenate"].evidence.id

        state.update(validate(state))
        assert not state["retry"].needs_retry
        assert not state["retry"].failed_requirement_ids
        assert decision_sources == [
            {"reshape": [initial_text], "stack": [], "concatenate": [source_texts["concatenate"]]},
            {"reshape": [initial_text, fresh_text], "stack": [source_texts["stack"]], "concatenate": [source_texts["concatenate"]]},
        ]
