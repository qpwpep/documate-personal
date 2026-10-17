"""Planner rewrites preserve search text through both request reuse layers."""
import json

import pytest
import requests

from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.infra.settings import AppSettings
from src.infra.tools.docs_search.tool import build_docs_search_tool
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.validation import make_pre_synthesis_validation_node
from tests.core.helpers import _CapturePlannerLLM, build_test_state


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
    validate = make_pre_synthesis_validation_node(False)
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
