"""Retrieval recovery keeps evidence ownership and retries only recoverable searches."""

import pytest

from src.core.contracts import RetryState
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from tests.core.helpers import build_test_state


@pytest.mark.parametrize("retry_reason", [None, "tool_error", "no_evidence"])
def test_dispatch_retries_timeout_after_partial_success_without_repeating_completed_http_query(monkeypatch, retry_reason):
    """A successful partial result is not a completed search when its next provider request failed."""
    import json
    import requests

    from src.infra.settings import AppSettings
    from src.infra.tools.docs_search.tool import build_docs_search_tool

    queries = []
    source_url = "https://numpy.org/doc/2.4/reference/generated/numpy.reshape.html"

    def response(url, data):
        value = requests.Response()
        value.status_code = 200
        value.url = url
        value._content = json.dumps(data).encode()
        value._content_consumed = True
        return value

    def post(url, **kwargs):
        queries.append(kwargs["json"]["query"])
        if len(queries) == 2:
            raise requests.Timeout("provider timeout")
        aspect = "order" if len(queries) == 1 else "copy"
        return response(url, {"results": [{
            "url": source_url, "title": "numpy.reshape", "score": 0.9,
            "raw_content": f"# numpy.reshape\nnumpy.reshape(a, shape, {aspect}=None)\n{aspect}: Controls array layout.\n",
        }]})

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "head", lambda url, **kwargs: response(url, {}))
    tool = build_docs_search_tool(AppSettings(_env_file=None, tavily_api_key="test", openai_api_key="test"))
    dispatch = make_retrieve_dispatch_node(tool, None, False)
    task = RetrievalTask(route="docs", query="numpy.reshape order", k=4,
                         requirement={"library": "numpy", "symbols": ["numpy.reshape"], "aspects": ["order", "copy"]})
    state = build_test_state({"request_contract": RequestContract(),
                              "planner_output": PlannerOutput(use_retrieval=True, tasks=[task])})
    state.update(dispatch(state))
    first_hit = state["retrieval"].hit_log[0]
    first = state["debug"].retrieval_diagnostics[-1]
    assert (first.status, first.answerability, first.error_code) == ("success", "partial", "RETRIEVAL_DOCS_TIMEOUT")
    state["retry"] = RetryState(attempt=1, retry_reason=retry_reason, failed_requirement_ids=[task.requirement_id])
    state.update(dispatch(state))
    final = state["debug"].retrieval_diagnostics[-1]
    assert (final.reused, final.answerability, final.error_code) == (False, "covered", None)
    assert len(queries) == 3
    assert queries[0] == task.query
    assert queries[1] == queries[2]
    assert queries[1] == task.query + " copy"
    assert first_hit.evidence in [hit.evidence for hit in state["retrieval"].hit_log[1:]]
