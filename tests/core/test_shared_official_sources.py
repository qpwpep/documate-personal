"""Concurrent documentation tasks can independently inspect one captured page."""

import json

import httpx
import pytest
from openai import OpenAI

from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.infra.decisions import DecisionsClient
from src.infra.tools._common import build_retrieval_result
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from tests.core.helpers import build_test_state
from tests.core.test_decisions_validation import read_decision_input
from tests.synthesis_fixtures import prepare_synthesis_state


def _retrieve_shared_page(*, parser="official_html", capture_scope="full_document",
                          source_uri="https://docs.python.org/3/library/codec.html", library="Python"):
    first = "codec.read(data)¶\n\nRead the supplied data.\n\n"
    condition = "Return the input without its leading marker when that marker matches."
    text = first + "Other reference details.\n\n" * 180 + "codec.trim(data, marker)¶\n\n" + condition + "\n"
    snapshot = build_snapshot(source_uri=source_uri, title="codec reference", media_type="text/html",
        source_type="official", content=text, parser=parser, parser_version="1", capture_scope=capture_scope)
    evidence = build_evidence(snapshot=snapshot, element=DocumentElement(element_id="body", kind="paragraph", text=text),
                              end=len(first))
    hit = SearchHit(evidence=evidence, rank=1, score={"metric": "provider", "direction": "higher"})
    tasks = [RetrievalTask(requirement_id=name, route="docs", query=f"codec.{name} behavior", k=1,
        requirement={"library": "Python" if name == "read" else library, "symbols": [f"codec.{name}"],
                     "evidence_needs": ["explanation"]}) for name in ("read", "trim")]

    def search(query, **kwargs):
        hits = [hit] if kwargs["requirement"].symbols == ["codec.read"] else []
        return build_retrieval_result(tool="tavily_search", route="docs", query=query, hits=hits,
                                      status="success" if hits else "no_result")

    state = build_test_state({"user_input": "Compare codec.read and codec.trim.",
        "request_contract": RequestContract(request_id="shared-docs"),
        "planner_output": PlannerOutput(use_retrieval=True, tasks=tasks)})
    state.update(make_retrieve_dispatch_node(search, None, False)(state))
    return state, evidence, condition


def test_completed_batch_reselects_another_api_from_the_captured_official_page():
    state, original, condition = _retrieve_shared_page()

    hits = [hit for hit in state["retrieval"].hit_log if hit.requirement_id == "trim"]
    assert any(condition in hit.evidence.excerpt for hit in hits)
    assert len(hits) <= 6
    assert all(hit.evidence.snapshot == original.snapshot for hit in hits)
    assert all(0 < len(hit.evidence.excerpt) <= 3200 for hit in hits)
    assert all(hit.evidence.excerpt == original.element.text[hit.evidence.selection.start:hit.evidence.selection.end]
               for hit in hits)
    diagnostic = next(item for item in state["debug"].retrieval_diagnostics if item.requirement_id == "trim")
    assert diagnostic.status == "success"
    assert diagnostic.answerability == "unknown"
    assert diagnostic.evidence_count == diagnostic.candidate_count == len(hits)
    assert diagnostic.provider_requests == []
    assert diagnostic.provider_result_count == 0


@pytest.mark.parametrize("changes", [
    {"parser": "provider", "capture_scope": "provider_excerpt"},
    {"capture_scope": "provider_excerpt"},
    {"source_uri": "https://untrusted.example/3/library/codec.html"},
    {"library": "numpy"},
    {"library": "unsupported"},
])
def test_shared_sources_keep_capture_and_library_boundaries(changes):
    state, _, _ = _retrieve_shared_page(**changes)

    assert not [hit for hit in state["retrieval"].hit_log if hit.requirement_id == "trim"]


def test_shared_candidates_need_a_new_semantic_decision_for_the_target_requirement():
    state, _, condition = _retrieve_shared_page()
    observed_targets = []

    def respond(request):
        payload = json.loads(request.content)
        context = read_decision_input(payload["input"])
        entries = context.get("evidence", [])
        rejected = {entry["name"] for entry in entries if entry["task"]["requirement_id"] == "trim"}
        observed_targets.extend(entry["source"]["excerpt"] for entry in entries
                                if entry["task"]["requirement_id"] == "trim")
        answers = [{"type": "predicate", "name": question["name"],
                    "probability": 0.1 if question["name"].split("_support_")[0] in rejected else 0.99}
                   for question in payload["questions"]]
        return httpx.Response(200, json={"model": "gpt-6-luna", "answers": answers,
            "usage": {"input_tokens": 32, "output_tokens": 0, "total_tokens": 32}})

    boundary = DecisionsClient(OpenAI(api_key="test-key", base_url="https://provider.test/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    prepare_synthesis_state(state, decisions=boundary)

    assert any(condition in excerpt for excerpt in observed_targets)
    packet = state["retrieval"].prepared_evidence.normal
    assert "trim" in packet.failed_requirement_ids
    assert all("trim" not in requirements for requirements in packet.requirement_map.values())
