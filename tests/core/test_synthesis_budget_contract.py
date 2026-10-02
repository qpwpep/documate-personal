import json
from pathlib import Path

import pytest

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import BoundAnswerReference, RequestContract, TransformAnswerBody
from src.infra.tools.save_text import build_save_text_tool
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.synthesis.budgets import ExcerptLimits, resolve_evidence_budgets
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_synthesis_validation import _hit, _state


@pytest.mark.parametrize(
    "routes,use_retrieval,totals",
    [([], False, (6000, 3000)), (["docs"], True, (6000, 3000)),
     (["upload"], True, (6000, 3000)), (["docs", "upload"], True, (8000, 4000))],
)
def test_resolved_budgets_preserve_reachable_normal_and_recovery_limits(routes, use_retrieval, totals):
    plan = PlannerOutput(use_retrieval=use_retrieval,
                         tasks=[RetrievalTask(route=route, query="settings", k=1) for route in routes])

    normal, compact = resolve_evidence_budgets(plan=plan, limits=ExcerptLimits(1800, 900))

    assert (normal.max_total_excerpt_chars, compact.max_total_excerpt_chars) == totals
    assert (normal.max_excerpt_chars, compact.max_excerpt_chars) == (1800, 900)
    assert normal.max_items == compact.max_items == 8


@pytest.mark.parametrize("normal_chars,compact_chars", [(80, 900), (2400, 1200), (9000, 10000)])
def test_recovery_never_expands_snippets_and_normal_limit_is_not_preclamped(normal_chars, compact_chars):
    normal, compact = resolve_evidence_budgets(
        plan=PlannerOutput(use_retrieval=False, tasks=[]),
        limits=ExcerptLimits(normal_chars, compact_chars),
    )

    assert normal.max_excerpt_chars == normal_chars
    assert compact.max_excerpt_chars == min(normal_chars, compact_chars)
    assert (normal.max_total_excerpt_chars, compact.max_total_excerpt_chars) == (6000, 3000)


def test_item_capacity_counts_independent_file_aspect_passages_without_growing_text_budget():
    tasks = [RetrievalTask(route="upload", query=f"compare {symbol}", k=4,
                           requirement={"file_ids": [f"file-{index}" for index in range(5)],
                                        "aspects": ["setup", "cleanup"], "symbols": [symbol]})
             for symbol in ("first", "second")]

    normal, compact = resolve_evidence_budgets(
        plan=PlannerOutput(use_retrieval=True, tasks=tasks), limits=ExcerptLimits(1800, 900),
    )

    assert normal.max_items == compact.max_items == 20
    assert (normal.max_total_excerpt_chars, compact.max_total_excerpt_chars) == (6000, 3000)


class PacketEchoModel:
    """Replace only the model boundary; preparation, citations and validation stay real."""

    def __init__(self, *, timeout=False):
        self.timeout = timeout
        self.packets = []

    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, messages, *, timeout=None):
        packet_message = next(str(message.content) for message in messages
                              if str(message.content).startswith("[Evidence Packet]"))
        packet = json.loads(packet_message[packet_message.index("\n[") + 1:])
        self.packets.append(packet)
        if self.timeout:
            raise TimeoutError("structured timeout")
        return {"blocks": [{"type": "paragraph", "content": [
            {"text": item["excerpt"], "basis": "excerpt", "refs": [item["id"]]}
            for item in packet
        ]}]}


def _long_hits(*, mixed=False):
    return [_hit(f"# setting_{index} " + "상세 source information " * 900 + "\n",
                 source="upload" if mixed and index % 2 else "official", rank=index + 1)
            for index in range(8)]


@pytest.mark.parametrize("mixed,normal_total,compact_total", [(False, 6000, 3000), (True, 8000, 4000)])
@pytest.mark.parametrize("timeout", [False, True])
def test_model_receives_bounded_excerpts_and_citations_use_the_attempt_that_answered(
    mixed, normal_total, compact_total, timeout,
):
    state = _state(_long_hits(mixed=mixed))
    normal, compact = PacketEchoModel(timeout=timeout), PacketEchoModel()

    state.update(make_synthesize_node(normal, compact, excerpt_limits=ExcerptLimits(1800, 900))(state))
    state.update(make_post_synthesis_validation_node(False)(state))

    assert sum(len(item["excerpt"]) for item in normal.packets[0]) == normal_total
    assert all(len(item["excerpt"]) <= 1800 for item in normal.packets[0])
    assert len(compact.packets) == int(timeout)
    actual = compact.packets[0] if timeout else normal.packets[0]
    assert sum(len(item["excerpt"]) for item in actual) == (compact_total if timeout else normal_total)
    assert all(len(item["excerpt"]) <= (900 if timeout else 1800) for item in actual)
    response = state["response"]
    assert response.kind == "answer"
    assert [item.excerpt for item in response.evidence_packet] == [item["excerpt"] for item in actual]
    assert [citation.evidence for citation in response.result.citations] == response.evidence_packet
    assert all(check.support_status == "exact_match" for check in response.result.checks)
    assert response.normal_evidence_missing_requirement_ids == []


@pytest.mark.parametrize("timeout", [False, True])
def test_saving_a_researched_answer_keeps_the_actual_packet_and_complete_validated_body(
    tmp_path, monkeypatch, timeout,
):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    results = []
    packets = []
    for save in (False, True):
        state = _state(_long_hits())
        contract = RequestContract.model_validate({
            "actions": {"save_text": {"intent": "requested" if save else "not_requested",
                                      "evidence_ids": ["save"] if save else []}},
            "evidence": ([{"id": "save", "turn_id": "current", "quote": "save the answer",
                           "scope": "actions.save_text", "interpretation": "instruction"}] if save else []),
        })
        state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract})
        normal, compact = PacketEchoModel(timeout=timeout), PacketEchoModel()
        state.update(make_synthesize_node(normal, compact, excerpt_limits=ExcerptLimits(1800, 900))(state))
        state.update(make_post_synthesis_validation_node(False)(state))
        assert state["response"].kind == "answer"
        packets.append(state["response"].evidence_packet)
        results.append(export_answer_text(state["response"].result, include_sources=True))
        state.update(make_action_postprocess_node(build_save_text_tool(), lambda **_kwargs: {}, False)(state))
        if save:
            receipt = state["response"].result.actions[0]
            assert receipt.status == "success"
            assert receipt.verification == "verified"
            assert Path(receipt.file_path).read_text(encoding="utf-8-sig") == results[-1]
        else:
            assert list(tmp_path.iterdir()) == []

    assert packets[0] == packets[1]
    assert results[0] == results[1]


@pytest.mark.parametrize("timeout", [False, True])
@pytest.mark.parametrize("retrieve", [False, True])
def test_transform_retains_complete_bound_citations_beyond_the_retrieval_budget(timeout, retrieve):
    original = _hit("original citation detail " * 400).evidence
    previous = finalize_answer(text_document(original.excerpt, basis="excerpt", refs=[original.id]),
                               [original], retrieval_required=True)
    previous_data = previous.model_dump(mode="json")
    state = _state(_long_hits() if retrieve else [])
    state["runtime"] = state["runtime"].model_copy(update={
        "previous_response": previous,
        "request_contract": RequestContract(body=TransformAnswerBody(
            source=BoundAnswerReference(ref="previous", response_hash=previous.content_hash),
            instruction="Retain the exact source wording.",
        )),
    })
    normal, compact = PacketEchoModel(timeout=timeout), PacketEchoModel()

    state.update(make_synthesize_node(normal, compact, excerpt_limits=ExcerptLimits(1800, 900))(state))
    state.update(make_post_synthesis_validation_node(False)(state))

    assert len(original.excerpt) > 6000
    assert state["response"].kind == "answer"
    packet = state["response"].evidence_packet
    assert packet[-1] == original
    assert sum(len(item.excerpt) for item in packet[:-1]) == ((3000 if timeout else 6000) if retrieve else 0)
    assert state["response"].result.citations[-1].evidence == original
    assert normal.packets[0][-1]["excerpt"] == original.excerpt
    if timeout:
        assert compact.packets[0][-1]["excerpt"] == original.excerpt
    assert previous.model_dump(mode="json") == previous_data


def test_exhausted_recovery_does_not_save_an_incomplete_source_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    state = _state(_long_hits())
    contract = RequestContract.model_validate({
        "actions": {"save_text": {"intent": "requested", "evidence_ids": ["save"]}},
        "evidence": [{"id": "save", "turn_id": "current", "quote": "save the answer",
                      "scope": "actions.save_text", "interpretation": "instruction"}],
    })
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract})
    normal, compact = PacketEchoModel(timeout=True), PacketEchoModel(timeout=True)

    state.update(make_synthesize_node(normal, compact, excerpt_limits=ExcerptLimits(1800, 900))(state))
    state.update(make_post_synthesis_validation_node(False)(state))
    state.update(make_action_postprocess_node(build_save_text_tool(), lambda **_kwargs: {}, False)(state))

    assert len(normal.packets) == 1
    assert len(compact.packets) == 2
    assert state["response"].kind == "failure"
    assert state["response"].problem.code == "provider_unavailable"
    assert all(receipt.status != "success" for receipt in state["response"].result.actions)
    assert list(tmp_path.iterdir()) == []
