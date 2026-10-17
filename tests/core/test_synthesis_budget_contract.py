import json
from pathlib import Path

import pytest

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.request_contracts import BoundAnswerReference, RequestContract, TransformAnswerBody
from src.infra.tools.save_text import build_save_text_tool
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.synthesis.budgets import ExcerptLimits
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_synthesis_validation import _hit, _state


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
    return [_hit(f"# setting_{index}" + (" " if mixed and index % 2 else "\n")
                 + "상세 source information " * 900 + "\n",
                 source="upload" if mixed and index % 2 else "official", rank=index + 1)
            for index in range(8)]


@pytest.mark.parametrize("mixed,normal_total,compact_total,timeout", [(False, 6000, 3000, False), (True, 8000, 4000, True)])
def test_model_receives_bounded_excerpts_and_citations_use_the_attempt_that_answered(
    mixed, normal_total, compact_total, timeout,
):
    state = _state(_long_hits(mixed=mixed))
    normal, compact = PacketEchoModel(timeout=timeout), PacketEchoModel()

    state.update(make_synthesize_node(normal, compact, excerpt_limits=ExcerptLimits(1800, 900))(state))
    state.update(make_post_synthesis_validation_node(False)(state))

    assert 0 < sum(len(item["excerpt"]) for item in normal.packets[0]) <= normal_total
    assert all(len(item["excerpt"]) <= 1800 for item in normal.packets[0])
    assert len(compact.packets) == int(timeout)
    actual = compact.packets[0] if timeout else normal.packets[0]
    assert 0 < sum(len(item["excerpt"]) for item in actual) <= (compact_total if timeout else normal_total)
    assert all(len(item["excerpt"]) <= (900 if timeout else 1800) for item in actual)
    response = state["response"]
    assert response.kind == "answer"
    assert [item.excerpt for item in response.evidence_packet] == [item["excerpt"] for item in actual]
    assert [citation.evidence for citation in response.result.citations] == response.evidence_packet
    assert all(check.support_status == "exact_match" for check in response.result.checks)
    assert response.normal_evidence_missing_requirement_ids == []


def test_saving_a_researched_answer_keeps_the_actual_packet_and_complete_validated_body(
    tmp_path, monkeypatch,
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
        normal, compact = PacketEchoModel(timeout=True), PacketEchoModel()
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


def test_transform_retains_complete_bound_citations_beyond_the_retrieval_budget():
    original = _hit("original citation detail " * 400).evidence
    previous = finalize_answer(text_document(original.excerpt, basis="excerpt", refs=[original.id]),
                               [original], retrieval_required=True)
    previous_data = previous.model_dump(mode="json")
    state = _state(_long_hits())
    state["runtime"] = state["runtime"].model_copy(update={
        "previous_response": previous,
        "request_contract": RequestContract(body=TransformAnswerBody(
            source=BoundAnswerReference(ref="previous", response_hash=previous.content_hash),
            instruction="Retain the exact source wording.",
        )),
    })
    normal, compact = PacketEchoModel(timeout=True), PacketEchoModel()

    state.update(make_synthesize_node(normal, compact, excerpt_limits=ExcerptLimits(1800, 900))(state))
    state.update(make_post_synthesis_validation_node(False)(state))

    assert len(original.excerpt) > 6000
    assert state["response"].kind == "answer"
    packet = state["response"].evidence_packet
    assert packet[-1] == original
    retrieved_chars = sum(len(item.excerpt) for item in packet[:-1])
    assert 0 < retrieved_chars <= 3000
    assert state["response"].result.citations[-1].evidence == original
    assert normal.packets[0][-1]["excerpt"] == original.excerpt
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

    assert normal.packets
    assert compact.packets
    assert len(normal.packets) + len(compact.packets) <= 3
    assert state["response"].kind == "failure"
    assert state["response"].problem.code == "provider_unavailable"
    assert all(receipt.status != "success" for receipt in state["response"].result.actions)
    assert list(tmp_path.iterdir()) == []
