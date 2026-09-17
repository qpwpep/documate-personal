"""Benchmark scenarios use real HTTP, graph, retrieval, and files with local model boundaries."""

from __future__ import annotations

import json
import errno
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from time import perf_counter
from types import SimpleNamespace

import pytest
import requests
from langchain_core.messages import AIMessage

from src.core.answer_schema import export_answer_text
from src.core.request_contracts import WireRequestContract
from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.judge_llm import LLMJudge
from src.eval.online_runner import _run_single_case
from src.eval.reporting.summary import build_summary
from src.eval.save_outcomes import revalidate_saved_artifacts
from tests.web.test_agent_stream_integration import agent_server
from tests.web.test_multi_upload_api import LocalChatModel, LocalEmbeddings


PREPARE_QUERY = "Compare the values in both attached files."
SAVE_QUERY = "Save the previous answer as a text file."


class ScenarioChatModel(LocalChatModel):
    """Interpret a fixed save request at the external model boundary using production messages."""

    def with_structured_output(self, schema, **kwargs):
        return ScenarioChatModel(self.controls, schema_name=schema["name"])

    def invoke(self, messages):
        if self.schema_name != "PlannerOutput":
            return super().invoke(messages)
        raw = next(message.content for message in messages if message.name == "request_context")
        context = json.loads(raw.split("\n", 1)[1])
        current = context["user_turn_ledger"][-1]
        if current["text"] != SAVE_QUERY:
            return super().invoke(messages)
        contract = WireRequestContract.model_validate({
            "evidence": [{
                "id": "save-request", "turn_id": current["turn_id"], "quote": current["text"],
                "scope": "current_request", "interpretation": "instruction",
            }],
            "body": {"kind": "copy_answer", "source": {"ref": "previous"}},
            "actions": {"save_text": {"intent": "requested", "evidence_ids": ["save-request"]}},
        })
        return {
            "parsed": {"use_retrieval": False, "tasks": [], "request_contract": contract.model_dump(mode="json")},
            "parsing_error": None,
            "raw": AIMessage(
                content="", response_metadata={"model_name": "local-test-model"},
                usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            ),
        }


@pytest.fixture
def scenario_server(agent_server, monkeypatch):
    # These nested-duration assertions need a shared clock. On Python 3.12
    # Windows, monotonic uses a coarser clock than the client's perf_counter.
    monkeypatch.setattr("src.eval.online_runner.case_runner.time", SimpleNamespace(monotonic=perf_counter))
    controls = SimpleNamespace(
        fail_synthesis=False, before_embedding=None, before_synthesis=None,
        planned_symbols=(), planned_file_ids=None,
    )
    monkeypatch.setattr("src.infra.chroma_store.OpenAIEmbeddings", lambda **kwargs: LocalEmbeddings(controls))
    monkeypatch.setattr("src.infra.llm.ChatOpenAI", lambda **kwargs: ScenarioChatModel(controls))
    return agent_server


def run_scenario(endpoint: str, root: Path, *, judge=None, config=None, **kwargs):
    case_id = kwargs.pop("case_id", "save-previous")
    if "save_expectation" not in kwargs:
        kwargs["save_expectation"] = (
            {"outcome": "required_success", "target": {"kind": "setup_answer", "setup_turn_index": 0}}
            if kwargs.get("setup_turns") else {"outcome": "must_not_execute"}
        )
    return _run_single_case(
        run_id="scenario-integration", endpoint=endpoint,
        fixtures_path=root / "fixtures" / "cases.jsonl",
        case=BenchmarkCase(
            case_id=case_id, category="tool_action", query=SAVE_QUERY,
            expected_tools=["save_text"], **kwargs,
        ),
        timeout_seconds=10, judge=judge or LLMJudge(model_name="unused", enabled=False),
        config=config or BenchmarkConfig(judge_enabled=False),
    )


def test_benchmark_retrieves_two_attachments_then_saves_the_actual_previous_answer(scenario_server):
    """A real preparation turn supplies both source citations and the exact delivered follow-up body."""
    endpoint, _, root = scenario_server
    uploads = root / "fixtures" / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "alpha.py").write_text("alpha = 1\n", encoding="utf-8")
    (uploads / "beta.py").write_text("beta = 2\n", encoding="utf-8")

    result = run_scenario(
        endpoint, root, setup_turns=[PREPARE_QUERY], upload_fixtures=["alpha.py", "beta.py"],
        require_local_citation=True, forbidden_tools=["upload_search"],
    )

    assert result.runtime_errors == result.response_errors == [], result.model_dump()
    assert len(result.scenario_turns) == 2
    preparation, delivery = result.scenario_turns
    assert preparation.query == PREPARE_QUERY
    assert delivery.query == SAVE_QUERY
    assert preparation.response is not None and delivery.response is not None
    assert result.response == delivery.response
    assert preparation.response.content == result.response.content
    assert preparation.response.citations == result.response.citations
    assert {citation.evidence.snapshot.title for citation in result.response.citations} == {"alpha.py", "beta.py"}
    assert {citation.evidence.element.metadata["file_id"] for citation in result.response.citations} == {
        file.file_id for file in preparation.upload_manifest.files
    }
    assert "upload_search" in preparation.debug["tool_calls"]
    assert "save_text" in delivery.debug["tool_calls"]
    assert result.tool_calls == ["save_text"]
    assert result.observed_hits == []
    assert result.rule_scores["reference_coverage"] == 1.0
    assert result.rule_scores["citation_traceability"] == 1.0
    assert result.rule_scores["tool_choice"] == 1.0
    assert preparation.debug["observed_hits"]
    assert preparation.request_id != delivery.request_id
    assert delivery.request_payload["uploads"] == {
        "epoch": preparation.upload_manifest.epoch,
        "revision": preparation.upload_manifest.revision,
    }
    assert {turn.request_payload["session_id"] for turn in result.scenario_turns} == {result.session_id}
    assert all("upload_file_path" not in turn.request_payload for turn in result.scenario_turns)
    receipts = [action for action in result.response.actions if action.kind == "save_text"]
    assert len(receipts) == 1 and receipts[0].status == "success"
    saved = Path(receipts[0].file_path)
    assert saved.read_bytes().decode("utf-8-sig") == export_answer_text(preparation.response, include_sources=True)
    assert saved.read_bytes().decode("utf-8-sig") == export_answer_text(result.response, include_sources=True)
    assert result.attachment_setup_ms is not None and result.attachment_setup_ms >= 0
    assert result.question_response_ms == delivery.question_response_ms
    # Each reported interval is rounded independently to the nearest millisecond.
    rounding_bound_ms = (len(result.scenario_turns) + 2) / 2
    assert result.scenario_total_ms + rounding_bound_ms >= result.attachment_setup_ms + sum(
        turn.question_response_ms for turn in result.scenario_turns
    )


def test_next_benchmark_scenario_cannot_save_another_scenarios_answer(scenario_server):
    """An isolated case has neither the earlier attachments nor its previous answer to deliver."""
    endpoint, _, root = scenario_server
    uploads = root / "fixtures" / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "alpha.py").write_text("alpha = 1\n", encoding="utf-8")
    first = run_scenario(endpoint, root, setup_turns=[PREPARE_QUERY], upload_fixtures=["alpha.py"])
    assert first.runtime_errors == first.response_errors == []
    assert len(first.response.actions) == 1 and first.response.actions[0].status == "success"
    saved = Path(first.response.actions[0].file_path)
    saved_bytes = saved.read_bytes()
    files_before = set(saved.parent.glob("*.txt"))

    second = run_scenario(endpoint, root)

    assert first.session_id != second.session_id
    assert second.runtime_errors == second.response_errors == []
    assert len(second.scenario_turns) == 1
    assert second.scenario_turns[0].upload_manifest.files == []
    assert second.response is not None
    assert second.response.citations == []
    assert not any(action.kind == "save_text" and action.status == "success" for action in second.response.actions)
    assert "save_text" not in second.tool_calls
    assert "alpha = 1" not in export_answer_text(second.response)
    assert not second.release_pass
    assert set(saved.parent.glob("*.txt")) == files_before
    assert saved.read_bytes() == saved_bytes


class _PerfectJudgeBoundary:
    def invoke(self, _messages):
        return SimpleNamespace(content=json.dumps({
            "score": 1.0, "reason": "Deliberately perfect: storage failures must still block release.",
            "subscores": {name: 1.0 for name in (
                "answer_quality", "groundedness", "citation_traceability", "tool_choice", "format_language",
            )},
        }))


def test_concurrent_scenarios_preserve_different_answers_at_the_same_time(scenario_server, monkeypatch):
    """Concurrent sessions retain distinct saved answers even when storage serializes writes."""
    import src.infra.saved_artifacts as storage
    import src.infra.tools.save_text as save_tool

    endpoint, _, root = scenario_server
    uploads = root / "fixtures" / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "alpha.py").write_text("alpha = 111\n", encoding="utf-8")
    (uploads / "beta.py").write_text("beta = 222\n", encoding="utf-8")
    fixed_time = storage.time()
    monkeypatch.setattr(storage, "time", lambda: fixed_time)
    save_ready = Barrier(2)
    real_save = save_tool.save_artifact

    def concurrent_save(*args, **kwargs):
        save_ready.wait(timeout=8)
        return real_save(*args, **kwargs)

    monkeypatch.setattr(save_tool, "save_artifact", concurrent_save)
    judge = LLMJudge(model_name="local-perfect-judge", enabled=False)
    judge.enabled = True
    judge.client = _PerfectJudgeBoundary()
    config = BenchmarkConfig(judge_enabled=True)
    config.hard_gates.p95_latency_ms = 1_000_000
    expectation = {"outcome": "required_success", "target": {"kind": "setup_answer", "setup_turn_index": 0}}

    def run_case(filename):
        return run_scenario(
            endpoint, root, case_id=f"save-{filename}", judge=judge, config=config,
            setup_turns=[PREPARE_QUERY], upload_fixtures=[filename], save_expectation=expectation,
        )

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(run_case, ["alpha.py", "beta.py"]))

    assert len({result.session_id for result in results}) == 2
    receipts = [result.response.actions[0] for result in results]
    assert len({receipt.file_path for receipt in receipts}) == 2
    assert {receipt.artifact.created_at for receipt in receipts} == {fixed_time}
    snapshots = {}
    for result, receipt in zip(results, receipts, strict=True):
        assert result.runtime_errors == result.response_errors == [], result.model_dump()
        assert result.judge_pass is result.release_pass is result.passed is True, result.model_dump()
        assert result.gate_failures == []
        assert receipt.operation.session_id == result.session_id
        expected = export_answer_text(result.scenario_turns[0].response, include_sources=True).encode("utf-8-sig")
        saved_path = Path(receipt.file_path)
        snapshots[saved_path] = expected
        assert saved_path.read_bytes() == expected
        download = requests.get(f"{endpoint}/download/{receipt.artifact.filename}", timeout=10)
        assert download.status_code == 200
        assert download.content == expected
    assert len(set(snapshots.values())) == 2

    cases = [BenchmarkCase(
        case_id=result.case_id, category="tool_action", query=SAVE_QUERY,
        setup_turns=[PREPARE_QUERY], expected_tools=["save_text"], save_expectation=expectation,
    ) for result in results]
    revalidate_saved_artifacts(cases=cases, results=results)
    assert all(result.save_assessment.phase == "run_end" and result.save_assessment.passed for result in results)
    summary = build_summary(
        run_id=results[0].run_id, endpoint=endpoint, fixtures_path="fixture", config_path="config",
        track="release", requested_limit=None, config=config, cases=cases, results=results,
    )
    assert summary.overall_passed, summary.model_dump()
    assert summary.metrics.save_contract_failures == 0
    assert len(list((root / "output" / "save_text").glob("*.txt"))) == 2
    assert all(path.read_bytes() == original for path, original in snapshots.items())


@pytest.mark.parametrize("fault,expected_failure", [
    (None, False), ("write_failed", False), ("partial_write", False),
    ("artifact_missing", False), ("artifact_mismatch", False), ("unverifiable", False),
    ("write_failed", True), ("partial_write", True),
])
def test_actual_save_and_evaluation_cannot_hide_artifact_failures(
    scenario_server, monkeypatch, fault, expected_failure,
):
    """Use the real graph, writer, HTTP bytes, judge parser, case verdict and release gates."""
    endpoint, _, root = scenario_server
    uploads = root / "fixtures" / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "alpha.py").write_text("alpha = 1\n", encoding="utf-8")
    if fault in {"write_failed", "partial_write"}:
        import src.infra.saved_artifacts as storage
        real_write = storage.os.write

        def write_with_fault(descriptor, data):
            if bytes(data).startswith(b"\xef\xbb\xbf"):
                if fault == "partial_write":
                    real_write(descriptor, data[:8])
                raise OSError(errno.ENOSPC, "injected disk full")
            return real_write(descriptor, data)

        monkeypatch.setattr(storage.os, "write", write_with_fault)
    elif fault:
        real_get = requests.get

        def observe_after_storage_fault(url, *args, **kwargs):
            if "/download/" in url:
                if fault == "unverifiable":
                    raise requests.Timeout("injected artifact observation timeout")
                path = root / "output" / "save_text" / url.rsplit("/", 1)[1]
                if fault == "artifact_missing":
                    path.unlink(missing_ok=True)
                elif fault == "artifact_mismatch":
                    path.write_bytes(b"different saved answer")
            return real_get(url, *args, **kwargs)

        monkeypatch.setattr(requests, "get", observe_after_storage_fault)
    expectation = {
        "outcome": "expected_failure" if expected_failure else "required_success",
        "target": {"kind": "setup_answer", "setup_turn_index": 0},
        "error_codes": ["write_failed"] if expected_failure else [],
    }
    judge = LLMJudge(model_name="local-perfect-judge", enabled=False)
    judge.enabled = True
    judge.client = _PerfectJudgeBoundary()
    config = BenchmarkConfig(judge_enabled=True)
    config.hard_gates.p95_latency_ms = 1_000_000
    result = run_scenario(
        endpoint, root, judge=judge, config=config,
        setup_turns=[PREPARE_QUERY], upload_fixtures=["alpha.py"], save_expectation=expectation,
    )
    expected_pass = fault is None or expected_failure
    assert result.judge_pass is True, result.model_dump()
    assert result.composite_quality_score >= 0.75
    assert result.save_assessment.passed is expected_pass, result.model_dump()
    assert result.release_pass is result.passed is expected_pass
    assert bool(result.gate_failures) is not expected_pass
    if fault in {"write_failed", "partial_write"}:
        assert result.response.actions[0].status == "error"
        assert list((root / "output" / "save_text").glob("*.txt")) == []
    elif fault:
        assert result.response.actions[0].status == "success"
    if fault == "unverifiable":
        assert result.eval_validity == "incomplete"
    case = BenchmarkCase(
        case_id=result.case_id, category="tool_action", query=SAVE_QUERY,
        setup_turns=[PREPARE_QUERY], expected_tools=["save_text"], save_expectation=expectation,
    )
    revalidate_saved_artifacts(cases=[case], results=[result])
    summary = build_summary(
        run_id=result.run_id, endpoint=endpoint, fixtures_path="fixture", config_path="config",
        track="release", requested_limit=None, config=config, cases=[case], results=[result],
    )
    assert summary.overall_passed is expected_pass, summary.model_dump()
