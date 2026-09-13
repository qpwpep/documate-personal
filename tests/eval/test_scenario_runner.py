"""Scenario execution through the real app client, replacing only HTTP and time."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from src.core.answer_schema import export_answer_text
from src.core.contracts.debug import DebugPayload, TokenUsage
from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.judge_llm import LLMJudge
from src.eval.online_runner import _run_single_case, run_online_benchmark
from tests.eval.response_fixtures import answer_provenance, plain_response, sse_http_response


def json_response(payload, status=200):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode()
    response._content_consumed = True
    return response


@pytest.fixture
def http_boundary(tmp_path, monkeypatch):
    """The HTTP peer enforces confirmed revisions and records observable requests."""
    state = {"requests": [], "sessions": {}, "now": 10.0, "fail_setup": False,
             "bad_manifest": False, "bad_provenance": False, "fail_sync": False, "disconnect": False}
    monkeypatch.setattr("src.infra.runtime_paths.get_project_root_path", lambda: tmp_path)
    monkeypatch.setattr("time.monotonic", lambda: state["now"])
    monkeypatch.setattr("src.app.client.perf_counter", lambda: state["now"])

    def request(method, endpoint, *, json=None, **kwargs):
        state["requests"].append((method.upper(), endpoint, json))
        session_id = endpoint.split("/sessions/")[1].split("/")[0]
        manifest = state["sessions"].setdefault(session_id, {"epoch": session_id, "revision": 0, "files": []})
        if method.lower() == "get":
            state["now"] += 0.1
            return json_response(manifest)
        assert endpoint.endswith("/uploads/sync")
        assert json["epoch"] == manifest["epoch"]
        assert json["expected_revision"] == manifest["revision"]
        if state["fail_sync"]:
            return json_response({"detail": {"code": "UPLOAD_INDEX_FAILED", "message": "index failed"}}, 503)
        if not json.get("clear"):
            for item in json["add"]:
                assert Path(item["path"]).read_text(encoding="utf-8").startswith("value =")
        manifest["revision"] += 1
        state["now"] += 0.4
        return json_response({"manifest": manifest, "changed": True})

    def post(endpoint, *, json, **kwargs):
        state["requests"].append(("POST", endpoint, json))
        if state["disconnect"]:
            raise requests.ConnectionError("lost before response")
        manifest = state["sessions"][json["session_id"]]
        assert json["uploads"] == {key: manifest[key] for key in ("epoch", "revision")}
        assert "upload_file_path" not in json
        assert json["include_debug"] is True
        turn = len([item for item in state["requests"] if item[1].endswith("/agent/stream")])
        debug = DebugPayload(token_usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
                             models_used=["fixture"], model_usage_status="llm_used").model_dump(mode="json")
        debug["extra_diagnostic"] = {"turn": turn}
        if state["fail_setup"]:
            debug["errors"] = ["setup model failed"]
        payload = {"response": plain_response("prepared body" if turn == 1 else "saved prepared body"),
                   "trace": f"Request ID: turn-{turn}", "debug": debug,
                   "upload_manifest": {"epoch": "next-epoch", "revision": 0, "files": []}}
        debug["answer_provenance"] = answer_provenance(payload["response"])
        if state["bad_provenance"]:
            debug["answer_provenance"]["response_hash"] = "another-answer"
        manifest.update(payload["upload_manifest"])
        if state["bad_manifest"]:
            payload["upload_manifest"] = {"epoch": "bad", "revision": -1}

        def chunks():
            from tests.eval.response_fixtures import sse_frame
            state["now"] += 1.0
            yield sse_frame("final_response", payload)
            raise AssertionError("Do not wait for done or replay the question")
        return sse_http_response(200, chunks=chunks(), headers={"x-request-id": f"turn-{turn}"})

    monkeypatch.setattr(requests, "request", request)
    monkeypatch.setattr(requests, "post", post)
    return state


def run_case(tmp_path, *, judge=None, **kwargs):
    return _run_single_case(run_id="scenario", endpoint="http://fixture",
                            fixtures_path=tmp_path / "fixtures" / "cases.jsonl",
                            case=BenchmarkCase(case_id="save", category="tool_action", query="save the previous answer", **kwargs),
                            timeout_seconds=5, judge=judge or LLMJudge(model_name="unused", enabled=False),
                            config=BenchmarkConfig(judge_enabled=judge is not None))


def test_preparation_and_final_turn_share_confirmed_uploads_and_preserve_diagnostics(http_boundary, tmp_path):
    """A scenario uses upload sync and the returned context for both real question requests."""
    uploads = tmp_path / "fixtures" / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "one.py").write_text("value = 1", encoding="utf-8")
    (uploads / "two.py").write_text("value = 2", encoding="utf-8")
    result = run_case(tmp_path, setup_turns=["prepare an answer"], upload_fixtures=["one.py", "two.py"])

    questions = [payload for _, endpoint, payload in http_boundary["requests"] if endpoint.endswith("/agent/stream")]
    assert [item["query"] for item in questions] == ["prepare an answer", "save the previous answer"]
    assert {item["session_id"] for item in questions} == {result.session_id}
    assert questions[0]["uploads"] == {"epoch": result.session_id, "revision": 1}
    assert questions[1]["uploads"] == {"epoch": "next-epoch", "revision": 0}
    assert [turn.query for turn in result.scenario_turns] == [item["query"] for item in questions]
    assert [turn.debug["extra_diagnostic"] for turn in result.scenario_turns] == [{"turn": 1}, {"turn": 2}]
    assert export_answer_text(result.response) == "saved prepared body"
    assert result.question_response_ms == result.latency_ms_e2e == 1000
    assert result.runtime_errors == result.response_errors == []


def test_failed_preparation_stops_before_dependent_action(http_boundary, tmp_path):
    """A failed prerequisite answer is retained diagnostically and cannot trigger the dependent action."""
    http_boundary["fail_setup"] = True
    result = run_case(tmp_path, setup_turns=["prepare an answer"])
    questions = [body for _, url, body in http_boundary["requests"] if url.endswith("/agent/stream")]
    assert [body["query"] for body in questions] == ["prepare an answer"]
    assert result.response is None
    assert result.scenario_turns[0].debug["errors"] == ["setup model failed"]
    assert any("setup turn 1" in error for error in result.runtime_errors)
    assert not result.release_pass


def test_unverified_preparation_provenance_stops_the_dependent_action(http_boundary, tmp_path):
    """A structurally valid prerequisite answer with invalid lineage cannot authorize a later action."""
    http_boundary["bad_provenance"] = True
    result = run_case(tmp_path, setup_turns=["prepare an answer"])
    questions = [body for _, url, body in http_boundary["requests"] if url.endswith("/agent/stream")]
    assert [body["query"] for body in questions] == ["prepare an answer"]
    assert result.scenario_turns[0].response is not None
    assert result.scenario_turns[0].evidence_assessment.status == "invalid"
    assert any("setup turn 1" in error for error in result.runtime_errors)
    assert not result.release_pass


def test_invalid_final_manifest_fails_evaluation_like_the_user_client(http_boundary, tmp_path):
    """A malformed attachment manifest prevents the final answer from passing evaluation."""
    http_boundary["bad_manifest"] = True
    result = run_case(tmp_path)
    assert result.response is None
    assert result.response_errors
    assert result.scenario_turns[0].raw_final_response["upload_manifest"] == {"epoch": "bad", "revision": -1}
    assert result.scenario_turns[0].debug["extra_diagnostic"] == {"turn": 1}
    assert not result.release_pass


def test_upload_sync_failure_does_not_submit_a_question(http_boundary, tmp_path):
    """Failure to confirm attachments never falls back to a legacy or attachment-free question."""
    uploads = tmp_path / "fixtures" / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "one.py").write_text("value = 1", encoding="utf-8")
    http_boundary["fail_sync"] = True
    result = run_case(tmp_path, upload_fixture="one.py")
    assert not any(url.endswith("/agent/stream") for _, url, _ in http_boundary["requests"])
    assert any("UPLOAD_INDEX_FAILED" in error for error in result.runtime_errors)
    assert result.question_response_ms is None
    assert not result.release_pass


def test_lost_question_is_not_replayed_and_next_case_has_another_session(http_boundary, tmp_path):
    """An uncertain action request is sent once and a following case has isolated state."""
    http_boundary["disconnect"] = True
    first = run_case(tmp_path)
    second = run_case(tmp_path)
    questions = [body for _, url, body in http_boundary["requests"] if url.endswith("/agent/stream")]
    assert [body["session_id"] for body in questions] == [first.session_id, second.session_id]
    assert first.session_id != second.session_id
    assert first.runtime_errors and second.runtime_errors
    assert not first.release_pass and not second.release_pass


def test_missing_fixture_is_recorded_without_losing_the_run_report(http_boundary, tmp_path):
    """A missing upload fails its case and still produces a complete diagnostic report."""
    fixture = tmp_path / "cases.jsonl"
    fixture.write_text(BenchmarkCase(case_id="missing", category="tool_action", query="read file",
                                     upload_fixture="missing.py").model_dump_json() + "\n", encoding="utf-8")
    run_dir, results, summary = run_online_benchmark(
        fixtures_path=fixture, endpoint="http://fixture", config=BenchmarkConfig(judge_enabled=False),
        config_path=tmp_path / "config.toml", output_root=tmp_path / "reports", track="smoke")
    assert len(results) == 1
    assert "upload fixture not found" in results[0].runtime_errors[0]
    assert not results[0].release_pass
    assert summary.metrics.total_cases == 1
    assert (run_dir / "raw_results.jsonl").is_file()
    assert (run_dir / "report.md").is_file()


def test_judge_receives_the_actual_prior_answer_for_a_followup_case(http_boundary, tmp_path):
    """The final answer evaluator sees the real predecessor body that the user's request refers to."""
    payloads = []

    class JudgeModelBoundary:
        def invoke(self, messages):
            payloads.append(json.loads(messages[-1].content))
            return SimpleNamespace(content=json.dumps({"score": 1, "reason": "preserved",
                "subscores": {key: 1 for key in ("answer_quality", "groundedness", "citation_traceability", "tool_choice", "format_language")}}))

    judge = LLMJudge(model_name="local-judge", enabled=False)
    judge.enabled = True
    judge.client = JudgeModelBoundary()
    result = run_case(tmp_path, setup_turns=["prepare an answer"], judge=judge)
    assert len(payloads) == 1
    assert payloads[0]["case"]["setup_turns"] == ["prepare an answer"]
    assert payloads[0]["conversation"] == [{"query": "prepare an answer", "response": plain_response("prepared body"),
                                            "observed_hits": []}]
    assert result.judge_input_complete is True
