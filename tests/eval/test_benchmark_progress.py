"""Completed cases survive interruption without becoming a completed run."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from src.app.transport_contracts import AgentResponse
from src.core.answer_schema import AnswerDocument, finalize_answer
from src.core.contracts.debug import DEBUG_SCHEMA_VERSION, DebugPayload
from src.core.contracts.provenance import AnswerProvenance
from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.online_runner.case_runner import run_online_benchmark
from src.eval.reporting.writer import load_run_outputs
from src.eval.result_models import CaseResult
from src.eval.summary_models import RunSummary


def _prepare_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, interrupt: bool = False,
                 fail_report: bool = False) -> tuple[Path, Path]:
    fixtures = tmp_path / "cases.jsonl"
    cases = [BenchmarkCase(case_id=f"case-{index}", category="tool_action", query=f"질문 {index}")
             for index in range(2)]
    fixtures.write_text("\n".join(case.model_dump_json() for case in cases) + "\n", encoding="utf-8")
    output_root = tmp_path / "runs"
    answer = finalize_answer(AnswerDocument.model_validate({
        "blocks": [{"type": "paragraph", "content": [{
            "text": "질문에 대한 답변입니다.", "basis": "interaction", "refs": [],
        }]}],
    }), [])
    question_count = 0

    def request(_session: requests.Session, method: str, url: str, **kwargs: object) -> requests.Response:
        nonlocal question_count
        response = requests.Response()
        response.status_code = 200
        response._content_consumed = True
        if method.lower() == "get":
            session_id = url.split("/sessions/", 1)[1].split("/", 1)[0]
            response._content = json.dumps({"epoch": session_id, "revision": 0, "files": []}).encode()
            response.headers["Content-Type"] = "application/json"
            return response
        question_count += 1
        run_dir, = output_root.iterdir()
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "running"
        assert manifest["completed_cases"] == question_count - 1
        if question_count == 2:
            recorded = (run_dir / "case_results.jsonl").read_text(encoding="utf-8").splitlines()
            assert [json.loads(line)["case_id"] for line in recorded] == ["case-0"]
        if question_count == 2 and interrupt:
            raise KeyboardInterrupt("controlled interruption")
        if question_count == 2 and fail_report:
            (run_dir / "report.md").mkdir()
        payload = kwargs["json"]
        assert isinstance(payload, dict)
        execution_id = f"request-{question_count}"
        body = AgentResponse(
            status="completed", response=answer, execution_id=execution_id, trace=f"Execution ID: {execution_id}",
            upload_manifest={"epoch": payload["session_id"], "revision": 0, "files": []},
            debug=DebugPayload(
                schema_version=DEBUG_SCHEMA_VERSION, observability_status="ok", route_decisions=[],
                answer_provenance=AnswerProvenance(
                    body_kind="compose", response_hash=answer.content_hash, evidence_packet=[],
                ),
                llm_calls=[],
                execution_evidence={"schema_version": 1, "execution_id": execution_id,
                                    "status": "complete", "events": []},
            ),
        )
        response._content = ("event: final_response\ndata: " + body.model_dump_json() + "\n\n").encode()
        response.headers["Content-Type"] = "text/event-stream"
        return response

    monkeypatch.setattr(requests.sessions.Session, "request", request)
    return fixtures, output_root


def _run(fixtures: Path, output_root: Path) -> tuple[Path, list[CaseResult], RunSummary]:
    return run_online_benchmark(
        fixtures_path=fixtures, endpoint="http://benchmark.invalid", config=BenchmarkConfig(judge_enabled=False),
        config_path=fixtures.parent / "config.toml", output_root=output_root, track="smoke",
    )


def test_interrupted_run_retains_finished_case_and_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixtures, output_root = _prepare_run(tmp_path, monkeypatch, interrupt=True)

    with pytest.raises(KeyboardInterrupt, match="controlled interruption"):
        _run(fixtures, output_root)

    run_dir, = output_root.iterdir()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run_dir / "case_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["status"] == "interrupted"
    assert manifest["case_ids"] == ["case-0", "case-1"]
    assert manifest["completed_cases"] == 1
    assert [record["case_id"] for record in records] == ["case-0"]
    assert records[0]["response_text"] == "질문에 대한 답변입니다."
    assert not (run_dir / "summary.json").exists()
    assert not (output_root / "latest_smoke_run.txt").exists()


def test_completed_run_publishes_final_outputs_after_case_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixtures, output_root = _prepare_run(tmp_path, monkeypatch)

    run_dir, results, summary = _run(fixtures, output_root)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run_dir / "case_results.jsonl").read_text(encoding="utf-8").splitlines()]
    saved_summary, saved_results = load_run_outputs(run_dir)
    assert manifest["status"] == "completed"
    assert manifest["completed_cases"] == 2
    assert manifest["finished_at_utc"]
    assert [record["case_id"] for record in records] == ["case-0", "case-1"]
    assert [result.case_id for result in saved_results] == [result.case_id for result in results]
    assert saved_summary == summary
    assert (output_root / "latest_smoke_run.txt").read_text(encoding="utf-8").strip() == summary.run_id


def test_failed_final_write_is_not_loadable_as_completed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixtures, output_root = _prepare_run(tmp_path, monkeypatch, fail_report=True)

    with pytest.raises(OSError):
        _run(fixtures, output_root)

    run_dir, = output_root.iterdir()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["completed_cases"] == 2
    assert len((run_dir / "case_results.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert not (output_root / "latest_smoke_run.txt").exists()
    with pytest.raises(ValueError, match="not completed"):
        load_run_outputs(run_dir)
