"""Run the entire release dataset through the client with an HTTP test peer.

This is a transport/schema/staging smoke, not a model-quality evaluation. The
peer deliberately emits a neutral response and no tool receipts, so it cannot
be used as evidence that the product meets the new cases' semantic oracles.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from uuid import uuid4

import requests
import pytest

from src.core.contracts.debug import DebugPayload, TokenUsage
from src.eval.config_models import BenchmarkConfig
from src.eval.io import load_cases_jsonl
from src.eval.online_runner import run_online_benchmark
from src.eval.release_dataset import promote
from tests.eval.response_fixtures import answer_provenance, plain_response, sse_http_response


FIXTURES = Path(__file__).resolve().parents[2] / "data/benchmarks/fixtures/cases.generated.jsonl"
SCENARIO_PLAN = {
    "docs_only": {"standard": 20, "boundary": 8, "regression": 2},
    "rag_only": {"standard": 14, "boundary": 6, "injection": 8, "regression": 2},
    "hybrid": {"standard": 14, "boundary": 6, "injection": 8, "regression": 2},
    "tool_action": {"standard": 12, "boundary": 4, "correction": 8, "failure": 4, "regression": 2},
}


def test_release_distribution_matches_the_authored_plan():
    cases = load_cases_jsonl(FIXTURES)
    assert len(cases) == 120
    assert Counter(case.category for case in cases) == {category: 30 for category in SCENARIO_PLAN}
    assert Counter(case.evaluation_role for case in cases) == {"new_evaluation": 112, "public_regression": 8}
    for category, scenarios in SCENARIO_PLAN.items():
        category_cases = [case for case in cases if case.category == category]
        assert Counter(case.scenario for case in category_cases) == scenarios
        assert Counter(case.difficulty for case in category_cases) == {"easy": 8, "medium": 14, "hard": 8}


@pytest.mark.parametrize("deployment", ["legacy", "approved_default", "approved_custom"])
def test_all_120_release_cases_execute_through_real_loader_client_and_runner(tmp_path, monkeypatch, deployment):
    cases = load_cases_jsonl(FIXTURES)
    execution = FIXTURES
    review = FIXTURES.parent.parent / "design/release_review.json"
    if deployment == "approved_custom":
        execution = tmp_path / "deployment/cases.jsonl"
        shutil.copytree(FIXTURES.parent / "uploads", execution.parent / "uploads")
        promote(FIXTURES, review=review, out=execution)
    sessions: dict[str, dict] = {}
    questions: dict[str, list[dict]] = {}
    staged: dict[str, dict[str, str]] = {}
    cleared: set[str] = set()
    # Storage remains real and uses a per-test temporary project root.
    monkeypatch.setattr("src.infra.runtime_paths.get_project_root_path", lambda: tmp_path)

    def json_response(payload):
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(payload).encode("utf-8")
        response._content_consumed = True
        return response

    def request(method, endpoint, *, json=None, **kwargs):
        assert endpoint.startswith("http://dataset-peer/sessions/"), endpoint
        session_id = endpoint.split("/sessions/", 1)[1].split("/", 1)[0]
        manifest = sessions.setdefault(session_id, {"epoch": uuid4().hex, "revision": 0, "files": []})
        if method.upper() == "GET":
            assert endpoint.endswith("/uploads")
            return json_response(manifest)
        assert method.upper() == "POST" and endpoint.endswith("/uploads/sync")
        assert json["epoch"] == manifest["epoch"]
        assert json["expected_revision"] == manifest["revision"]
        if json.get("clear"):
            manifest["files"] = []
            cleared.add(session_id)
        else:
            assert not manifest["files"]
            staged[session_id] = {}
            for addition in json["add"]:
                source = Path(addition["path"])
                assert source.is_relative_to(tmp_path)
                content = source.read_bytes()
                assert content
                digest = hashlib.sha256(content).hexdigest()
                staged[session_id][addition["name"]] = digest
                manifest["files"].append({
                    "file_id": uuid4().hex, "name": addition["name"], "size_bytes": len(content),
                    "content_hash": "sha256:" + digest,
                    "source_uri": f"upload://{session_id}/{addition['name']}",
                })
        manifest["revision"] += 1
        return json_response({"manifest": manifest, "changed": True})

    def post(endpoint, *, json, **kwargs):
        assert endpoint == "http://dataset-peer/agent/stream"
        session_id = json["session_id"]
        manifest = sessions[session_id]
        assert json["include_debug"] is True
        assert json["uploads"] == {key: manifest[key] for key in ("epoch", "revision")}
        questions.setdefault(session_id, []).append(json.copy())
        manifest["revision"] += 1
        response = plain_response("HTTP 경계 검사 응답입니다. 모델 품질을 평가하는 답변이 아닙니다.")
        debug = DebugPayload(
            token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            models_used=["http-boundary-fixture"], model_usage_status="llm_used",
        ).model_dump(mode="json")
        debug["answer_provenance"] = answer_provenance(response)
        return sse_http_response(200, {
            "response": response, "trace": "dataset-http-boundary", "debug": debug,
            "upload_manifest": manifest,
        }, headers={"x-request-id": uuid4().hex})

    def prohibit_real_network(*args, **kwargs):
        raise AssertionError("The dataset transport smoke must never perform a real network request")

    monkeypatch.setattr(requests, "request", request)
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests.sessions.Session, "request", prohibit_real_network)
    run_dir, results, summary = run_online_benchmark(
        fixtures_path=execution, endpoint="http://dataset-peer",
        release_review=review if deployment.startswith("approved_") else None,
        config=BenchmarkConfig(judge_enabled=False), config_path=tmp_path / "unused-config.toml",
        output_root=tmp_path / "runs", track="smoke",
    )
    assert len(cases) == len(results) == summary.metrics.total_cases == 120
    assert len(sessions) == len(questions) == len({result.session_id for result in results}) == 120
    assert {result.case_id for result in results} == {case.case_id for case in cases}
    assert sum(map(len, questions.values())) == 120 + sum(len(case.setup_turns) for case in cases)
    for case, result in zip(cases, results, strict=True):
        assert result.runtime_errors == result.response_errors == [], result.model_dump()
        assert result.cleanup_errors == []
        assert result.http_status == 200 and result.response is not None
        expected_queries = [*case.setup_turns, case.query]
        assert [payload["query"] for payload in questions[result.session_id]] == expected_queries
        assert [turn.query for turn in result.scenario_turns] == expected_queries
        assert len({turn.request_id for turn in result.scenario_turns}) == len(expected_queries)
        expected_hashes = {name: hashlib.sha256((FIXTURES.parent / "uploads" / name).read_bytes()).hexdigest()
                           for name in case.resolved_upload_fixtures}
        assert result.attachment_fingerprints == expected_hashes
        assert staged.get(result.session_id, {}) == {Path(name).name: digest for name, digest in expected_hashes.items()}
        if case.resolved_upload_fixtures:
            assert result.session_id in cleared
        assert result.judge_status == "disabled"
        assert result.release_pass is False
    assert (run_dir / "raw_results.jsonl").is_file()
    assert len((run_dir / "raw_results.jsonl").read_text(encoding="utf-8").splitlines()) == 120
    assert (run_dir / "report.md").is_file()
    approval = summary.audit_metrics["dataset_approval"]
    assert approval["status"] == ("verified" if deployment.startswith("approved_") else "not_verified")
    if deployment.startswith("approved_"):
        assert approval["candidate_sha256"] == hashlib.sha256(FIXTURES.read_bytes()).hexdigest()
