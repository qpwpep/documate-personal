"""Promotion carries its exact approval to subsequent release runs."""

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from uuid import uuid4

import pytest
import requests

from src.core.contracts.debug import DebugPayload, TokenUsage
from src.eval.config_models import BenchmarkConfig
from src.eval.online_runner import run_online_benchmark
from src.eval.release_dataset import promote, sha256
from tests.eval.response_fixtures import answer_provenance, plain_response, sse_http_response
from tests.eval.test_release_promotion import reviewed_candidates


@pytest.fixture
def release_http(tmp_path, monkeypatch):
    """Keep approval, staging, judge parsing and reporting real; fake services."""
    manifests, staged, queries, judgments = {}, [], [], []
    monkeypatch.setattr("src.infra.runtime_paths.get_project_root_path", lambda: tmp_path)

    def invoke(messages):
        judgments.append(messages)
        return SimpleNamespace(content=json.dumps({
            "score": 1.0, "reason": "검수 메모 설명",
            "subscores": {name: 1.0 for name in (
                "answer_quality", "groundedness", "citation_traceability", "tool_choice", "format_language",
            )},
        }))

    monkeypatch.setattr("src.eval.judge_llm.ChatOpenAI", lambda **kwargs: SimpleNamespace(invoke=invoke))

    def response_json(payload):
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(payload).encode("utf-8")
        response._content_consumed = True
        return response

    def request(method, url, *, json=None, **kwargs):
        session = url.split("/sessions/", 1)[1].split("/", 1)[0]
        manifest = manifests.setdefault(session, {"epoch": uuid4().hex, "revision": 0, "files": []})
        if method.upper() == "GET":
            return response_json(manifest)
        assert method.upper() == "POST"
        if json.get("clear"):
            manifest["files"] = []
        else:
            for addition in json["add"]:
                content = Path(addition["path"]).read_bytes()
                staged.append(content)
                manifest["files"].append({
                    "file_id": uuid4().hex, "name": addition["name"], "size_bytes": len(content),
                    "content_hash": "sha256:" + hashlib.sha256(content).hexdigest(),
                    "source_uri": f"upload://{session}/{addition['name']}",
                })
        manifest["revision"] += 1
        return response_json({"manifest": manifest, "changed": True})

    def post(url, *, json, **kwargs):
        queries.append(json["query"])
        body = plain_response("검수 메모를 설명했습니다.")
        debug = DebugPayload(
            token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            models_used=["http-boundary-fixture"], model_usage_status="llm_used",
        ).model_dump(mode="json")
        debug["answer_provenance"] = answer_provenance(body)
        return sse_http_response(200, {
            "response": body, "debug": debug, "trace": "promotion-test",
            "upload_manifest": manifests[json["session_id"]],
        }, headers={"x-request-id": uuid4().hex})

    def unexpected_request(*args, **kwargs):
        pytest.fail("Unconfigured network boundary")

    monkeypatch.setattr(requests.sessions.Session, "request", unexpected_request)
    monkeypatch.setattr(requests, "request", request)
    monkeypatch.setattr(requests, "post", post)
    return SimpleNamespace(staged=staged, queries=queries, judgments=judgments)


def promotion_package(tmp_path, monkeypatch, *, default):
    source = tmp_path / "source"
    source.mkdir()
    candidates, review, design, rows = reviewed_candidates(source, with_attachment=True)
    audit = design / "similarity_report.json"
    audit.write_bytes(b'{"findings": []}\n')
    review_data = json.loads(review.read_bytes())
    review_data["audit_artifacts"] = {"similarity_report.json": sha256(audit)}
    review.write_bytes(json.dumps(review_data).encode("utf-8"))

    execution = tmp_path / "deployment" / ("cases.generated.jsonl" if default else "custom.jsonl")
    shutil.copytree(source / "uploads", execution.parent / "uploads")
    legacy_design = tmp_path / "legacy-design"
    shutil.copytree(design, legacy_design)
    (legacy_design / "release_review.json").write_bytes(review.read_bytes())
    monkeypatch.setattr("src.eval.release_dataset.DEFAULT_DESIGN", legacy_design)
    monkeypatch.setattr("src.eval.release_dataset.DEFAULT_RELEASE", execution if default else tmp_path / "default.jsonl")
    if default:
        execution.write_bytes(candidates.read_bytes())

    # Preserve all authored semantics while approving a different byte sequence.
    lines = candidates.read_bytes().splitlines(keepends=True)
    lines[-2], lines[-1] = lines[-1], lines[-2]
    candidates.write_bytes(b"".join(lines))
    review_data["candidate_sha256"] = sha256(candidates)
    review.write_bytes(json.dumps(review_data).encode("utf-8"))
    return SimpleNamespace(
        source=source, candidates=candidates, review=review, design=design, execution=execution,
        rows=rows, upload=execution.parent / "uploads" / rows[0]["upload_fixtures"][0],
        candidate_sha256=sha256(candidates), review_sha256=sha256(review),
    )


def run_release(package, tmp_path, *, track="release", **kwargs):
    return run_online_benchmark(
        fixtures_path=package.execution, endpoint="http://promotion-fixture", config=BenchmarkConfig(),
        config_path=tmp_path / "config.toml", output_root=tmp_path / "runs", track=track, limit=1,
        **kwargs,
    )


@pytest.mark.parametrize("default", [True, False], ids=["default", "custom"])
@pytest.mark.parametrize("remove_source", [False, True], ids=["source-present", "source-removed"])
def test_promoted_release_automatically_uses_its_approval(
    tmp_path, monkeypatch, release_http, default, remove_source,
):
    package = promotion_package(tmp_path, monkeypatch, default=default)
    expected_upload = package.upload.read_bytes()
    assert promote(package.candidates, review=package.review, out=package.execution, design=package.design)["errors"] == []
    if remove_source:
        shutil.rmtree(package.source)
    outside = tmp_path / "unrelated-working-directory"
    outside.mkdir()
    monkeypatch.chdir(outside)

    run_dir, results, summary = run_release(package, tmp_path)

    approval = summary.audit_metrics["dataset_approval"]
    assert approval["status"] == "verified"
    assert approval["candidate_sha256"] == package.candidate_sha256
    assert approval["review_sha256"] == package.review_sha256
    assert release_http.queries == [package.rows[0]["query"]]
    assert release_http.staged == [expected_upload]
    assert len(release_http.judgments) == 1
    assert results[0].runtime_errors == results[0].response_errors == []
    assert results[0].judge_status == "succeeded"
    assert json.loads((run_dir / "summary.json").read_bytes())["audit_metrics"]["dataset_approval"] == approval


@pytest.mark.parametrize("changed", ["candidate", "upload"])
def test_promoted_release_rejects_changed_deployment_before_http(tmp_path, monkeypatch, changed):
    package = promotion_package(tmp_path, monkeypatch, default=False)
    promote(package.candidates, review=package.review, out=package.execution, design=package.design)
    target = package.execution if changed == "candidate" else package.upload
    target.write_bytes(target.read_bytes() + b"\n")

    def network_forbidden(*args, **kwargs):
        pytest.fail("Changed approved bytes must be rejected before any HTTP request")

    monkeypatch.setattr(requests.sessions.Session, "request", network_forbidden)
    monkeypatch.setattr(requests, "request", network_forbidden)
    monkeypatch.setattr(requests, "post", network_forbidden)
    monkeypatch.setattr("src.eval.judge_llm.ChatOpenAI", network_forbidden)
    with pytest.raises(ValueError, match="approval|approved|review|binding"):
        run_release(package, tmp_path)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("track", ["release", "smoke"])
def test_explicit_approval_keeps_precedence_over_promoted_binding(tmp_path, monkeypatch, release_http, track):
    package = promotion_package(tmp_path, monkeypatch, default=False)
    promote(package.candidates, review=package.review, out=package.execution, design=package.design)
    replacement_review = json.loads(package.review.read_bytes())
    replacement_review["oracle_review"] = "reviewed again explicitly"
    package.review.write_bytes(json.dumps(replacement_review).encode("utf-8"))
    expected_review = sha256(package.review)
    assert expected_review != package.review_sha256

    _, _, summary = run_release(
        package, tmp_path, track=track, release_review=package.review, release_design=package.design,
    )

    approval = summary.audit_metrics["dataset_approval"]
    assert approval["status"] == "verified"
    assert approval["review_sha256"] == expected_review
    assert release_http.queries == [package.rows[0]["query"]]


def test_smoke_keeps_its_existing_unverified_diagnostic_behavior(tmp_path, monkeypatch, release_http):
    package = promotion_package(tmp_path, monkeypatch, default=True)
    promote(package.candidates, review=package.review, out=package.execution, design=package.design)

    _, _, summary = run_release(package, tmp_path, track="smoke")

    assert summary.audit_metrics["dataset_approval"] == {"status": "not_verified"}
    assert release_http.queries == [package.rows[0]["query"]]
