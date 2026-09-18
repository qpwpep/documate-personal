"""An approved run must stage the bytes it checked, even if source files change."""

import hashlib
import importlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

import pytest
import requests

from src.core.contracts.debug import DebugPayload, TokenUsage
from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.approved_uploads import ApprovedUpload
from src.eval.nemo_generate import _sha256
from src.eval.online_runner import run_online_benchmark
from src.eval.online_runner.scenario_inputs import resolve_fixture_uploads
from src.eval.release_dataset import release_artifact_hashes, sha256
from tests.eval.response_fixtures import answer_provenance, plain_response, sse_frame, sse_http_response
from tests.eval.test_release_promotion import reviewed_candidates
from tests.eval.test_promoted_release_binding import release_http


def approved_package(tmp_path, *, setup_turns=()):
    candidates, review, design, rows = reviewed_candidates(tmp_path)
    execution = tmp_path / "deployed" / "cases.jsonl"
    upload = execution.parent / "uploads" / "settings.py"
    upload.parent.mkdir(parents=True)
    upload.write_bytes(b"LIMIT = 17\n# reviewed context\n")
    specs_path = design / "retrieval_specs.jsonl"
    specs = [json.loads(line) for line in specs_path.read_text(encoding="utf-8").splitlines()]
    specs[0]["upload_fixtures"] = [upload.name]
    specs[0]["setup_turns"] = list(setup_turns)
    specs[0]["oracle"]["evidence"].append({"source": upload.name, "locator": "line 1", "excerpt": "LIMIT = 17"})
    rows[0]["upload_fixtures"] = [upload.name]
    rows[0]["setup_turns"] = list(setup_turns)
    rows[0]["oracle"] = specs[0]["oracle"]
    rows[0]["provenance"]["nemo_generation"]["source_spec_sha256"] = _sha256(specs[0])
    specs_path.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in specs).encode("utf-8"))
    execution.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8"))
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["candidate_sha256"] = sha256(execution)
    payload["artifact_hashes"] = release_artifact_hashes(
        [BenchmarkCase.model_validate(row) for row in rows], design=design, fixtures_path=execution,
    )
    review.write_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return execution, review, design, rows, upload


def test_verified_upload_buffer_survives_source_removal(tmp_path):
    case = BenchmarkCase(case_id="snapshot", category="rag_only", query="Explain", upload_fixtures=["settings.py"])
    content = b"LIMIT = 17\n"
    uploads = resolve_fixture_uploads(tmp_path / "cases.jsonl", case,
                                     verified_uploads={"settings.py": ApprovedUpload("settings.py", content)})
    assert uploads[0].size == len(content)
    assert uploads[0].getbuffer() == content
    assert uploads[0].fingerprint == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("changed", ["candidate", "attachment"])
def test_approved_run_rejects_changes_before_any_http_request(tmp_path, monkeypatch, changed):
    execution, review, design, _, upload = approved_package(tmp_path)
    target = execution if changed == "candidate" else upload
    target.write_bytes(target.read_bytes() + b"\n")

    def network_forbidden(*args, **kwargs):
        pytest.fail("Approval failure must precede every HTTP request")

    monkeypatch.setattr(requests.sessions.Session, "request", network_forbidden)
    with pytest.raises(ValueError, match="review|approved"):
        run_online_benchmark(
            fixtures_path=execution, release_review=review, release_design=design,
            endpoint="http://approved-peer", config=BenchmarkConfig(judge_enabled=False),
            config_path=tmp_path / "config.toml", output_root=tmp_path / "runs", track="smoke", limit=1,
        )
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("source_change", ["delete", "overwrite", "rename", "extension", "outside", "parent"])
def test_approved_run_uses_checked_case_and_upload_bytes_through_staging(tmp_path, monkeypatch, source_change):
    execution, review, design, rows, upload = approved_package(tmp_path)
    original_upload = upload.read_bytes()
    expected_candidate = sha256(execution)
    expected_review = sha256(review)
    manifests, staged_bytes, staged_names, queries = {}, [], [], []
    monkeypatch.setattr("src.infra.runtime_paths.get_project_root_path", lambda: tmp_path)

    def response_json(payload):
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(payload).encode()
        response._content_consumed = True
        return response

    def request(method, url, *, json=None, **kwargs):
        session = url.split("/sessions/", 1)[1].split("/", 1)[0]
        manifest = manifests.setdefault(session, {"epoch": uuid4().hex, "revision": 0, "files": []})
        if method.upper() == "GET":
            # External files change after approval loading, before actual staging.
            execution.write_bytes(b"not a dataset anymore\n")
            if source_change == "overwrite":
                upload.write_bytes(b"LIMIT = 999\n")
            elif source_change == "parent":
                upload.parent.rename(upload.parent.with_name("previous-uploads"))
            else:
                upload.unlink()
                if source_change != "delete":
                    target = (tmp_path / "outside.py" if source_change == "outside" else
                              upload.with_name("renamed.py" if source_change == "rename" else "other.txt"))
                    target.write_bytes(b"LIMIT = 999\n")
                    try:
                        upload.symlink_to(target)
                    except (NotImplementedError, OSError) as exc:
                        pytest.skip(f"File symlinks are unavailable: {exc}")
            return response_json(manifest)
        assert method.upper() == "POST"
        if json.get("clear"):
            manifest["files"] = []
        else:
            for addition in json["add"]:
                content = Path(addition["path"]).read_bytes()
                staged_bytes.append(content)
                staged_names.append(addition["name"])
                manifest["files"].append({
                    "file_id": uuid4().hex, "name": addition["name"], "size_bytes": len(content),
                    "content_hash": "sha256:" + hashlib.sha256(content).hexdigest(),
                    "source_uri": f"upload://{session}/{addition['name']}",
                })
        manifest["revision"] += 1
        return response_json({"manifest": manifest, "changed": True})

    def post(url, *, json, **kwargs):
        queries.append(json["query"])
        body = plain_response("검증한 입력으로 실행했습니다.")
        debug = DebugPayload(
            token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            models_used=["http-boundary-fixture"], model_usage_status="llm_used",
        ).model_dump(mode="json")
        debug["answer_provenance"] = answer_provenance(body)
        return sse_http_response(200, {"response": body, "debug": debug, "trace": "snapshot-test",
                                      "upload_manifest": manifests[json["session_id"]]},
                                 headers={"x-request-id": uuid4().hex})

    monkeypatch.setattr(requests, "request", request)
    monkeypatch.setattr(requests, "post", post)
    run_dir, results, summary = run_online_benchmark(
        fixtures_path=execution, release_review=review, release_design=design,
        endpoint="http://approved-peer", config=BenchmarkConfig(judge_enabled=False),
        config_path=tmp_path / "config.toml", output_root=tmp_path / "runs", track="smoke", limit=1,
    )
    assert queries == [rows[0]["query"]]
    assert staged_bytes == [original_upload]
    assert staged_names == [upload.name]
    assert results[0].runtime_errors == results[0].response_errors == []
    assert results[0].attachment_fingerprints == {upload.name: hashlib.sha256(original_upload).hexdigest()}
    approval = summary.audit_metrics["dataset_approval"]
    assert approval["status"] == "verified"
    assert approval["candidate_sha256"] == expected_candidate
    assert approval["review_sha256"] == expected_review
    saved_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["audit_metrics"]["dataset_approval"] == approval


@pytest.mark.parametrize("mismatch", ["name", "case", "size", "hash", "missing", "extra", "duplicate"])
def test_approved_run_rejects_manifest_mismatch_before_any_question(
    tmp_path, monkeypatch, release_http, mismatch,
):
    execution, review, design, _, _ = approved_package(tmp_path, setup_turns=["Prepare the attached code."])
    request = requests.request

    def mismatched_upload_response(method, url, **kwargs):
        response = request(method, url, **kwargs)
        if method.upper() == "POST" and kwargs.get("json", {}).get("add"):
            payload = response.json()
            files = payload["manifest"]["files"]
            if mismatch == "name":
                files[0]["name"] = "renamed.py"
            elif mismatch == "case":
                files[0]["name"] = files[0]["name"].upper()
            elif mismatch == "size":
                files[0]["size_bytes"] += 1
            elif mismatch == "hash":
                files[0]["content_hash"] = "sha256:" + "0" * 64
            elif mismatch == "missing":
                files.clear()
            else:
                extra = dict(files[0], file_id=uuid4().hex)
                if mismatch == "extra":
                    extra["name"] = "unapproved.py"
                files.append(extra)
            response._content = json.dumps(payload).encode("utf-8")
        return response

    monkeypatch.setattr(requests, "request", mismatched_upload_response)
    _, results, _ = run_online_benchmark(
        fixtures_path=execution, release_review=review, release_design=design,
        endpoint="http://approved-peer", config=BenchmarkConfig(judge_enabled=False),
        config_path=tmp_path / "config.toml", output_root=tmp_path / "runs", track="smoke", limit=1,
    )
    assert release_http.queries == []
    assert results[0].scenario_turns == []
    assert any("manifest" in error.lower() for error in results[0].runtime_errors)


def test_approved_run_stops_followup_when_preparation_returns_different_attachments(
    tmp_path, monkeypatch, release_http,
):
    setup = "Prepare the attached code."
    execution, review, design, _, _ = approved_package(tmp_path, setup_turns=[setup])
    post = requests.post

    def changed_manifest_after_setup(url, **kwargs):
        response = post(url, **kwargs)
        frames = []
        for frame in response.raw.stream(8192):
            if frame.startswith(b"event: final_response\n"):
                payload = json.loads(frame.decode("utf-8").split("data: ", 1)[1])
                payload["upload_manifest"]["files"] = []
                frame = sse_frame("final_response", payload)
            frames.append(frame)
        return sse_http_response(200, chunks=frames, headers=dict(response.headers))

    monkeypatch.setattr(requests, "post", changed_manifest_after_setup)
    _, results, _ = run_online_benchmark(
        fixtures_path=execution, release_review=review, release_design=design,
        endpoint="http://approved-peer", config=BenchmarkConfig(judge_enabled=False),
        config_path=tmp_path / "config.toml", output_root=tmp_path / "runs", track="smoke", limit=1,
    )
    assert release_http.queries == [setup]
    assert [turn.query for turn in results[0].scenario_turns] == [setup]
    assert any("manifest" in error.lower() for error in results[0].runtime_errors)


@pytest.mark.parametrize("explicit_review", [False, True], ids=["default-approval", "default-design"])
def test_default_release_approval_is_independent_of_working_directory(tmp_path, monkeypatch, explicit_review):
    import src.eval.release_dataset as release_dataset

    execution, review, design, _, upload = approved_package(tmp_path)
    project_root = tmp_path / "project"
    benchmark_root = project_root / "data" / "benchmarks"
    deployed = benchmark_root / "fixtures" / "cases.generated.jsonl"
    (deployed.parent / "uploads").mkdir(parents=True)
    deployed.write_bytes(execution.read_bytes() + b"\n")  # Valid JSONL, unapproved bytes.
    (deployed.parent / "uploads" / upload.name).write_bytes(upload.read_bytes())
    shutil.copytree(design, benchmark_root / "design")
    deployed_review = benchmark_root / "design" / "release_review.json"
    deployed_review.write_bytes(review.read_bytes())
    outside = tmp_path / "outside-project"
    outside.mkdir()

    def network_forbidden(*args, **kwargs):
        pytest.fail("Default release approval must reject changed bytes before HTTP")

    # Re-import as if this temporary project were the checkout containing the CLI.
    # Restore module constants after the filesystem-boundary override is removed.
    try:
        with monkeypatch.context() as isolated:
            isolated.setattr("src.infra.runtime_paths.get_project_root_path", lambda: project_root)
            importlib.reload(release_dataset)
            isolated.chdir(outside)
            isolated.setenv("OPENAI_API_KEY", "fake-boundary-key")
            isolated.setattr(requests, "request", network_forbidden)
            isolated.setattr(requests, "post", network_forbidden)
            with pytest.raises(ValueError, match="exact candidate bytes"):
                run_online_benchmark(
                    fixtures_path=deployed, release_review=deployed_review if explicit_review else None,
                    endpoint="http://must-not-call", config=BenchmarkConfig(),
                    config_path=tmp_path / "config.toml", output_root=tmp_path / "runs",
                    track="release", limit=1,
                )
    finally:
        importlib.reload(release_dataset)
