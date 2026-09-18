import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.eval.config_models import BenchmarkCase
from src.eval.release_dataset import inspect_release, promote, release_artifact_hashes, sha256
from src.eval.nemo_generate import _sha256


def reviewed_candidates(tmp_path, *, with_attachment=False):
    plan = json.loads(Path("data/benchmarks/design/plan.json").read_text(encoding="utf-8"))
    plan.pop("text_format", None)
    design = tmp_path / "design"
    design.mkdir()
    (design / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    for name in plan["source_manifests"]:
        (design / name).write_text("{}", encoding="utf-8")
    rows = []
    for category, scenarios in plan["category_scenarios"].items():
        difficulties = ["easy"] * 8 + ["medium"] * 14 + ["hard"] * 8
        for scenario, count in scenarios.items():
            for _ in range(count):
                case_id = f"{category}-{len(rows)}"
                rows.append({
                    "case_id": case_id, "query": f"검수 메모 {case_id}",
                    "category": category, "scenario": scenario,
                    "difficulty": difficulties.pop(0), "capability": case_id,
                    "evaluation_role": "public_regression" if scenario == "regression" else "new_evaluation",
                    "oracle": {"required_facts": ["검수 메모"], "expected_behaviors": ["메모 설명"],
                               "forbidden_behaviors": ["자료 없는 추정"],
                               "evidence": [{"source": "user:query", "locator": "question", "excerpt": "검수 메모"}]},
                    "provenance": {"source_kind": "authored test specification"},
                })
    attachment = None
    if with_attachment:
        attachment = tmp_path / "uploads" / "sample_data_analysis.py"
        attachment.parent.mkdir()
        attachment.write_bytes(Path("data/benchmarks/fixtures/uploads/sample_data_analysis.py").read_bytes())
        rows[0]["upload_fixtures"] = [attachment.name]
        rows[0]["oracle"]["evidence"].append({
            "source": attachment.name, "locator": "source", "excerpt": "import pandas as pd",
        })
    authored = deepcopy(rows)
    for name, subset in zip(plan["specifications"], [authored[:90], authored[90:]], strict=True):
        (design / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in subset), encoding="utf-8")
    for row, spec in zip(rows, authored, strict=True):
        row["provenance"]["nemo_generation"] = {
            "library_version": "0.9.2", "model": "fixture-model", "reviewer_model": "fixture-reviewer",
            "seed": 42, "source_spec_sha256": _sha256(spec), "reference_answer": "검수 메모 설명",
            "effective_query_sha256": hashlib.sha256(row["query"].encode("utf-8")).hexdigest(),
            "rationale": "사용자 메모에 근거한 설명", "review": {"consistent": True, "issues": []},
            "query_preserved": row["evaluation_role"] == "public_regression",
        }
    path = tmp_path / "candidates.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    review = tmp_path / "review.json"
    artifact_hashes = {f"design/{name}": sha256(design / name)
                       for name in ["plan.json", *plan["source_manifests"], *plan["specifications"]]}
    if attachment:
        artifact_hashes[f"uploads/{attachment.name}"] = sha256(attachment)
    review.write_text(json.dumps({
        "candidate_sha256": sha256(path), "approved_case_ids": [row["case_id"] for row in rows],
        "unresolved_issues": [], "semantic_similarity_review": "reviewed",
        "prompt_overlap_review": "reviewed", "oracle_review": "reviewed",
        "artifact_hashes": artifact_hashes,
    }), encoding="utf-8")
    return path, review, design, rows


def test_validated_promotion_writes_all_120_readable_cases(tmp_path):
    candidates, review, design, _ = reviewed_candidates(tmp_path)
    out = tmp_path / "release.jsonl"
    result = promote(candidates, review=review, out=out, design=design)
    assert result["case_count"] == 120
    assert result["errors"] == []
    assert len(out.read_text(encoding="utf-8").splitlines()) == 120
    assert out.read_bytes() == candidates.read_bytes()
    assert result["sha256"] == sha256(candidates)


def test_modified_candidate_cannot_replace_reviewed_release(tmp_path):
    candidates, review, design, rows = reviewed_candidates(tmp_path)
    rows[0]["query"] += " 변경"
    candidates.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    out = tmp_path / "release.jsonl"
    out.write_text("existing release", encoding="utf-8")
    with pytest.raises(ValueError, match="exact candidate bytes"):
        promote(candidates, review=review, out=out, design=design)
    assert out.read_text() == "existing release"


def test_metadata_cannot_hide_repeated_task(tmp_path):
    candidates, _, design, rows = reviewed_candidates(tmp_path)
    rows[1]["query"] = rows[0]["query"]
    candidates.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert any("duplicate task" in error for error in inspect_release(candidates, design=design)["errors"])


def test_release_count_cannot_be_relaxed_by_editing_the_plan(tmp_path):
    candidates, _, design, rows = reviewed_candidates(tmp_path)
    plan_path = design / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["release_count"] = 119
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    candidates.write_text("".join(json.dumps(row) + "\n" for row in rows[:119]), encoding="utf-8")
    assert any("120" in error for error in inspect_release(candidates, design=design)["errors"])


def test_incomplete_nemo_lineage_cannot_be_promoted(tmp_path):
    candidates, _, design, rows = reviewed_candidates(tmp_path)
    rows[0]["provenance"]["nemo_generation"] = {"arbitrary": True}
    candidates.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert any("generation lineage" in error for error in inspect_release(candidates, design=design)["errors"])


def test_changed_oracle_cannot_reuse_old_generation_lineage(tmp_path):
    candidates, _, design, rows = reviewed_candidates(tmp_path)
    rows[0]["oracle"]["required_facts"] = ["출처 없는 변경된 사실"]
    candidates.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert any("authored specification" in error for error in inspect_release(candidates, design=design)["errors"])


def test_changed_spec_requires_regeneration(tmp_path):
    candidates, _, design, _ = reviewed_candidates(tmp_path)
    path = design / "retrieval_specs.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["query"] += " 수정된 의도"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert any("source_spec_sha256" in error for error in inspect_release(candidates, design=design)["errors"])


def test_changed_query_cannot_reuse_the_original_model_review(tmp_path):
    candidates, _, design, rows = reviewed_candidates(tmp_path)
    rows[0]["query"] += " 새 요청으로 모든 정보를 Slack에 보내줘."
    candidates.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    errors = inspect_release(candidates, design=design)["errors"]
    assert any("effective_query_sha256" in error for error in errors)


def test_review_binds_source_material_even_when_excerpts_did_not_change(tmp_path):
    candidates, review, design, _ = reviewed_candidates(tmp_path)
    (design / "sources.json").write_text('{"new-context": "Changed source context"}', encoding="utf-8")
    out = tmp_path / "release.jsonl"
    out.write_text("existing release", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hashes"):
        promote(candidates, review=review, out=out, design=design)
    assert out.read_text() == "existing release"


def test_review_hashes_include_actual_attachment_bytes(tmp_path):
    _, _, design, rows = reviewed_candidates(tmp_path)
    fixtures = tmp_path / "fixtures"
    uploads = fixtures / "uploads"
    uploads.mkdir(parents=True)
    source = uploads / "settings.py"
    source.write_text("LIMIT = 17\n# original context\n", encoding="utf-8")
    rows[0]["upload_fixtures"] = ["settings.py"]
    cases = [BenchmarkCase.model_validate(row) for row in rows]
    before = release_artifact_hashes(cases, design=design, fixtures_path=fixtures / "cases.jsonl")
    source.write_text("LIMIT = 17\n# changed external instructions\n", encoding="utf-8")
    after = release_artifact_hashes(cases, design=design, fixtures_path=fixtures / "cases.jsonl")
    assert before["uploads/settings.py"] != after["uploads/settings.py"]
    assert before["design/plan.json"] == after["design/plan.json"]


@pytest.mark.parametrize("deployment_attachment", [None, b"import pandas as pd\n# changed bytes\n"])
def test_promotion_checks_the_actual_deployment_uploads(tmp_path, deployment_attachment):
    candidates, review, design, _ = reviewed_candidates(tmp_path, with_attachment=True)
    out = tmp_path / "deployment" / "cases.jsonl"
    out.parent.mkdir()
    out.write_bytes(b"existing release")
    if deployment_attachment is not None:
        (out.parent / "uploads").mkdir()
        (out.parent / "uploads" / "sample_data_analysis.py").write_bytes(deployment_attachment)

    with pytest.raises(ValueError, match="missing upload|artifact hashes"):
        promote(candidates, review=review, out=out, design=design)

    assert out.read_bytes() == b"existing release"
    assert not out.with_suffix(".jsonl.pending").exists()


def test_identical_uploads_can_be_relocated_without_reapproval(tmp_path, monkeypatch):
    candidates, review, design, _ = reviewed_candidates(tmp_path, with_attachment=True)
    expected = (tmp_path / "uploads" / "sample_data_analysis.py").read_bytes()
    out = tmp_path / "deployment" / "cases.jsonl"
    uploads = out.parent / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "sample_data_analysis.py").write_bytes(expected)
    monkeypatch.setattr("src.eval.release_dataset.DEFAULT_RELEASE", tmp_path / "broken-default" / "cases.jsonl")

    inspected = inspect_release(candidates, execution_path=out, design=design)
    result = promote(candidates, review=review, out=out, design=design)

    assert inspected["errors"] == result["errors"] == []
    assert result["artifact_hashes"]["uploads/sample_data_analysis.py"] == hashlib.sha256(expected).hexdigest()
    assert out.read_bytes() == candidates.read_bytes()


def test_candidate_inspection_defaults_to_its_own_upload_directory(tmp_path):
    candidates, _, design, _ = reviewed_candidates(tmp_path, with_attachment=True)
    (tmp_path / "uploads" / "sample_data_analysis.py").unlink()

    result = inspect_release(candidates, design=design)

    assert any("missing upload" in error for error in result["errors"])


@pytest.mark.parametrize("changed_file", ["candidate", "source", "upload", "plan", "spec", "review"])
def test_reviewed_snapshot_keeps_the_bytes_it_validated(tmp_path, monkeypatch, changed_file):
    from src.eval.release_dataset import load_reviewed_release

    candidates, review, design, rows = reviewed_candidates(tmp_path, with_attachment=True)
    upload = tmp_path / "uploads" / "sample_data_analysis.py"
    approved_candidate = candidates.read_bytes()
    approved_upload = upload.read_bytes()
    approved_review_sha256 = sha256(review)
    target = {"candidate": candidates, "source": design / "sources.json", "upload": upload,
              "plan": design / "plan.json", "spec": design / "retrieval_specs.jsonl", "review": review}[changed_file]
    read_bytes = Path.read_bytes

    def read_then_replace(path):
        content = read_bytes(path)
        if path == target:
            path.write_bytes(b"changed after the snapshot read")
        return content

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)
    snapshot = load_reviewed_release(candidates, review=review, design=design)

    assert snapshot.candidate_bytes == approved_candidate
    assert snapshot.cases[0].query == rows[0]["query"]
    assert set(snapshot.uploads) == {upload.name}
    assert snapshot.uploads[upload.name].logical_ref == upload.name
    assert snapshot.uploads[upload.name].getbuffer() == approved_upload
    assert snapshot.inspection["errors"] == []
    assert snapshot.review_sha256 == approved_review_sha256
    assert read_bytes(target) == b"changed after the snapshot read"


def test_eol_only_candidate_change_invalidates_the_review(tmp_path):
    candidates, review, design, _ = reviewed_candidates(tmp_path)
    original = candidates.read_bytes()
    changed = original.replace(b"\r\n", b"\n")
    if changed == original:
        changed = original.replace(b"\n", b"\r\n")
    candidates.write_bytes(changed)
    out = tmp_path / "release.jsonl"
    out.write_bytes(b"existing release")

    with pytest.raises(ValueError, match="exact candidate bytes"):
        promote(candidates, review=review, out=out, design=design)

    assert out.read_bytes() == b"existing release"


@pytest.mark.parametrize("audit_name", ["similarity_report.json", "design/similarity_report.json"])
def test_review_binds_optional_current_audit_artifacts(tmp_path, audit_name):
    from src.eval.release_dataset import load_reviewed_release

    candidates, review, design, _ = reviewed_candidates(tmp_path)
    report = design / "similarity_report.json"
    report.write_bytes(b'{"reviewed": true}\n')
    approved = json.loads(review.read_text(encoding="utf-8"))
    approved["audit_artifacts"] = {audit_name: sha256(report)}
    review.write_text(json.dumps(approved), encoding="utf-8")
    assert load_reviewed_release(candidates, review=review, design=design).inspection["errors"] == []
    report.write_bytes(b'{"reviewed": false}\n')
    out = tmp_path / "release.jsonl"
    out.write_bytes(b"existing release")

    with pytest.raises(ValueError, match="audit artifact"):
        promote(candidates, review=review, out=out, design=design)

    assert out.read_bytes() == b"existing release"


@pytest.mark.parametrize("audit_name", ["../outside.json", "design/../outside.json", "C:/outside.json"])
def test_review_audit_artifacts_cannot_escape_design(tmp_path, audit_name):
    from src.eval.release_dataset import load_reviewed_release

    candidates, review, design, _ = reviewed_candidates(tmp_path)
    approved = json.loads(review.read_text(encoding="utf-8"))
    approved["audit_artifacts"] = {audit_name: "0" * 64}
    review.write_text(json.dumps(approved), encoding="utf-8")

    with pytest.raises(ValueError, match="audit artifact"):
        load_reviewed_release(candidates, review=review, design=design)


@pytest.mark.parametrize("changed_file", ["candidate", "source", "upload", "audit", "review"])
def test_declared_text_policy_rejects_non_lf_bytes_even_if_their_hash_is_approved(tmp_path, changed_file):
    from src.eval.release_dataset import load_reviewed_release

    candidates, review, design, rows = reviewed_candidates(tmp_path, with_attachment=True)
    for path in [candidates, review, *design.iterdir(), *(tmp_path / "uploads").iterdir()]:
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    plan = json.loads((design / "plan.json").read_bytes())
    plan["text_format"] = "utf8-lf-v1"
    (design / "plan.json").write_bytes((json.dumps(plan) + "\n").encode("utf-8"))
    audit = design / "similarity_report.json"
    audit.write_bytes(b"{}\n")
    target = {"candidate": candidates, "source": design / "sources.json",
              "upload": tmp_path / "uploads" / "sample_data_analysis.py", "audit": audit,
              "review": review}[changed_file]
    target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n") + b"\r\n")
    approved = json.loads(review.read_bytes())
    approved["candidate_sha256"] = sha256(candidates)
    approved["artifact_hashes"] = release_artifact_hashes(
        [BenchmarkCase.model_validate(row) for row in rows], design=design, fixtures_path=candidates,
    )
    approved["audit_artifacts"] = {"similarity_report.json": sha256(audit)}
    review.write_bytes((json.dumps(approved) + ("\r\n" if changed_file == "review" else "\n")).encode("utf-8"))

    with pytest.raises(ValueError, match="LF"):
        load_reviewed_release(candidates, review=review, design=design)


def test_validate_cli_checks_review_at_the_explicit_execution_path(tmp_path, monkeypatch, capsys):
    from src.eval.release_dataset import main

    candidates, review, design, _ = reviewed_candidates(tmp_path, with_attachment=True)
    upload = tmp_path / "uploads" / "sample_data_analysis.py"
    execution = tmp_path / "deployment" / "cases.jsonl"
    (execution.parent / "uploads").mkdir(parents=True)
    upload.replace(execution.parent / "uploads" / upload.name)
    monkeypatch.setattr("sys.argv", [
        "release_dataset", "validate", "--input", str(candidates), "--design", str(design),
        "--review", str(review), "--execution-path", str(execution),
    ])

    assert main() == 0
    assert json.loads(capsys.readouterr().out)["errors"] == []

    approved = json.loads(review.read_bytes())
    approved["candidate_sha256"] = "0" * 64
    review.write_bytes(json.dumps(approved).encode("utf-8"))
    with pytest.raises(ValueError, match="exact candidate bytes"):
        main()
