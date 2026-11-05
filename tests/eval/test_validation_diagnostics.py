from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.decisions import raw_results_fingerprint
from src.eval.validation_diagnostics import diagnose_run
from src.core.answer_schema import AnswerDocument
from src.core.answer_schema.models import document_hash
from src.core.documents import DocumentElement, build_snapshot
from src.core.decisions import prepared_packet_fingerprint
from src.core.evidence import EvidenceRef, build_evidence
from src.eval.diagnostic_review import DiagnosticAssessment, DiagnosticReviewOutcome


def _write_run(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "raw_results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
    )
    (run / "summary.json").write_text(json.dumps({
        "run_id": "run", "results_fingerprint": raw_results_fingerprint(rows),
    }), encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    return run


def _old_result() -> dict[str, object]:
    return {
        "run_id": "run", "case_id": "case_1", "category": "hybrid", "query": "Explain the code.",
        "debug": {"schema_version": 14, "validation_events": ["arbitrary failure text"]},
        "product_pass": False, "release_pass": False, "judge_pass": None,
        "turn_result": {"status": "failed"}, "scenario_turns": [],
    }


def test_historical_diagnostics_stay_unobserved_and_source_files_unchanged(tmp_path):
    run = _write_run(tmp_path, [_old_result()])
    before = {path.name: path.read_bytes() for path in run.iterdir()}

    report = diagnose_run(run, tmp_path / "diagnosis.json")

    assert report.scope == "diagnostic_only"
    assert report.cause_counts == {"diagnostics_unobserved": 1}
    assert report.cases[0].original_release_pass is False
    assert report.reviews == []
    assert {path.name: path.read_bytes() for path in run.iterdir()} == before
    saved = json.loads((tmp_path / "diagnosis.json").read_text(encoding="utf-8"))
    assert saved["source_results_fingerprint"] == raw_results_fingerprint([_old_result()])


def test_diagnosis_rejects_changed_raw_results_before_writing(tmp_path):
    run = _write_run(tmp_path, [_old_result()])
    with (run / "raw_results.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_old_result()) + "\n")

    with pytest.raises(ValueError, match="fingerprint"):
        diagnose_run(run, tmp_path / "diagnosis.json")
    assert not (tmp_path / "diagnosis.json").exists()


def test_diagnosis_rejects_source_directory_and_existing_destination(tmp_path):
    run = _write_run(tmp_path, [_old_result()])
    with pytest.raises(ValueError, match="outside"):
        diagnose_run(run, run / "diagnosis.json")
    output = tmp_path / "diagnosis.json"
    output.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        diagnose_run(run, output)
    assert output.read_text(encoding="utf-8") == "original"


def test_diagnosis_rejects_invalid_sample_size(tmp_path):
    run = _write_run(tmp_path, [_old_result()])
    with pytest.raises(ValueError, match="sample_per_cause"):
        diagnose_run(run, tmp_path / "diagnosis.json", sample_per_cause=0)


def _recorded_result(case_id="case_1"):
    text = "ignored prefix\ncount = 3\nignored suffix"
    source = build_evidence(
        snapshot=build_snapshot(
            source_uri="upload:///session/file", title="counts.py", media_type="text/plain",
            source_type="upload", content=text, parser="fixture", parser_version="1",
        ),
        element=DocumentElement(element_id="source", kind="code", text=text), start=15, end=24,
    )
    document = AnswerDocument.model_validate({"blocks": [{"type": "paragraph", "content": [
        {"text": "There are three items.", "basis": "source", "refs": [source.id]},
    ]}]})
    task = {"query": "Find count", "k": 1, "route": "upload", "requirement_id": "count"}
    common = {
        "attempt": 1, "model": "runtime-model", "threshold": 0.8,
        "user_input": "Explain the code.", "request_contract": None, "status": "completed",
    }
    pre = {
        **common, "sequence": 1, "stage": "pre_synthesis_validation", "revision": "preparation",
        "tasks": [task, {**task, "requirement_id": "missing"}],
        "candidates": [{"requirement_id": "count", "evidence_id": source.id, "role": "candidate"}],
        "packets": [{"mode": "normal", "evidence_ids": [], "requirement_map": {},
                     "fingerprint": prepared_packet_fingerprint([], {})}],
        "selected_packet_mode": "normal",
        "predicates": [{"name": "insufficient", "criterion": "target_analysis_supported",
                        "requirement_id": "count", "status": "evaluated", "probability": 0.4},
                       {"name": "refused", "criterion": "target_identified", "status": "refused"}],
    }
    post = {
        **common, "sequence": 2, "stage": "post_synthesis_validation", "revision": "generated", "tasks": [task],
        "body": document.model_dump(mode="json"), "body_hash": document_hash(document),
        "packets": [{"mode": "used", "evidence_ids": [source.id],
                     "requirement_map": {source.id: ["count"]},
                     "fingerprint": prepared_packet_fingerprint([source], {source.id: ["count"]})}],
        "terminal_failed_requirement_ids": ["count"], "local_failures": ["missing_required:comparison"],
        "predicates": [{"name": "request_fulfilled", "criterion": "request_fulfilled",
                        "status": "evaluated", "probability": 0.3}],
    }
    failed = {
        **common, "sequence": 3, "stage": "pre_synthesis_validation", "revision": "preparation",
        "tasks": [task], "status": "failed", "failure_type": "timeout",
    }
    return {**_old_result(), "case_id": case_id, "debug": {"validation_diagnostics": {
        "schema_version": 1, "evidence": {source.id: source.model_dump(mode="json")},
        "observations": [pre, post, failed],
    }}}


def test_diagnosis_retains_overlapping_signals_exact_body_and_packet(tmp_path):
    row = _recorded_result()
    run = _write_run(tmp_path, [row])
    report = diagnose_run(run, tmp_path / "diagnosis.json", sample_per_cause=10)

    assert report.cause_counts == {cause: 1 for cause in (
        "candidate_gap", "selection_gap", "semantic_rejection", "predicate_refusal",
        "body_condition_failure", "local_validation_failure", "unassessed_requirements", "technical_failure",
    )}
    body_review = next(review for review in report.reviews if review.observation.body is not None)
    assert body_review.scope == "unapproved_validation_sample"
    assert body_review.observation.body.blocks[0].content[0].text == "There are three items."
    evidence = next(iter(body_review.evidence.values()))
    assert evidence.selection.start == 15
    assert evidence.selection.end == 24
    assert evidence.excerpt == "count = 3"
    assert body_review.observation.packets[0].evidence_ids == [evidence.id]
    assert body_review.original_release_pass is False


def test_samples_are_stable_when_result_order_changes(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    rows = [_recorded_result(f"case_{index}") for index in range(5)]
    first_run = _write_run(first, rows)
    second_run = _write_run(second, list(reversed(rows)))

    one = diagnose_run(first_run, first / "out.json", sample_per_cause=1)
    two = diagnose_run(second_run, second / "out.json", sample_per_cause=1)

    assert [review.sample_id for review in one.reviews] == [review.sample_id for review in two.reviews]
    assert one.cause_counts == two.cause_counts


def test_diagnostic_judge_cannot_change_original_verdict_or_candidate(tmp_path):
    run = _write_run(tmp_path, [_recorded_result()])
    before = (run / "raw_results.jsonl").read_bytes()
    reviewed = []

    def review_candidate(bundle):
        assert bundle.scope == "unapproved_validation_sample"
        assert bundle.observation.body is not None
        reviewed.append(bundle.sample_id)
        bundle.observation.body.blocks[0].content[0].text = "changed by collaborator"
        return DiagnosticReviewOutcome(status="succeeded", model_name="review-model", assessment=DiagnosticAssessment(
            body_support="supported", body_fulfillment="fulfilled", oracle_scope="unavailable", reason="Supported candidate",
        ))

    report = diagnose_run(run, tmp_path / "out.json", sample_per_cause=10, judge=review_candidate)

    assert len(reviewed) == 1
    assert report.diagnostic_judgments[0].outcome.assessment.body_support == "supported"
    assert report.cases[0].original_release_pass is False
    assert next(r for r in report.reviews if r.observation.body).observation.body.blocks[0].content[0].text == "There are three items."
    assert (run / "raw_results.jsonl").read_bytes() == before


@pytest.mark.parametrize("changed", [False, True])
def test_oracle_requires_saved_approval_hash(tmp_path, changed):
    run = _write_run(tmp_path, [_recorded_result()])
    fixture = tmp_path / "fixtures.jsonl"
    fixture.write_text(json.dumps({
        "case_id": "case_1", "category": "hybrid", "query": "Explain the code.",
        "oracle": {"required_facts": ["count is three"]},
    }) + "\n", encoding="utf-8")
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    (run / "manifest.json").write_text(json.dumps({
        "status": "completed", "fixtures_path": str(fixture),
        "dataset_approval": {"status": "verified", "candidate_sha256": digest},
    }), encoding="utf-8")
    if changed:
        fixture.write_text(fixture.read_text(encoding="utf-8").replace("three", "four"), encoding="utf-8")

    report = diagnose_run(run, tmp_path / "out.json")

    assert report.oracle_status == ("hash_mismatch" if changed else "verified")
    for review in report.reviews:
        if changed:
            assert review.case is None
        else:
            assert review.case.oracle.required_facts == ["count is three"]


def test_diagnose_cli_exports_without_network_client(tmp_path, capsys):
    from src.eval.main import build_parser

    run = _write_run(tmp_path, [_old_result()])
    output = tmp_path / "cli.json"
    args = build_parser().parse_args([
        "diagnose", "--run", str(run), "--out", str(output), "--sample-per-cause", "1",
    ])

    assert args.func(args) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["scope"] == "diagnostic_only"
    assert str(output) in capsys.readouterr().out


def test_deterministic_skip_is_not_an_unassessed_failure(tmp_path):
    row = _recorded_result()
    diagnostics = row["debug"]["validation_diagnostics"]
    diagnostics["observations"] = [{
        "sequence": 1, "stage": "pre_synthesis_validation", "attempt": 1, "revision": "preparation",
        "model": "deterministic-exact-copy", "threshold": 0.8, "status": "skipped",
        "user_input": row["query"], "request_contract": None, "tasks": [], "skip_reason": "exact_copy",
    }]
    run = _write_run(tmp_path, [row])

    report = diagnose_run(run, tmp_path / "out.json")

    assert report.cause_counts == {}
    assert report.reviews == []


def test_selected_compact_packet_does_not_inherit_normal_selection_gap(tmp_path):
    row = _recorded_result()
    diagnostics = row["debug"]["validation_diagnostics"]
    pre = diagnostics["observations"][0]
    pre["tasks"] = pre["tasks"][:1]
    pre["selected_packet_mode"] = "compact"
    source = next(iter(diagnostics["evidence"]))
    used = diagnostics["observations"][1]["packets"][0]
    pre["packets"].append({**used, "mode": "compact"})
    pre["predicates"] = []
    diagnostics["observations"] = [pre]
    run = _write_run(tmp_path, [row])

    report = diagnose_run(run, tmp_path / "out.json")

    assert source in pre["packets"][1]["evidence_ids"]
    assert report.cause_counts == {}


def test_scenario_turns_are_counted_once_with_separate_body_inputs(tmp_path):
    row = _recorded_result()
    diagnostics = row["debug"]["validation_diagnostics"]
    pre = diagnostics["observations"][0]
    post = {**diagnostics["observations"][1], "sequence": 1}
    row["scenario_turns"] = [
        {"role": "setup", "query": "Prepare the source.", "execution_id": "setup-id", "debug": {
            "validation_diagnostics": {**diagnostics, "observations": [pre]},
        }},
        {"role": "question", "query": row["query"], "execution_id": "question-id", "debug": {
            "validation_diagnostics": {**diagnostics, "observations": [post]},
        }},
    ]
    run = _write_run(tmp_path, [row])

    report = diagnose_run(run, tmp_path / "out.json", sample_per_cause=10)

    assert len(report.reviews) == 2
    assert report.cases[0].observed_turns == 2
    assert {(review.turn_role, review.execution_id) for review in report.reviews} == {
        ("setup", "setup-id"), ("question", "question-id"),
    }
    assert "technical_failure" not in report.cause_counts


def test_malformed_new_diagnostic_evidence_is_not_treated_as_unobserved(tmp_path):
    row = _recorded_result()
    row["debug"]["validation_diagnostics"]["evidence"] = {}
    run = _write_run(tmp_path, [row])

    with pytest.raises(ValueError, match="unrecorded evidence"):
        diagnose_run(run, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_sample_preserves_candidate_context_and_resolved_semantic_tasks(tmp_path):
    row = _recorded_result()
    diagnostics = row["debug"]["validation_diagnostics"]
    source = EvidenceRef.model_validate(next(iter(diagnostics["evidence"].values())))
    context = build_evidence(snapshot=source.snapshot, element=source.element, start=0, end=14)
    diagnostics["evidence"][context.id] = context.model_dump(mode="json")
    observation = diagnostics["observations"][0]
    observation["semantic_tasks"] = [{**observation["tasks"][0], "query": "Original semantic requirement"}]
    observation["candidates"][0]["context_evidence_ids"] = [context.id]
    run = _write_run(tmp_path, [row])

    report = diagnose_run(run, tmp_path / "out.json", sample_per_cause=10)

    sample = next(review for review in report.reviews if review.observation.sequence == 1)
    assert sample.evidence[context.id].excerpt == "ignored prefix"
    assert sample.observation.semantic_tasks[0].query == "Original semantic requirement"


@pytest.mark.parametrize("status", ["failed", "completed"])
def test_pending_predicates_remain_unassessed_independently_of_observation_status(tmp_path, status):
    row = _recorded_result()
    diagnostics = row["debug"]["validation_diagnostics"]
    observation = diagnostics["observations"][0]
    observation["status"] = status
    observation["failure_type"] = "timeout" if status == "failed" else None
    observation["predicates"] = [{
        "name": "count_support", "criterion": "target_analysis_supported", "requirement_id": "count",
        "status": "pending", "probability": None,
    }]
    diagnostics["observations"] = [observation]
    run = _write_run(tmp_path, [row])

    report = diagnose_run(run, tmp_path / "out.json")

    assert report.cause_counts["unassessed_requirements"] == 1
    assert ("technical_failure" in report.cause_counts) is (status == "failed")
    assert "semantic_rejection" not in report.cause_counts


def test_context_only_evidence_does_not_establish_primary_candidate_coverage(tmp_path):
    row = _recorded_result()
    diagnostics = row["debug"]["validation_diagnostics"]
    observation = diagnostics["observations"][0]
    observation["tasks"] = observation["tasks"][:1]
    observation["candidates"][0]["role"] = "context"
    observation["predicates"] = []
    diagnostics["observations"] = [observation]
    run = _write_run(tmp_path, [row])

    report = diagnose_run(run, tmp_path / "out.json")

    assert report.cause_counts == {"candidate_gap": 1}


@pytest.mark.parametrize("missing_field", ["validation_diagnostics", "latency_breakdown"])
def test_explicit_missing_validation_observation_is_not_an_empty_success(tmp_path, missing_field):
    row = _old_result()
    row["debug"] = {
        "observability_status": "failed", "missing_required_debug_fields": [missing_field],
        "validation_diagnostics": {"schema_version": 1, "evidence": {}, "observations": []},
    }
    run = _write_run(tmp_path, [row])

    report = diagnose_run(run, tmp_path / "out.json")

    missing_validation = missing_field == "validation_diagnostics"
    assert report.cause_counts == ({"diagnostics_unobserved": 1} if missing_validation else {})
    assert report.cases[0].observed_turns == (0 if missing_validation else 1)
    assert report.cases[0].unobserved_turns == (1 if missing_validation else 0)
    assert report.reviews == []
