"""Independent candidate review uses exact evidence without changing release decisions."""
from __future__ import annotations

import hashlib
import json

import httpx
import pytest
from langchain_openai import ChatOpenAI

from src.core.request_contracts import RequestContract
from src.eval.diagnostic_review import DiagnosticReviewJudge
from src.eval.config_models import BenchmarkCase, CaseOracle
from src.eval.decisions import raw_results_fingerprint
from src.eval.main import build_parser
from src.eval.validation_diagnostics import DiagnosticReviewBundle
from src.infra.settings import AppSettings
from tests.validation_diagnostic_fixtures import rejected_answer_diagnostics


@pytest.fixture
def review_sample():
    diagnostics = rejected_answer_diagnostics()
    return DiagnosticReviewBundle(
        sample_id="case:question:0:1", case_id="case", category="docs_only",
        turn_role="question", turn_index=0, execution_id="execution",
        query=diagnostics.observations[0].user_input, causes=["semantic_rejection"],
        original_product_pass=False, original_judge_pass=None, original_release_pass=False,
        oracle_status="not_available", case=None, observation=diagnostics.observations[0],
        evidence=diagnostics.evidence,
    )


@pytest.fixture
def review_provider(monkeypatch):
    requests = []
    output = {
        "body_support": "unsupported", "body_fulfillment": "unfulfilled",
        "oracle_scope": "unavailable", "reason": "The cited service accepts reads, not writes.",
    }

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={
            "id": "review", "object": "chat.completion", "created": 1, "model": payload["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(output),
            }}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        def create_model(**kwargs):
            return ChatOpenAI(**(kwargs | {
                "base_url": "https://diagnostic-review.invalid/v1", "http_client": client,
            }))

        monkeypatch.setattr("src.eval.diagnostic_review.ChatOpenAI", create_model)
        yield requests, output


def test_review_uses_candidate_and_exact_packet_without_runtime_verdicts(review_sample, review_provider):
    requests, _ = review_provider
    original = review_sample.model_dump_json()

    result = DiagnosticReviewJudge(model_name="gpt-6.1-terra", api_key="test-key")(review_sample)

    assert result.status == "succeeded"
    assert result.assessment.body_support == "unsupported"
    assert result.input_stats["provider_input_tokens"] == 50
    assert review_sample.model_dump_json() == original
    payload = json.loads(requests[0]["messages"][1]["content"])
    assert payload["scope"] == "diagnostic_candidate_body_only"
    assert payload["candidate_body"] == review_sample.observation.body.model_dump(mode="json")
    source_id = next(iter(review_sample.evidence))
    assert payload["evidence"][source_id]["excerpt"] == "A read-only service accepts read requests."
    assert "predicates" not in payload
    assert "original_release_pass" not in payload
    assert "temperature" not in requests[0]


def test_review_keeps_invalid_judge_output_unobserved(review_sample, review_provider):
    _, output = review_provider
    output["body_support"] = "probably fine"

    result = DiagnosticReviewJudge(model_name="gpt-6.1-terra", api_key="test-key")(review_sample)

    assert result.status == "failed"
    assert result.failure_kind == "output_invalid"
    assert result.assessment is None


def test_review_budget_does_not_truncate_evidence_or_call_provider(review_sample, review_provider):
    requests, _ = review_provider

    result = DiagnosticReviewJudge(
        model_name="gpt-6.1-terra", api_key="test-key", max_input_tokens=1,
    )(review_sample)

    assert result.status == "not_run"
    assert result.failure_kind == "input_budget_exceeded"
    assert result.assessment is None
    assert requests == []


@pytest.mark.parametrize("body", [
    {"kind": "copy_answer", "source": {"ref": "previous", "response_hash": "source-answer-hash"}},
    {"kind": "transform_answer", "source": {"ref": "previous", "response_hash": "source-answer-hash"},
     "change_requirement": "required", "instruction": "Translate the previous answer into English."},
])
def test_answer_source_without_recorded_body_is_not_reviewed(body, review_sample, review_provider):
    requests, _ = review_provider
    review_sample.observation.request_contract = RequestContract.model_validate({"body": body})
    original = review_sample.model_dump_json()

    result = DiagnosticReviewJudge(model_name="gpt-6.1-terra", api_key="test-key")(review_sample)

    assert result.status == "not_run"
    assert result.failure_kind == "input_incomplete"
    assert result.input_stats["incomplete_reason"] == "source_answer_not_recorded"
    assert result.assessment is None
    assert requests == []
    assert review_sample.model_dump_json() == original


def test_bound_input_transformation_remains_reviewable(review_sample, review_provider):
    requests, _ = review_provider
    source_text = "Read requests only."
    review_sample.observation.request_contract = RequestContract.model_validate({"body": {
        "kind": "transform_input", "change_requirement": "required", "instruction": "Translate into Korean.",
        "source": {
            "turn_id": "turn-1", "text": source_text, "start": 0, "end": len(source_text),
            "content_hash": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        },
    }})
    original = review_sample.model_dump_json()

    result = DiagnosticReviewJudge(model_name="gpt-6.1-terra", api_key="test-key")(review_sample)

    assert result.status == "succeeded"
    assert len(requests) == 1
    payload = json.loads(requests[0]["messages"][1]["content"])
    assert payload["confirmed_request"]["body"]["source"]["text"] == source_text
    assert review_sample.model_dump_json() == original


def test_diagnose_cli_judge_preserves_release_with_verified_oracle(
    tmp_path, monkeypatch, review_provider,
):
    requests, judgment = review_provider
    judgment.update(body_support="supported", body_fulfillment="fulfilled", oracle_scope="aligned")
    diagnostics = rejected_answer_diagnostics()
    case = BenchmarkCase(
        case_id="case", category="docs_only", query=diagnostics.observations[0].user_input,
        oracle=CaseOracle(required_facts=["The service accepts read requests."]),
    )
    fixture = tmp_path / "cases.jsonl"
    fixture.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    row = {
        "run_id": "run", "case_id": case.case_id, "category": case.category, "query": case.query,
        "execution_id": "execution", "debug": {"validation_diagnostics": diagnostics.model_dump(mode="json")},
        "product_pass": False, "judge_pass": None, "release_pass": False, "scenario_turns": [],
    }
    (run / "raw_results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({
        "run_id": "run", "results_fingerprint": raw_results_fingerprint([row]),
    }), encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({
        "status": "completed", "fixtures_path": str(fixture),
        "dataset_approval": {
            "status": "verified", "candidate_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        },
    }), encoding="utf-8")
    original = {path: path.read_bytes() for path in run.iterdir()}
    settings_file = tmp_path / ".env"
    settings_file.write_text("OPENAI_API_KEY=test-key\nJUDGE_MODEL=gpt-6.1-terra\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    settings = AppSettings(_env_file=settings_file)
    monkeypatch.setattr("src.eval.main.get_settings", lambda: settings)
    monkeypatch.setattr("src.infra.settings.get_env_file_path", lambda: settings_file)
    config = tmp_path / "config.toml"
    config.write_text("[runtime]\njudge_max_output_tokens = 128\n", encoding="utf-8")
    output = tmp_path / "diagnosis.json"
    args = build_parser().parse_args([
        "diagnose", "--run", str(run), "--out", str(output), "--judge", "--config", str(config),
    ])

    assert args.func(args) == 0

    report_text = output.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["scope"] == "diagnostic_only"
    assert report["cases"][0]["original_release_pass"] is False
    assert report["reviews"][0]["original_product_pass"] is False
    assert report["reviews"][0]["original_judge_pass"] is None
    outcome = report["diagnostic_judgments"][0]["outcome"]
    assert outcome["status"] == "succeeded"
    assert outcome["assessment"]["body_support"] == "supported"
    assert outcome["model_name"] == "gpt-6.1-terra"
    assert outcome["input_stats"]["output_token_limit"] == 128
    assert "test-key" not in report_text
    assert len(requests) == 1
    payload = json.loads(requests[0]["messages"][1]["content"])
    assert payload["case_oracle"] == case.oracle.model_dump(mode="json")
    assert requests[0]["max_completion_tokens"] == 128
    assert {path: path.read_bytes() for path in original} == original
