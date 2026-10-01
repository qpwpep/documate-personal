import json

import pytest

from src.eval.config_models import BenchmarkConfig
from src.eval.online_runner import run_online_benchmark
from tests.eval.response_fixtures import plain_response, sse_http_response


@pytest.mark.parametrize(
    "override",
    [
        {key: 0 for key in (
            "answer_quality", "reference_coverage", "citation_traceability",
            "tool_choice", "format_language", "llm_judge",
        )},
        {"reference_coverage": 0.1},
        {"citation_traceability": 0.1},
    ],
    ids=["zero-total", "reference-reactivation", "citation-reactivation"],
)
def test_all_case_weights_are_validated_before_external_execution(tmp_path, monkeypatch, override):
    fixtures = tmp_path / "cases.jsonl"
    cases = [
        {"case_id": "first-valid", "category": "docs_only", "query": "Explain a feature."},
        {"case_id": "second-invalid", "category": "tool_action", "query": "Share the answer.",
         "require_official_citation": False, "require_local_citation": False,
         "weight_override": override},
    ]
    fixtures.write_text("\n".join(json.dumps(case) for case in cases), encoding="utf-8")
    external_calls = []

    def unexpected_external_call(*args, **kwargs):
        external_calls.append((args, kwargs))
        raise AssertionError("A judge or HTTP client was reached before validating all case weights")

    monkeypatch.setattr("src.eval.judge_llm.ChatOpenAI", unexpected_external_call)
    monkeypatch.setattr("requests.sessions.Session.request", unexpected_external_call)
    with pytest.raises(ValueError, match="second-invalid"):
        run_online_benchmark(
            fixtures_path=fixtures,
            endpoint="http://benchmark.invalid",
            config=BenchmarkConfig(),
            config_path=tmp_path / "config.toml",
            output_root=tmp_path / "output",
            track="smoke",
        )

    assert external_calls == []
    assert not (tmp_path / "output").exists()


def test_limit_selects_cases_before_weight_preflight(tmp_path, monkeypatch, empty_upload_manifest_http):
    fixtures = tmp_path / "cases.jsonl"
    fixtures.write_text("\n".join(json.dumps(case) for case in [
        {"case_id": "selected", "category": "tool_action", "query": "Share the answer."},
        {"case_id": "excluded", "category": "tool_action", "query": "Do not run this case.",
         "weight_override": {"citation_traceability": 0.1}},
    ]), encoding="utf-8")
    questions = []

    def post(url, *, json, **kwargs):
        questions.append(json["query"])
        return sse_http_response(200, {
            "upload_manifest": {"epoch": "fixture-epoch", "revision": 0, "files": []},
            "response": plain_response("Shared the answer."),
        })

    monkeypatch.setattr("requests.post", post)
    _, results, summary = run_online_benchmark(
        fixtures_path=fixtures,
        endpoint="http://benchmark.invalid",
        config=BenchmarkConfig(judge_enabled=False),
        config_path=tmp_path / "config.toml",
        output_root=tmp_path / "output",
        track="smoke",
        limit=1,
    )

    assert questions == ["Share the answer."]
    assert [result.case_id for result in results] == ["selected"]
    assert summary.requested_limit == 1
    assert results[0].effective_weights["reference_coverage"] == 0
    assert results[0].effective_weights["citation_traceability"] == 0
