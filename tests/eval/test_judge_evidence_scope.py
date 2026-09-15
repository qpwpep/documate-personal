import json
from types import SimpleNamespace

from src.eval.config_models import BenchmarkCase
from src.eval.judge_llm import LLMJudge
from tests.web.answer_fixtures import cited_response
from tests.eval.response_fixtures import answer_provenance


def test_judge_receives_verified_reused_evidence_separately_from_final_tool_observations():
    """A follow-up evaluation sees its selected source without turning preparation searches into final calls."""
    response = cited_response()
    source = {"ref": "previous", "response_hash": response.content_hash,
              "citation_ids": [citation.evidence.id for citation in response.citations]}
    provenance = answer_provenance(response, source=source, body_kind="copy_answer")
    scope = {"status": "complete", "errors": [],
             "verified_evidence": [citation.evidence.model_dump(mode="json") for citation in response.citations]}
    received = []

    class JudgeModelBoundary:
        def invoke(self, messages):
            received.append(json.loads(messages[-1].content))
            return SimpleNamespace(content=json.dumps({
                "score": 1.0, "reason": "source retained", "subscores": {
                    key: 1.0 for key in ("answer_quality", "groundedness", "citation_traceability", "tool_choice", "format_language")
                },
            }))

    judge = LLMJudge(model_name="fixture", enabled=False)
    judge.enabled = True
    judge.client = JudgeModelBoundary()
    outcome = judge.score_case(
        case=BenchmarkCase(case_id="copy", category="tool_action", query="Save that answer",
                           setup_turns=["Explain this file"], require_local_citation=True),
        response=response, tool_calls=["save_text"], observed_hits=[],
        conversation=[{"query": "Explain this file", "response": response, "observed_hits": []}],
        evidence_scope=scope, answer_provenance=provenance,
    )

    assert outcome.status == "succeeded" and outcome.score == 1.0 and outcome.error is None
    assert len(received) == 1
    assert received[0]["called_tools"] == ["save_text"]
    assert received[0]["observed_hits"] == []
    assert received[0]["evidence_scope"] == scope
    assert received[0]["answer_provenance"] == provenance
    assert received[0]["case"]["require_local_citation"] is True


def test_judge_does_not_treat_missing_provenance_as_a_complete_new_evaluation():
    """Unavailable provenance stops a paid judge call instead of silently blessing incomplete evidence."""
    class UnusedModelBoundary:
        def invoke(self, messages):
            raise AssertionError("incomplete evidence must not reach the model")

    judge = LLMJudge(model_name="fixture", enabled=False)
    judge.enabled = True
    judge.client = UnusedModelBoundary()
    outcome = judge.score_case(
        case=BenchmarkCase(case_id="missing", category="tool_action", query="Save that answer"),
        response=cited_response(), tool_calls=["save_text"],
        evidence_scope={"status": "unavailable", "errors": ["missing packet"], "verified_evidence": []},
    )

    assert outcome.status == "not_run"
    assert outcome.score is None
    assert outcome.error == "invalid_eval: judge payload is incomplete"
