"""Retrieval and evaluation share official sources, independently of UI links."""
from __future__ import annotations

import json

import pytest
import requests

from src.core.answer_schema import AnswerDocument, finalize_answer
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import RetrievalRequirement
from src.eval.config_models import BenchmarkCase
from src.eval.metric_rules import score_citation_traceability
from src.eval.result_models import EvidenceAssessment
from src.infra.settings import AppSettings
from src.infra.tools.docs_search.tool import build_docs_search_tool


@pytest.mark.parametrize("url", [
    "https://pydantic.dev/docs/validation/latest/get-started/",
    "https://docs.pears.com/reference/bare",
    "https://pydantic.dev/docs/validation/dev/concepts/strict_mode",
    "https://pydantic.dev/docs/validation/2.0/usage/strict_mode",
    "https://pydantic.dev/docs/validation/2.4/api/pydantic/config",
    "https://docs.pydantic.dev/docs/validation/latest/concepts/strict_mode/",
    "https://docs.pydantic.dev/docs/validation/2.4/api/pydantic/config",
])
def test_official_document_survives_retrieval_and_citation_evaluation(monkeypatch, url):
    source_text = "The reference describes the supported interface."

    def post(_url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"results": [{
            "url": url, "title": "Reference", "raw_content": source_text, "score": 0.9,
        }]}).encode()
        return response

    monkeypatch.setattr(requests, "post", post)
    tool = build_docs_search_tool(AppSettings(_env_file=None, tavily_api_key="fixture"))
    result = tool("official reference", requirement=RetrievalRequirement())
    assert len(result.hits) == 1
    evidence = result.hits[0].evidence
    answer = finalize_answer(AnswerDocument.model_validate({"blocks": [{
        "type": "paragraph", "content": [{"text": source_text, "basis": "source", "refs": [evidence.id]}],
    }]}), [evidence], retrieval_required=True)
    case = BenchmarkCase(
        case_id="official-source", category="docs_only", query="Explain the official reference.",
        require_official_citation=True,
    )

    assert score_citation_traceability(
        case=case, response=answer, observed_hits=result.hits, called_tools=["tavily_search"],
    ) == 1.0


@pytest.mark.parametrize("url, expected", [
    ("https://numpy.org/community/", 0.0),
    ("https://numpy.org/doc/stableish/", 0.0),
    ("https://numpy.org/doc/stable/../community/", 0.0),
    ("https://numpy.org/doc/stable/%2e%2e/community/", 0.0),
    ("https://numpy.org.attacker.example/doc/stable/", 0.0),
    ("https://docs.numpy.org/doc/stable/", 0.0),
    ("http://numpy.org/doc/stable/", 0.0),
    ("https://pydantic.dev/docs/validation/2.4evil/concepts/strict_mode/", 0.0),
    ("https://pydantic.dev/docs/validation/latestish/concepts/strict_mode/", 0.0),
    ("https://pydantic.dev/docs/validation/2.4/%2e%2e/blog/", 0.0),
    ("https://www.numpy.org/doc/stable/", 1.0),
    ("https://numpy.org/doc/2.4/reference/generated/numpy.reshape.html", 1.0),
    ("https://docs.python.org/ko/3/library/pathlib.html", 1.0),
])
@pytest.mark.parametrize("scope_status", [None, "complete", "invalid"])
def test_observed_citation_still_requires_an_allowed_official_document_url(url, expected, scope_status):
    text = "This is captured source text."
    evidence = build_evidence(
        snapshot=build_snapshot(
            source_uri=url, title="Captured reference", media_type="text/plain",
            source_type="official", content=text, parser="fixture", parser_version="1",
        ),
        element=DocumentElement(
            element_id="body", kind="paragraph", text=text,
            anchors=[SourceAnchor(kind="web", start=0, end=len(text))],
        ),
    )
    hit = SearchHit(evidence=evidence, score=RetrievalScore(metric="rank", raw=1, direction="lower"), rank=1)
    answer = finalize_answer(AnswerDocument.model_validate({"blocks": [{
        "type": "paragraph", "content": [{"text": text, "basis": "source", "refs": [evidence.id]}],
    }]}), [evidence], retrieval_required=True)
    case = BenchmarkCase(case_id="source-policy", category="docs_only", query="Explain the reference.", require_official_citation=True)

    assert score_citation_traceability(
        case=case, response=answer, observed_hits=[hit] if scope_status is None else [],
        called_tools=["tavily_search"] if scope_status is None else [],
        evidence_scope=EvidenceAssessment(status=scope_status, verified_evidence=[evidence]) if scope_status else None,
    ) == (0.0 if scope_status == "invalid" else expected)


@pytest.mark.parametrize("reuse_scope", [False, True])
def test_allowed_citation_cannot_mask_another_citation_outside_document_policy(reuse_scope):
    evidence = []
    for index, url in enumerate([
        "https://pydantic.dev/docs/validation/latest/concepts/strict_mode/",
        "https://numpy.org/community/",
    ]):
        text = f"Captured statement {index}."
        evidence.append(build_evidence(
            snapshot=build_snapshot(
                source_uri=url, title="Reference", media_type="text/plain", source_type="official",
                content=text, parser="fixture", parser_version="1",
            ),
            element=DocumentElement(
                element_id=f"paragraph-{index}", kind="paragraph", text=text,
                anchors=[SourceAnchor(kind="web", start=0, end=len(text))],
            ),
        ))
    answer = finalize_answer(AnswerDocument.model_validate({"blocks": [
        {"type": "paragraph", "content": [{"text": ref.excerpt, "basis": "source", "refs": [ref.id]}]}
        for ref in evidence
    ]}), evidence, retrieval_required=True)
    hits = [SearchHit(evidence=ref, score=RetrievalScore(metric="rank", raw=1, direction="lower"), rank=1) for ref in evidence]
    case = BenchmarkCase(case_id="mixed-official-sources", category="docs_only", query="Explain the references.", require_official_citation=True)

    assert score_citation_traceability(
        case=case, response=answer, observed_hits=[] if reuse_scope else hits,
        called_tools=[] if reuse_scope else ["tavily_search"],
        evidence_scope=EvidenceAssessment(status="complete", verified_evidence=evidence) if reuse_scope else None,
    ) == 0.5
