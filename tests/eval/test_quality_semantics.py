"""Quality rules must not invent task requirements from incidental representations."""

from __future__ import annotations

import pytest

from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.eval.config_models import BenchmarkCase, CaseOracle
from src.eval.metric_rules import score_answer_quality


def _hit(text: str) -> SearchHit:
    snapshot = build_snapshot(
        source_uri="https://numpy.org/doc/stable/", title="Array guide",
        media_type="text/plain", source_type="official", content=text,
        parser="test", parser_version="1",
    )
    evidence = build_evidence(
        snapshot=snapshot,
        element=DocumentElement(
            element_id="paragraph", kind="paragraph", text=text,
            anchors=[SourceAnchor(kind="web", start=0, end=len(text))],
        ),
    )
    return SearchHit(
        evidence=evidence, score=RetrievalScore(metric="rank", raw=1, direction="lower"),
        rank=1,
    )


@pytest.mark.parametrize(
    ("query", "answer", "facts"),
    [
        (
            "규칙과 첨부 배열의 shape을 사용해 연산 결과를 설명해줘.",
            "(3, 1)과 (1, 4)는 축마다 한쪽 크기가 1이어서 (3, 4)로 확장됩니다.",
            ["(3, 1)", "(1, 4)", "(3, 4)"],
        ),
        (
            "공식 설명과 첨부 코드의 반환 타입이 어떻게 다른지 알려줘.",
            "공식 함수는 ndarray를 반환합니다. 첨부 코드는 tolist를 거쳐 list를 반환합니다.",
            ["ndarray", "tolist", "list"],
        ),
    ],
)
def test_comparison_word_does_not_change_quality_for_the_same_answer(query, answer, facts):
    case = BenchmarkCase(
        case_id="semantic-task", category="hybrid", query=query, must_include=facts,
        oracle=CaseOracle(required_facts=facts, expected_behaviors=[query]),
    )

    score = score_answer_quality(case, answer, [])
    with_marker = score_answer_quality(case, answer + " 비교", [])

    assert score == with_marker == 1.0


@pytest.mark.parametrize("category", ["docs_only", "rag_only", "hybrid"])
def test_evidence_category_does_not_invent_an_answer_requirement(category):
    case = BenchmarkCase(
        case_id="same-request", category=category,
        query="배열의 차원 수를 알려줘.", must_include=["2"],
    )

    assert score_answer_quality(case, "이 배열의 차원 수는 2입니다.", []) == 1.0


def test_requested_quotation_is_invariant_to_retrieval_chunk_boundaries():
    source = (
        "The dimensions of two arrays are compatible when they are equal, "
        "or one of them is one. Missing leading dimensions are treated as "
        "dimensions of size one when the arrays are aligned for broadcasting."
    )
    case = BenchmarkCase(
        case_id="quotation", category="docs_only",
        query="해당 규칙의 원문을 그대로 인용해줘.", must_include=[source],
        oracle=CaseOracle(expected_behaviors=["Quote the requested rule verbatim."]),
    )
    answer = f"요청한 규칙의 원문입니다:\n> {source}"
    exact = score_answer_quality(case, answer, [_hit(source)])
    expanded = score_answer_quality(case, answer, [_hit(source + " See the next example.")])
    reused = score_answer_quality(case, answer, [])

    assert exact == expanded == reused == 1.0


def test_incidental_comparison_word_cannot_satisfy_explicit_case_constraints():
    case = BenchmarkCase(
        case_id="explicit-constraints", category="hybrid", query="두 반환 타입을 설명해줘.",
        must_include=["ndarray", "list"], must_not_include=["동일한 타입"],
    )

    complete = score_answer_quality(
        case, "공식 함수는 ndarray를 반환하고 첨부 코드는 list를 반환합니다.", [],
    )
    missing = score_answer_quality(case, "비교", [])
    forbidden = score_answer_quality(case, "ndarray와 list는 동일한 타입입니다. 비교", [])

    assert complete == 1.0
    assert missing < complete
    assert forbidden < complete
    assert score_answer_quality(case, " ", []) == 0.0
