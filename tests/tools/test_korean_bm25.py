"""Korean retrieval is observed through the upload search tool and original sources."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings

from src.core.planner_schema import RetrievalRequirement
from src.core.rules import HybridSearchConfig
from src.core.uploads import UploadRecord
from src.infra.settings import AppSettings
from src.infra.tools import build_tool_registry
from src.infra.tools.local_rag import build_upload_retriever, build_upload_search_tool


class DistantDeliveryEmbeddings(Embeddings):
    """An external embedding boundary that misses delivery passages at top one."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[100.0 if "배송" in text else 0.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 0.0]


class OrderedPassageEmbeddings(Embeddings):
    def __init__(self, passages: list[str]) -> None:
        self._vectors = {text: [float(rank), 0.0] for rank, text in enumerate(passages)}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 0.0]


@pytest.fixture
def indexed_notebooks(tmp_path):
    @contextmanager
    def indexed(contents: dict[str, list[str]], *, hybrid_config=None, embeddings=None):
        files = []
        for name, passages in contents.items():
            data = json.dumps({
                "nbformat": 4, "nbformat_minor": 5, "metadata": {},
                "cells": [{"id": f"passage-{i}", "cell_type": "markdown", "metadata": {}, "source": passage}
                          for i, passage in enumerate(passages)],
            }, ensure_ascii=False).encode("utf-8")
            path = tmp_path / name
            path.write_bytes(data)
            file_id = uuid4().hex
            files.append(UploadRecord(file_id=file_id, name=name, path=str(path), size_bytes=len(data),
                                      content_hash="sha256:" + hashlib.sha256(data).hexdigest(),
                                      source_uri=f"upload://korean-test/{file_id}"))
        handle = build_upload_retriever(files, session_id="korean-test", generation=uuid4().hex,
                                       embeddings=embeddings or DistantDeliveryEmbeddings(), hybrid_config=hybrid_config)
        try:
            yield handle.retriever, files
        finally:
            handle.cleanup()

    return indexed


@pytest.mark.parametrize("mode, metric", [("bm25", "bm25"), ("hybrid", "rrf")])
def test_korean_bm25_recovers_inflected_term_outside_dense_top_one(indexed_notebooks, mode, metric):
    source = "배송이 늦으면 판매자에게 연락합니다."
    with indexed_notebooks({"guide.ipynb": ["보관 온도는 영상 오 도입니다.", source]},
                           hybrid_config=HybridSearchConfig(lexical_coverage_power=2)) as (retriever, _):
        search = build_upload_search_tool()
        baseline = search("배송은", k=1, retriever=retriever, requirement=RetrievalRequirement(), mode="baseline")
        lexical = search("배송은", k=1, retriever=retriever, requirement=RetrievalRequirement(), mode=mode)
    assert "보관 온도" in baseline.hits[0].evidence.excerpt
    assert [hit.evidence.excerpt for hit in lexical.hits] == [source]
    assert lexical.hits[0].score.metric == metric
    assert lexical.hits[0].score.direction == "higher"
    assert lexical.hits[0].score.normalized is None


@pytest.mark.parametrize("mode, metric, expected", [
    ("baseline", "l2", "보관 온도 안내"),
    ("bm25", "bm25", "배송은 당일 출발합니다."),
    ("hybrid", "rrf", "배송은 당일 출발합니다."),
])
def test_environment_selects_search_default_and_allows_explicit_override(indexed_notebooks, tmp_path,
                                                                       mode, metric, expected):
    env_file = tmp_path / "search.env"
    env_file.write_text(f"UPLOAD_SEARCH_MODE={mode}\n", encoding="utf-8")
    settings = AppSettings(_env_file=env_file, slack_bot_token="")
    search = build_tool_registry(settings).upload_search_tool
    with indexed_notebooks({"guide.ipynb": ["보관 온도 안내", "배송은 당일 출발합니다."]}) as (retriever, _):
        default = search("배송은", k=1, retriever=retriever, requirement=RetrievalRequirement())
        baseline = search("배송은", k=1, retriever=retriever, requirement=RetrievalRequirement(), mode="baseline")
        lexical = search("배송은", k=1, retriever=retriever, requirement=RetrievalRequirement(), mode="bm25")
    assert [(hit.evidence.excerpt, hit.score.metric) for hit in default.hits] == [(expected, metric)]
    assert [(hit.evidence.excerpt, hit.score.metric) for hit in baseline.hits] == [("보관 온도 안내", "l2")]
    assert [(hit.evidence.excerpt, hit.score.metric) for hit in lexical.hits] == [("배송은 당일 출발합니다.", "bm25")]


def test_baseline_keeps_original_lexical_order_and_raw_distances(indexed_notebooks):
    target = "배송 안내"
    other = "보관 온도 안내"
    with indexed_notebooks({"guide.ipynb": [other, target]}) as (retriever, _):
        result = build_upload_search_tool()("배송", k=2, retriever=retriever,
                                           requirement=RetrievalRequirement(), mode="baseline")
    assert [hit.evidence.excerpt for hit in result.hits] == [target, other]
    assert [hit.score.raw for hit in result.hits] == [10000.0, 0.0]
    assert [hit.score.normalized for hit in result.hits] == [0.0, 1.0]


@pytest.mark.parametrize("coverage_power", [0, 2])
def test_hybrid_combines_lexical_and_dense_candidates(indexed_notebooks, coverage_power):
    target = "배송이 늦으면 판매자에게 연락합니다."
    other = "보관 온도는 영상 오 도입니다."
    with indexed_notebooks({"guide.ipynb": [other, target]},
                           hybrid_config=HybridSearchConfig(lexical_coverage_power=coverage_power)) as (retriever, _):
        result = build_upload_search_tool()("배송은", k=2, retriever=retriever,
                                           requirement=RetrievalRequirement(), mode="hybrid")
    assert [hit.evidence.excerpt for hit in result.hits] == [target, other]
    assert all(hit.score.metric == "rrf" and hit.score.direction == "higher" for hit in result.hits)
    assert all(hit.score.normalized is None for hit in result.hits)


@pytest.mark.parametrize("query, source", [
    ("행", "행마다 날짜를 확인합니다."),
    ("값", "값을 음수로 지정할 수 없습니다."),
    ("fit_transform", "입력에는 fit_transform 메서드를 적용합니다."),
    ("2.9", "이 동작은 2.9 버전에 적용됩니다."),
])
def test_bm25_preserves_short_terms_identifiers_and_versions(indexed_notebooks, query, source):
    with indexed_notebooks({"guide.ipynb": ["재고 기록을 보관합니다.", source]}) as (retriever, _):
        result = build_upload_search_tool()(query, k=1, retriever=retriever,
                                           requirement=RetrievalRequirement(), mode="bm25")
    assert [hit.evidence.excerpt for hit in result.hits] == [source]


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_file_scope_and_per_file_candidates_survive_ranking(indexed_notebooks, mode):
    contents = {"first.ipynb": ["배송은 당일 출발합니다."],
                "second.ipynb": ["배송은 영업일 기준 이틀이 필요합니다."],
                "outside.ipynb": ["배송 배송 배송 배송"]}
    with indexed_notebooks(contents) as (retriever, files):
        result = build_upload_search_tool()("배송", k=1, retriever=retriever,
            requirement=RetrievalRequirement(file_ids=[file.file_id for file in files[:2]]), mode=mode)
    assert {hit.evidence.snapshot.title for hit in result.hits} == {"first.ipynb", "second.ipynb"}
    assert [hit.rank for hit in result.hits] == [1, 2]
    assert result.diagnostics.missing_requirements == []
    for hit in result.hits:
        assert hit.evidence.excerpt == contents[hit.evidence.snapshot.title][0]


def test_hybrid_dense_ranking_does_not_depend_on_requested_file_order(indexed_notebooks):
    with indexed_notebooks({"far.ipynb": ["배송 안내"], "near.ipynb": ["보관 온도 안내"]}) as (retriever, files):
        search = build_upload_search_tool()
        for ordered in (files, list(reversed(files))):
            result = search("cold", k=1, retriever=retriever,
                requirement=RetrievalRequirement(file_ids=[file.file_id for file in ordered]), mode="hybrid")
            assert [hit.evidence.snapshot.title for hit in result.hits] == ["near.ipynb", "far.ipynb"]


@pytest.mark.parametrize("query", ["!!!", "qzxv_nonexistent"])
def test_bm25_does_not_fill_missing_matches_with_zero_score_documents(indexed_notebooks, query):
    with indexed_notebooks({"guide.ipynb": ["배송은 당일 출발합니다."]}) as (retriever, files):
        result = build_upload_search_tool()(query, k=4, retriever=retriever,
            requirement=RetrievalRequirement(file_ids=[files[0].file_id]), mode="bm25")
    assert result.hits == []
    assert result.diagnostics.status == "no_result"
    assert result.diagnostics.answerability == "missing"
    assert result.diagnostics.missing_requirements == [f"file:{files[0].file_id}"]


def test_hybrid_retains_dense_candidates_without_lexical_matches(indexed_notebooks):
    source = "보관 온도는 영상 오 도입니다."
    with indexed_notebooks({"guide.ipynb": [source]}) as (retriever, _):
        result = build_upload_search_tool()("cold storage", k=1, retriever=retriever,
                                           requirement=RetrievalRequirement(), mode="hybrid")
    assert [hit.evidence.excerpt for hit in result.hits] == [source]
    assert result.hits[0].score.metric == "rrf"


def test_configurable_hybrid_preserves_a_cross_language_dense_match(indexed_notebooks):
    target = "Convert timestamps to UTC before joining the two tables."
    distractor = "배송 시각을 기록합니다."
    with indexed_notebooks({"guide.ipynb": [target, distractor]},
                           hybrid_config=HybridSearchConfig(dense_weight=0.7, rrf_rank_constant=0,
                                                           lexical_coverage_power=0)) as (retriever, _):
        result = build_upload_search_tool()("시각을 맞추는 방법", k=1, retriever=retriever,
                                           requirement=RetrievalRequirement(), mode="hybrid")
    assert [hit.evidence.excerpt for hit in result.hits] == [target]


def test_equal_fusion_scores_prefer_the_more_precise_lexical_match(indexed_notebooks):
    target = "배송 기록은 보관합니다."
    other = "기록"
    with indexed_notebooks({"guide.ipynb": [other, target]},
                           hybrid_config=HybridSearchConfig(lexical_coverage_power=0)) as (retriever, _):
        search = build_upload_search_tool()
        lexical = search("배송 기록", k=2, retriever=retriever,
                         requirement=RetrievalRequirement(), mode="bm25")
        hybrid = search("배송 기록", k=1, retriever=retriever,
                        requirement=RetrievalRequirement(), mode="hybrid")
    assert [hit.evidence.excerpt for hit in lexical.hits] == [target, other]
    assert [hit.evidence.excerpt for hit in hybrid.hits] == [target]


@pytest.mark.parametrize("weight, query, expected", [
    (0.0, "cold storage", []),
    (1.0, "배송", ["보관 온도 안내"]),
])
def test_zero_weight_branch_does_not_fill_hybrid_results(indexed_notebooks, weight, query, expected):
    with indexed_notebooks({"guide.ipynb": ["보관 온도 안내", "배송 안내"]},
                           hybrid_config=HybridSearchConfig(dense_weight=weight, rrf_rank_constant=0)) as (retriever, _):
        result = build_upload_search_tool()(query, k=1, retriever=retriever,
                                           requirement=RetrievalRequirement(), mode="hybrid")
    assert [hit.evidence.excerpt for hit in result.hits] == expected


@pytest.mark.parametrize("query", [
    "배송이 끊기면 다시 이어 받나요?",
    "배송 qzxvone qzxvtwo qzxvthree",
])
def test_partial_lexical_matches_do_not_evict_cross_language_evidence(indexed_notebooks, query):
    target = "An interrupted delivery resumes from the last verified block."
    passages = ["Store apples in a cool place.", "Invoices are issued monthly.", target,
                "Archive completed records.", "배송 안내", "배송 정보", "배송 조회", "배송 공지"]
    hybrid_results = {}
    lexical_results = {}
    configurations = {"unattenuated": HybridSearchConfig(lexical_coverage_power=0),
                      "default": HybridSearchConfig(), "squared": HybridSearchConfig(lexical_coverage_power=2)}
    for name, configuration in configurations.items():
        with indexed_notebooks({"guide.ipynb": passages}, embeddings=OrderedPassageEmbeddings(passages),
                               hybrid_config=configuration) as (retriever, _):
            search = build_upload_search_tool()
            hybrid = search(query, k=4, retriever=retriever, requirement=RetrievalRequirement(), mode="hybrid")
            lexical = search(query, k=4, retriever=retriever, requirement=RetrievalRequirement(), mode="bm25")
            hybrid_results[name] = [hit.evidence.excerpt for hit in hybrid.hits]
            lexical_results[name] = [(hit.evidence.excerpt, hit.score.raw) for hit in lexical.hits]
    assert target not in hybrid_results["unattenuated"]
    assert target in hybrid_results["default"]
    assert hybrid_results["squared"][:3] == passages[:3]
    assert len(hybrid_results["default"]) == len(hybrid_results["squared"]) == 4
    assert lexical_results["default"] == lexical_results["squared"] == lexical_results["unattenuated"]
