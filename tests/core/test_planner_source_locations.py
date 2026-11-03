"""Opt-in real-provider checks for source locations versus literal anchors.

From the repository root in Git Bash:
    RUN_LIVE_PLANNER_SOURCE_LOCATIONS=1 uv run --locked pytest -q tests/core/test_planner_source_locations.py

This sends the synthetic prompts and upload manifests below to the configured
planner model over the network and incurs provider usage. The normal test suite
skips these checks. Model responses are not mocked; results can vary across runs.
Only planning runs: no attachment contents, retrieval, or synthesis are sent.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from langchain_core.messages import HumanMessage

from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import ParsedDocument
from src.core.evidence import SearchHit
from src.core.request_contracts import UserTurnSnapshot
from src.core.uploads import UploadFileInfo
from src.infra.llm import LLMRegistry, build_llm_registry
from src.infra.settings import get_settings
from src.runtime.nodes.planner import make_planner_node


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_PLANNER_SOURCE_LOCATIONS") != "1",
    reason="Set RUN_LIVE_PLANNER_SOURCE_LOCATIONS=1 to call the configured planner provider.",
)


@dataclass(frozen=True)
class _ManifestOnlyRetriever:
    """Advertise available uploads without starting a vector database or embedding call."""

    upload_files: tuple[UploadFileInfo, ...]
    source_documents: tuple[ParsedDocument, ...] = ()

    def search(
        self, query: str, k: int = 4, *, file_ids: Sequence[str] = (), mode: str = "hybrid",
    ) -> list[SearchHit]:
        raise AssertionError("The planner must not execute attachment retrieval")


@dataclass(frozen=True)
class _LocationCase:
    file_name: str
    prompt: str
    expected_aspects: tuple[str, ...]
    query_term_groups: tuple[tuple[str, ...], ...]
    inline_mode: str | None = None


@pytest.fixture(scope="module")
def live_planner_registry() -> LLMRegistry:
    return build_llm_registry(get_settings().model_copy(update={"verbose": False}))


@pytest.mark.parametrize("case", [
    pytest.param(_LocationCase(
        file_name="p2_lines.py",
        prompt=("첨부한 p2_lines.py의 1행과 2행을 원문 코드와 줄 번호를 함께 넣어 한 문장으로 설명해줘. "
                "각 원문 코드는 인라인 코드로 표시하고, 한 문단 안에서 “1행은 …이고 2행은 …입니다” 형태로 작성해줘."),
        expected_aspects=(),
        query_term_groups=(("1", "first"), ("2", "second"), ("행", "줄", "line")),
        inline_mode="required",
    ), id="original-korean-lines"),
    pytest.param(_LocationCase(
        file_name="range_example.py",
        prompt="Explain lines 14 through 18 of the attached range_example.py, including their original code and line numbers.",
        expected_aspects=(),
        query_term_groups=(("14", "fourteenth"), ("18", "eighteenth"), ("line", "행", "줄")),
    ), id="english-line-range"),
    pytest.param(_LocationCase(
        file_name="analysis.ipynb",
        prompt="첨부한 analysis.ipynb의 세 번째 셀에 있는 코드를 원문과 함께 설명해줘.",
        expected_aspects=(),
        query_term_groups=(("3", "세 번째", "세번째", "third"), ("셀", "cell")),
    ), id="notebook-cell"),
    pytest.param(_LocationCase(
        file_name="report.pdf",
        prompt="Summarize page 7 of the attached report.pdf and cite that page.",
        expected_aspects=(),
        query_term_groups=(("7", "seventh"), ("page", "페이지", "쪽")),
    ), id="pdf-page"),
    pytest.param(_LocationCase(
        file_name="labels.py",
        prompt=("첨부한 labels.py에서 정확한 원문 문자열 `1행`이 포함된 부분을 찾아 인용해줘. "
                "파일의 첫 번째 줄을 뜻하는 것이 아니라, 문자 그대로 `1행`을 검색하는 요청이야."),
        expected_aspects=("1행",),
        query_term_groups=(("1행",),),
    ), id="literal-coordinate-looking-text"),
    pytest.param(_LocationCase(
        file_name="snippets.py",
        prompt="Explain lines 1 and 2 of the attached snippets.py in one sentence. Format each quoted source statement as inline code.",
        expected_aspects=(),
        query_term_groups=(("1", "first"), ("2", "second"), ("line", "행", "줄")),
        inline_mode="required",
    ), id="english-inline-code"),
    pytest.param(_LocationCase(
        file_name="plain.py",
        prompt="Explain line 1 of the attached plain.py. Use plain prose; do not format any text as inline code.",
        expected_aspects=(),
        query_term_groups=(("1", "first"), ("line", "행", "줄")),
        inline_mode="forbidden",
    ), id="forbidden-inline-code"),
    pytest.param(_LocationCase(
        file_name="optional.py",
        prompt="Explain line 1 of the attached optional.py. Inline code formatting would be nice, but plain text is fine.",
        expected_aspects=(),
        query_term_groups=(("1", "first"), ("line", "행", "줄")),
        inline_mode="preferred",
    ), id="preferred-inline-code"),
])
def test_live_planner_distinguishes_source_locations_from_literal_anchors(
    case: _LocationCase, live_planner_registry: LLMRegistry,
) -> None:
    selected = UploadFileInfo(
        file_id="selected-attachment", name=case.file_name, size_bytes=128,
        content_hash="sha256:" + "a" * 64,
        source_uri="upload:///location-planning/selected-attachment",
    )
    other = UploadFileInfo(
        file_id="other-attachment", name="other_" + case.file_name, size_bytes=128,
        content_hash="sha256:" + "b" * 64,
        source_uri="upload:///location-planning/other-attachment",
    )
    attachments = (selected, other)
    state = build_graph_state_input(
        user_input=case.prompt, current_turn_id="location-request",
        user_turns=(UserTurnSnapshot(turn_id="location-request", text=case.prompt),),
        messages=[HumanMessage(id="location-request", content=case.prompt)],
        upload_files=attachments, retriever=_ManifestOnlyRetriever(attachments),
    )
    planner = make_planner_node(live_planner_registry.llm_planner, False)

    updates = planner(state)
    result = updates["planner"].output

    assert result.use_retrieval
    assert result.tasks
    inline_modes = [item.mode for item in updates["runtime"].request_contract.answer.format if item.kind == "inline_code"]
    assert inline_modes == ([case.inline_mode] if case.inline_mode else [])
    for task in result.tasks:
        assert task.route == "upload"
        assert task.requirement.file_ids == ["selected-attachment"]
        assert task.requirement.match == "topic"
        assert task.requirement.symbols == []
        assert task.requirement.aspects == list(case.expected_aspects)

    # These case-specific terms describe the requested position. Remove the
    # filename first so a digit in p2_lines.py cannot stand in for line 2.
    queries = " ".join(task.query for task in result.tasks).replace(case.file_name, "").casefold()
    for alternatives in case.query_term_groups:
        assert any(term in queries for term in alternatives), f"Query lost a requested location/content term: {alternatives}"
