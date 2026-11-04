"""Opt-in real-provider checks for queried identity and source-selection roles.

From the repository root in Git Bash:
    RUN_LIVE_PLANNER_SYMBOL_IDENTITY=1 uv run --locked pytest -q -s tests/core/test_planner_symbol_identity_live.py

These checks send the synthetic prompts and attachment manifests below to the
configured planner provider and incur API usage. The normal suite skips them.
Model responses are not mocked and can vary across runs. Only planning runs:
no attachment contents, retrieval, synthesis, file saving, or Slack sending.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import ParsedDocument
from src.core.evidence import SearchHit
from src.core.request_contracts import UserTurnSnapshot
from src.core.uploads import UploadFileInfo
from src.infra.llm import LLMRegistry, build_llm_registry
from src.infra.llm_boundary import capture_structured_calls
from src.infra.settings import get_settings
from src.runtime.nodes.planner import make_planner_node


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_PLANNER_SYMBOL_IDENTITY") != "1",
    reason="Set RUN_LIVE_PLANNER_SYMBOL_IDENTITY=1 to call the configured planner provider.",
)


@dataclass(frozen=True)
class _ManifestOnlyRetriever:
    """Provide the attachment catalog without constructing an index or embeddings."""

    upload_files: tuple[UploadFileInfo, ...]
    source_documents: tuple[ParsedDocument, ...] = ()

    def search(
        self, query: str, k: int = 4, *, file_ids: Sequence[str] = (), mode: str = "hybrid",
    ) -> list[SearchHit]:
        raise AssertionError("The planner must not execute attachment retrieval")


@pytest.fixture(scope="module")
def live_planner_registry() -> LLMRegistry:
    return build_llm_registry(get_settings().model_copy(update={"verbose": False}))


@pytest.mark.parametrize("prompt, filename, expected_symbol, expected_aspects", [
    pytest.param(
        "첨부 capture.py의 마지막 np.array 호출을 NumPy API 사용 사례로 볼 수 있어? "
        "match 문이 np에 어떤 영향을 주는지 코드 원문을 근거로 설명해줘.",
        "capture.py",
        "np.array",
        (),
        id="queried-identity-keeps-observed-call",
    ),
    pytest.param(
        "첨부한 branch.py에서 실제 NumPy의 numpy.array를 호출하는 문장만 원문 그대로 인용하고 줄 번호를 알려줘. "
        "다른 객체의 array 호출은 포함하지 마.",
        "branch.py",
        "numpy.array",
        (),
        id="symbol-identity-allows-import-aliases",
    ),
    pytest.param(
        "첨부한 branch.py에서 실제 NumPy의 numpy.array를 호출하면서 코드에 정확히 numpy.array라고 적힌 문장만 "
        "찾아 원문과 줄 번호를 알려줘. 별칭 np.array로 적힌 호출은 제외해줘.",
        "branch.py",
        "numpy.array",
        ("numpy.array",),
        id="explicit-literal-spelling-also-retained",
    ),
])
def test_live_planner_distinguishes_symbol_identity_from_literal_spelling(
    prompt: str, filename: str, expected_symbol: str, expected_aspects: tuple[str, ...],
    live_planner_registry: LLMRegistry,
) -> None:
    selected = UploadFileInfo(
        file_id="selected-attachment", name=filename, size_bytes=128,
        content_hash="sha256:" + "a" * 64,
        source_uri="upload:///symbol-planning/selected-attachment",
    )
    other = UploadFileInfo(
        file_id="other-attachment", name="other.py", size_bytes=128,
        content_hash="sha256:" + "b" * 64,
        source_uri="upload:///symbol-planning/other-attachment",
    )
    attachments = (selected, other)
    state = build_graph_state_input(
        user_input=prompt, current_turn_id="symbol-request",
        user_turns=(UserTurnSnapshot(turn_id="symbol-request", text=prompt),),
        messages=[HumanMessage(id="symbol-request", content=prompt)],
        upload_files=attachments, retriever=_ManifestOnlyRetriever(attachments),
    )
    planner = make_planner_node(live_planner_registry.llm_planner, False)

    with capture_structured_calls() as calls:
        output = planner(state)["planner"].output

    provider_requirements = []
    for call in calls:
        if call.stage != "planner":
            continue
        parsed: object = call.output.get("parsed", call.output) if isinstance(call.output, dict) else call.output
        if isinstance(parsed, BaseModel):
            parsed = parsed.model_dump(mode="json")
        tasks = parsed.get("tasks", []) if isinstance(parsed, dict) else []
        provider_requirements.append({
            "attempt": call.attempt,
            "raw_text": call.output["raw"].content
            if isinstance(call.output, dict) and isinstance(call.output.get("raw"), AIMessage) else None,
            "requirements": [task.get("requirement") for task in tasks if isinstance(task, dict)]
            if isinstance(tasks, list) else None,
        })
    diagnostics = json.dumps({
        "query": prompt,
        "provider_requirements": provider_requirements,
        "final_requirements": [task.requirement.model_dump(mode="json") for task in output.tasks],
    }, ensure_ascii=False)
    print(diagnostics)

    assert output.use_retrieval, diagnostics
    assert len(output.tasks) == 1, diagnostics
    task = output.tasks[0]
    assert task.route == "upload", diagnostics
    assert task.requirement.file_ids == ["selected-attachment"], diagnostics
    assert task.requirement.match == "symbol", diagnostics
    assert task.requirement.symbols == [expected_symbol], diagnostics
    assert task.requirement.aspects == list(expected_aspects), diagnostics
