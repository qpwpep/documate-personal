import json

import pytest
from pydantic import ValidationError

from src.core.answer_schema import export_answer_text
from src.core.contracts import PlannerState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.contracts.boundary.runtime import parse_runtime_state
from src.core.planner_schema import PlannerOutput, RetrievalRewriteOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.planner.prompt_builder import build_planner_messages
from src.runtime.nodes.retrieval.executor import retrieval_fingerprint
from src.runtime.nodes.validation.pre_synthesis import make_pre_synthesis_validation_node
from .helpers import _CapturePlannerLLM








def test_planner_cannot_widen_an_unavailable_file_scope_to_active_uploads():
    """An unavailable file stays scoped through planning and execution without borrowing active files."""
    from types import SimpleNamespace
    from src.infra.tools.local_rag.tool import build_upload_search_tool
    from src.runtime.nodes.retrieval import make_retrieve_dispatch_node

    class ScopedVectorStore:
        def similarity_search_with_score(self, query, *, k, file_ids):
            assert file_ids == ["missing"]
            return []

    task = RetrievalTask(route="upload", query="explain setup", k=4, requirement={"file_ids": ["missing"]})
    retriever = SimpleNamespace(vectorstore=ScopedVectorStore())
    state = build_graph_state_input(user_input=task.query, retriever=retriever)

    state.update(make_planner_node(_CapturePlannerLLM(PlannerOutput(use_retrieval=True, tasks=[task])), False)(state))
    state.update(make_retrieve_dispatch_node(None, build_upload_search_tool(), False)(state))

    assert state["planner"].output.tasks == [task]
    assert state["planner"].guided_followup is None
    assert state["retrieval"].hit_log == []
    diagnostic = state["debug"].retrieval_diagnostics[0]
    assert diagnostic.status == "no_result"
    assert diagnostic.missing_requirements == ["file:missing"]


def test_retrieval_retry_retains_original_file_scope_when_model_changes_it():
    """Retry query rewrites cannot remove an originally requested comparison file."""
    first = RetrievalTask(route="upload", query="explain setup", k=4, requirement={"file_ids": ["alpha", "beta"]})
    changed = RetrievalTask(route="upload", query="setup usage", k=5,
                            requirement_id=first.requirement_id, requirement={"file_ids": ["alpha"]})
    uploads = tuple({"file_id": file_id, "name": f"{file_id}.py", "size_bytes": 10,
                     "content_hash": "sha256:" + "a" * 64, "source_uri": f"upload://session/{file_id}"}
                    for file_id in ["alpha", "beta"])
    state = build_graph_state_input(user_input=first.query, retriever=object(), upload_files=uploads,
                                    request_contract=RequestContract(),
                                    retry={"attempt": 1, "original_tasks": [first.model_dump()], "failed_requirement_ids": [first.requirement_id]})

    rewrites = RetrievalRewriteOutput(rewrites=[{"requirement_id": first.requirement_id, "query": changed.query, "k": changed.k}])
    result = make_planner_node(None, verbose=False, llm_planner_retry=_CapturePlannerLLM(rewrites))(state)

    assert result["planner"].output.tasks == [first.model_copy(update={"query": changed.query, "k": changed.k})]
