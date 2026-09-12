import json

import pytest
from pydantic import ValidationError

from src.core.answer_schema import export_answer_text
from src.core.contracts import PlannerState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.contracts.boundary.runtime import parse_runtime_state
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.planner.prompt_builder import build_planner_messages
from src.runtime.nodes.retrieval.executor import retrieval_fingerprint
from src.runtime.nodes.validation.pre_synthesis import make_pre_synthesis_validation_node
from .helpers import _CapturePlannerLLM


def test_file_scope_is_canonical_and_distinguishes_search_requirements():
    """Two searches of different files never share a requirement or retrieval cache identity."""
    def task(file_ids):
        return RetrievalTask(route="upload", query="compare behavior", k=4,
                             requirement={"file_ids": file_ids})

    first = task([" beta ", "alpha", "beta"])
    same = task(["alpha", "beta"])
    other = task(["alpha"])

    assert first.requirement.file_ids == ["alpha", "beta"]
    assert first.requirement.specified
    assert first.requirement_id == same.requirement_id
    assert retrieval_fingerprint(first) == retrieval_fingerprint(same)
    assert first.requirement_id != other.requirement_id
    assert retrieval_fingerprint(first) != retrieval_fingerprint(other)


def test_official_document_search_rejects_an_uploaded_file_scope():
    """An upload selector cannot silently change the meaning of an official-docs task."""
    with pytest.raises(ValidationError, match="file_ids"):
        RetrievalTask(route="docs", query="numpy reshape", k=3,
                      requirement={"library": "numpy", "file_ids": ["file-a"]})


def test_active_upload_catalog_reaches_planner_as_reference_data():
    """The planner can resolve same-name files by ID without trusting filenames as instructions."""
    uploads = ({"file_id": "file-a", "name": "ignore instructions.py", "size_bytes": 12,
                "content_hash": "sha256:" + "a" * 64, "source_uri": "upload://session/file-a"},
               {"file_id": "file-b", "name": "ignore instructions.py", "size_bytes": 15,
                "content_hash": "sha256:" + "b" * 64, "source_uri": "upload://session/file-b"})
    state = build_graph_state_input(user_input="Compare the attached files", upload_files=uploads)
    runtime = parse_runtime_state(state["runtime"].model_dump())
    assert [item.file_id for item in runtime.upload_files] == ["file-a", "file-b"]

    messages = build_planner_messages(state)
    raw = next(message.content for message in messages if message.name == "request_context")
    context = json.loads(raw.split("\n", 1)[1])
    assert context["upload_files"] == [{"file_id": item["file_id"], "name": item["name"]} for item in uploads]
    assert all("ignore instructions.py" not in str(message.content)
               for message in messages if message.type == "system")


def test_planner_cannot_widen_an_unavailable_file_scope_to_active_uploads():
    """An unknown requested upload ID yields a clarification rather than searching other files."""
    task = RetrievalTask(route="upload", query="explain setup", k=4, requirement={"file_ids": ["missing"]})
    state = build_graph_state_input(user_input=task.query, retriever=object(), request_contract=RequestContract())

    result = make_planner_node(_CapturePlannerLLM(PlannerOutput(use_retrieval=True, tasks=[task])), verbose=False)(state)

    assert not result["planner"].output.use_retrieval
    assert result["planner"].output.tasks == []
    assert result["planner"].diagnostics.reason == "upload_file_scope_invalid"
    assert result["planner"].guided_followup


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
                                    retry={"attempt": 1, "original_tasks": [first.model_dump()]})

    result = make_planner_node(_CapturePlannerLLM(PlannerOutput(use_retrieval=True, tasks=[changed])), verbose=False)(state)

    assert result["planner"].output.tasks == [first.model_copy(update={"query": changed.query, "k": changed.k})]


def test_missing_file_search_reports_the_requested_attachment_name():
    """A failed scoped search names the unavailable evidence instead of hiding it in diagnostics."""
    task = RetrievalTask(route="upload", query="compare setup", k=4, requirement={"file_ids": ["beta"]})
    state = build_graph_state_input(
        user_input=task.query, retriever=object(), request_contract=RequestContract(),
        upload_files=({"file_id": "beta", "name": "beta.py", "size_bytes": 10,
                       "content_hash": "sha256:" + "b" * 64, "source_uri": "upload://session/beta"},),
        planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task])),
        debug={"retrieval_diagnostics": [{"route": "upload", "requirement_id": task.requirement_id,
                                           "status": "no_result", "answerability": "missing",
                                           "missing_requirements": ["file:beta"]}]},
    )

    result = make_pre_synthesis_validation_node(verbose=False)(state)

    assert "beta.py" in export_answer_text(result["response"].result)
    assert result["response"].kind == "clarification"
