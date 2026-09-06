import importlib

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.core.contracts import RetrievalDiagnostic
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.prompts import SYS_POLICY
from src.runtime.nodes.retrieval import collect_retrieval_result
from src.runtime.nodes.synthesis.evidence_selection import select_evidence_hits
from src.runtime.nodes.synthesis.prompt_builder import build_synthesis_messages
from src.runtime.nodes.validation.evidence_validator import assess_retrieval_quality, build_validation_snapshot


def _hit(text, *, source="official", uri="https://numpy.org/doc/stable/", rank=1, metadata=None):
    snapshot = build_snapshot(
        source_uri=uri, title="Source", media_type="text/plain", source_type=source,
        content=text, parser="test", parser_version="1",
    )
    element = DocumentElement(element_id="source", kind="paragraph" if source == "official" else "code", text=text, metadata=metadata or {})
    return SearchHit(evidence=build_evidence(snapshot=snapshot, element=element), rank=rank,
                     score=RetrievalScore(metric="test", raw=0.01, normalized=0.01, direction="higher"))


def test_synthesis_package_reexports_models_and_factory():
    synthesis = importlib.import_module("src.runtime.nodes.synthesis")
    models = importlib.import_module("src.runtime.nodes.synthesis.models")
    node = importlib.import_module("src.runtime.nodes.synthesis.node")
    assert synthesis.PreparedSynthesisInputs is models.PreparedSynthesisInputs
    assert synthesis.SynthesisPipelineResult is models.SynthesisPipelineResult
    assert synthesis.make_synthesize_node is node.make_synthesize_node
    assert not hasattr(synthesis, "build_structured_synthesizer")


def test_online_runner_package_reexports_case_runner_api():
    runner = importlib.import_module("src.eval.online_runner")
    cases = importlib.import_module("src.eval.online_runner.case_runner")
    assert runner.run_online_benchmark is cases.run_online_benchmark
    assert runner._run_single_case is cases._run_single_case


def test_reporting_package_reexports_public_helpers():
    reporting = importlib.import_module("src.eval.reporting")
    assert reporting.build_markdown_report is importlib.import_module("src.eval.reporting.markdown").build_markdown_report
    assert reporting.build_summary is importlib.import_module("src.eval.reporting.summary").build_summary
    assert reporting.write_run_outputs is importlib.import_module("src.eval.reporting.writer").write_run_outputs


def test_validation_package_exposes_only_public_entrypoints():
    validation = importlib.import_module("src.runtime.nodes.validation")
    for name in ("ValidationAssessment", "ValidationSnapshot", "make_pre_synthesis_validation_node", "make_post_synthesis_validation_node"):
        assert hasattr(validation, name)
    assert not hasattr(validation, "apply_validation_outcome")


def test_prompt_preserves_current_question_and_contract_while_trimming_history():
    """History trimming removes old turns and tool messages while retaining source and question contracts."""
    state = build_graph_state_input(user_input="u4", memory_summary="older summary", messages=[
        HumanMessage(content="u1"), AIMessage(content="a1"),
        ToolMessage(content="tool output", name="tavily_search", tool_call_id="1"),
        HumanMessage(content="u2"), AIMessage(content="a2"),
        HumanMessage(content="u3"), AIMessage(content="a3"), HumanMessage(content="u4"),
    ])
    evidence = _hit("Original evidence").evidence
    messages, before, after = build_synthesis_messages(
        state=state, action_rules=["Save the current content."], evidence_packet=[evidence],
        attempt=2, max_turns=2,
    )
    assert (before, after) == (7, 5)
    assert messages[0] == SystemMessage(content=SYS_POLICY)
    assert not any(isinstance(message, ToolMessage) for message in messages)
    assert not any(message.content == "u1" for message in messages)
    assert any(message.content == "u4" for message in messages)
    assert evidence.id in str(messages[-1].content)
    assert any("Repair the displayed content and its references together" in str(message.content) for message in messages)
    assert any("older summary" in str(message.content) and isinstance(message, AIMessage) for message in messages)


def test_selection_preserves_route_coverage_and_additional_source_ranges():
    """Hybrid evidence includes each requested source before adding other useful original ranges."""
    docs = _hit("train_test_split divides the data")
    upload1 = _hit("train_test_split(X, test_size=0.2)", source="upload", uri="uploads/a.py")
    upload2 = _hit("train_test_split(y, random_state=42)", source="upload", uri="uploads/b.py", rank=2)
    plan = PlannerOutput(use_retrieval=True, tasks=[
        RetrievalTask(route="docs", query="train_test_split", k=3),
        RetrievalTask(route="upload", query="train_test_split", k=3),
    ])
    selected = select_evidence_hits(user_input="비교해줘", hits=[upload2, docs, upload1], planner_output=plan)
    assert [item.evidence.route for item in selected[:2]] == ["docs", "upload"]
    assert {item.evidence.id for item in selected} == {docs.evidence.id, upload1.evidence.id, upload2.evidence.id}


def test_upload_selection_uses_code_metadata_without_converting_score_to_confidence():
    """Exact option metadata can rank a relevant cell ahead of a higher-ranked unrelated cell."""
    histogram = _hit("plt.hist(values, bins=20)", source="upload", uri="uploads/a.py", rank=1)
    pie = _hit("chart formatting", source="upload", uri="uploads/b.py", rank=9,
               metadata={"code_metadata": {"option_literals": ["startangle=90", "autopct='%1.1f%%'"]}})
    plan = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="upload", query="startangle autopct options", k=3)])
    selected = select_evidence_hits(user_input="설명해줘", hits=[histogram, pie], planner_output=plan)
    assert selected[0] == pie
    assert selected[1] == histogram
    assert "confidence" not in selected[0].evidence.model_dump()


def test_retrieval_quality_does_not_reject_available_sources_by_low_score_alone():
    """A low search rank score remains a retrieval signal and does not certify or reject an answer."""
    hits = [_hit("official values"), _hit("local values", source="upload", uri="uploads/a.py")]
    plan = PlannerOutput(use_retrieval=True, tasks=[
        RetrievalTask(route="docs", query="values", k=3), RetrievalTask(route="upload", query="values", k=3),
    ])
    snapshot = build_validation_snapshot(
        user_input="Compare values", planner_output=plan, parsed_hits=hits,
        current_attempt_retrieval_errors=[], current_attempt_retrieval_diagnostics=[
            RetrievalDiagnostic(tool="tavily_search", route="docs", status="success", query="values", normalized_score=0.01),
            RetrievalDiagnostic(tool="upload_search", route="upload", status="success", query="values", normalized_score=0.01),
        ], response_result=None, evidence_packet=[],
    )
    assessment = assess_retrieval_quality(snapshot)
    assert assessment.retry_reason is None
    assert assessment.failed_routes == set()
    assert assessment.blocked_missing_upload is False


def test_collect_retrieval_result_filters_cross_library_domains_without_losing_snapshot():
    """Library-scoped search retains the correct original source object and records domain filtering."""
    numpy = _hit("Join a sequence of arrays.", uri="https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html")
    pandas = _hit("Concatenate pandas objects.", uri="https://pandas.pydata.org/docs/reference/api/pandas.concat.html")
    errors = []
    hits, diagnostic = collect_retrieval_result(
        raw_payload={"hits": [numpy.model_dump(mode="json"), pandas.model_dump(mode="json")],
                     "diagnostics": {"status": "success", "query": "numpy official docs"}},
        tool_name="tavily_search", route="docs", query="numpy official docs", attempt=1, local_errors=errors,
    )
    assert [SearchHit.model_validate(hit) for hit in hits] == [numpy]
    assert errors == []
    assert diagnostic.filtered_cross_domain_count == 1
    assert diagnostic.final_evidence_count == 1
    assert "cross_library_domain_filtered" in diagnostic.warnings
