import json

import pytest

from src.core.contracts import PlannerState
from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalRequirement, RetrievalTask
from src.core.request_contracts import RequestContract
from src.runtime.nodes.synthesis.budgets import SynthesisBudgetProfile, resolve_synthesis_budget_profile
from src.runtime.nodes.synthesis.context import build_synthesis_context, prepare_synthesis_inputs
from src.runtime.nodes.synthesis.evidence_selection import select_evidence_hits
from src.runtime.nodes.synthesis.prompt_builder import select_evidence_packet
from src.runtime.nodes.validation.assessment import assess_validation
from src.runtime.nodes.validation.node import make_post_synthesis_validation_node
from src.runtime.nodes.validation.snapshot import build_validation_snapshot


def _task(library, aspect):
    return RetrievalTask(
        route="docs", query=f"{library} {aspect}", k=3,
        requirement=RetrievalRequirement(library=library, aspects=[aspect]),
    )


def _hit(task, text, *, rank=1, uri=None):
    snapshot = build_snapshot(
        source_uri=uri or f"https://example.com/{task.requirement.library}/{rank}",
        title=task.requirement.library, media_type="text/markdown", source_type="official",
        content=text, parser="test", parser_version="1",
    )
    return SearchHit(
        evidence=build_evidence(snapshot=snapshot, element=DocumentElement(element_id="body", kind="paragraph", text=text)),
        score=RetrievalScore(metric="provider_score", direction="higher", raw=0.9, normalized=0.9),
        rank=rank, requirement_id=task.requirement_id,
    )


def _prepare(tasks, hits, *, budget=6000, snippet=960, max_items=6):
    plan = PlannerOutput(use_retrieval=True, tasks=tasks)
    state = build_graph_state_input(
        user_input="Compare the requested settings", planner=PlannerState(output=plan),
        request_contract=RequestContract(),
        retrieval={"hit_log": [hit.model_dump(mode="json") for hit in hits]},
    )
    context = build_synthesis_context(state=state)
    prepared = prepare_synthesis_inputs(
        state=state, context=context,
        budget_profile=SynthesisBudgetProfile("docs", snippet, budget, max_items),
        max_turns=6, prompt_snippet_char_limit=snippet, prompt_evidence_char_budget=budget,
    )
    raw = str(prepared.model_messages[-1].content)
    packet = json.loads(raw[raw.index("[", len("[Evidence Packet]")):])
    return prepared, packet


def test_each_docs_requirement_precedes_additional_hits_from_the_same_source():
    """A comparison retains one candidate for every independent docs requirement before duplicates."""
    alpha, beta = _task("alpha", "rows"), _task("beta", "columns")
    alpha_first = _hit(alpha, "alpha rows are combined", rank=1)
    alpha_second = _hit(alpha, "alpha rows are preserved", rank=2)
    beta_hit = _hit(beta, "beta columns are combined", rank=9)
    plan = PlannerOutput(use_retrieval=True, tasks=[alpha, beta])

    selected = select_evidence_hits(
        user_input="Compare alpha and beta", hits=[alpha_first, alpha_second, beta_hit], planner_output=plan,
    )

    assert selected[:2] == [alpha_first, beta_hit]
    assert set(hit.evidence.id for hit in selected) == {alpha_first.evidence.id, alpha_second.evidence.id, beta_hit.evidence.id}














def _upload_hit(task, file_id, text, *, rank=1):
    snapshot = build_snapshot(
        source_uri=f"upload://session/{file_id}", title=f"{file_id}.py", media_type="text/x-python",
        source_type="upload", content=text, parser="test", parser_version="1",
    )
    return SearchHit(
        evidence=build_evidence(snapshot=snapshot, element=DocumentElement(
            element_id="python-source", kind="code", text=text, language="python", metadata={"file_id": file_id})),
        score=RetrievalScore(metric="l2", direction="lower", raw=0.1),
        rank=rank, requirement_id=task.requirement_id,
    )


def test_file_comparison_reserves_a_candidate_for_each_requested_file():
    """Extra matches in one upload cannot displace another requested comparison file."""
    task = RetrievalTask(route="upload", query="compare the setup", k=4, requirement={"file_ids": ["alpha", "beta"]})
    first = _upload_hit(task, "alpha", "setup = 1\n")
    extra = _upload_hit(task, "alpha", "setup = 2\n", rank=2)
    second = _upload_hit(task, "beta", "setup = 3\n", rank=9)
    unrelated = _upload_hit(task, "gamma", "setup = 4\n")

    selected = select_evidence_hits(user_input=task.query, hits=[first, extra, unrelated, second],
                                    planner_output=PlannerOutput(use_retrieval=True, tasks=[task]))

    assert selected[:2] == [first, second]
    assert unrelated not in selected






@pytest.mark.parametrize("cited_files, expected_failure", [(["alpha"], True), (["alpha", "beta"], False)])
def test_final_comparison_citations_must_cover_each_requested_file(cited_files, expected_failure):
    """A two-file comparison passes only when the final citations retain both requested files."""
    task = RetrievalTask(route="upload", query="compare the setup", k=4, requirement={"file_ids": ["alpha", "beta"]})
    hits = [_upload_hit(task, "alpha", "setup = 1\n"), _upload_hit(task, "beta", "setup = 2\n")]
    packet = [hit.evidence for hit in hits]
    result = finalize_answer(text_document("setup differs", basis="source", refs=[
        item.id for item in packet if item.element.metadata["file_id"] in cited_files
    ]), packet, retrieval_required=True)
    contract = RequestContract()
    snapshot = build_validation_snapshot(
        user_input=task.query, planner_output=PlannerOutput(use_retrieval=True, tasks=[task]), parsed_hits=hits,
        current_attempt_retrieval_errors=[], current_attempt_retrieval_diagnostics=[],
        response_result=result, evidence_packet=packet,
        evidence_requirement_map={item.id: [task.requirement_id] for item in packet},
        request_contract=contract, response_request_id=contract.request_id, response_contract_revision=contract.revision,
    )

    assessment = assess_validation(snapshot)

    assert assessment.retry_reason == ("missing_route_coverage" if expected_failure else None)
    assert assessment.failed_requirement_ids == ({task.requirement_id} if expected_failure else set())


def _prompt_requirements(prepared):
    raw = next(str(message.content) for message in prepared.model_messages
               if str(message.content).startswith("[Retrieval Requirements]"))
    return json.loads(raw[raw.index("\n[") + 1:])












def test_each_file_must_retain_its_own_aspects_in_the_prompt_and_final_citations():
    """An aspect from one file cannot conceal its absence in another requested file."""
    task = RetrievalTask(route="upload", query="compare setup cleanup", k=2,
                         requirement={"file_ids": ["alpha", "beta"], "aspects": ["setup", "cleanup"]})
    hits = [_upload_hit(task, "alpha", "setup()\ncleanup()\n"), _upload_hit(task, "beta", "setup()\n")]
    prepared, _ = _prepare([task], hits)
    result = finalize_answer(text_document("The files differ.", basis="inference",
                                         refs=[item.id for item in prepared.evidence_packet]),
                             prepared.evidence_packet, retrieval_required=True)
    contract = prepared.request_contract
    snapshot = build_validation_snapshot(
        user_input=task.query, planner_output=PlannerOutput(use_retrieval=True, tasks=[task]), parsed_hits=hits,
        current_attempt_retrieval_errors=[], current_attempt_retrieval_diagnostics=[], response_result=result,
        evidence_packet=prepared.evidence_packet, evidence_requirement_map=prepared.evidence_requirement_map,
        request_contract=contract, response_request_id=contract.request_id, response_contract_revision=contract.revision,
    )

    assert _prompt_requirements(prepared)[0]["coverage"]["missing_aspects_by_file"] == {
        "alpha": [], "beta": ["cleanup"]}
    assert assess_validation(snapshot).failed_requirement_ids == {task.requirement_id}


@pytest.mark.parametrize("include_requested_file", [False, True])
def test_failed_comparison_fallback_excludes_files_outside_the_requested_scope(include_requested_file):
    """A failed comparison exposes original excerpts only from the requested upload scope."""
    task = RetrievalTask(route="upload", query="compare alpha and beta", k=2,
                         requirement={"file_ids": ["alpha", "beta"]})
    requested = _upload_hit(task, "alpha", "setup = 1\n")
    excluded = _upload_hit(task, "gamma", "excluded_value = 99\n")
    hits = [requested, excluded] if include_requested_file else [excluded]
    packet = [requested.evidence] if include_requested_file else []
    contract = RequestContract()
    state = build_graph_state_input(
        user_input=task.query, request_contract=contract,
        planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task])),
        retrieval={"hit_log": [hit.model_dump(mode="json") for hit in hits]},
        response={
            "result": finalize_answer(text_document("The comparison is incomplete."), packet),
            "evidence_packet": packet,
            "evidence_requirement_map": {item.id: [task.requirement_id] for item in packet},
            "request_id": contract.request_id, "contract_revision": contract.revision,
        },
        retry={"max_retries": 0},
    )

    response = make_post_synthesis_validation_node(False)(state)["response"]

    assert response.kind == "failure"
    assert [citation.evidence.snapshot.title for citation in response.result.citations] == (
        ["alpha.py"] if include_requested_file else [])
    assert [item.snapshot.title for item in response.evidence_packet] == (
        ["alpha.py"] if include_requested_file else [])








