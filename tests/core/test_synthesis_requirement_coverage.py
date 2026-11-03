import json


from src.core.contracts import PlannerState, ResponseState, RetrievalState, RetryState
from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.runtime.nodes.synthesis.context import build_synthesis_context, prepare_synthesis_inputs
from src.runtime.nodes.validation.assessment import assess_validation
from src.runtime.nodes.validation.node import make_post_synthesis_validation_node
from tests.core.helpers import make_decisions_client
from src.runtime.nodes.validation.snapshot import build_validation_snapshot
from tests.synthesis_fixtures import prepare_synthesis_state, synthesis_excerpt_limits


def _prepare(tasks, hits, *, snippet=960):
    plan = PlannerOutput(use_retrieval=True, tasks=tasks)
    state = build_graph_state_input(
        user_input="Compare the requested settings", planner=PlannerState(output=plan),
        request_contract=RequestContract(),
        retrieval=RetrievalState(hit_log=hits),
    )
    prepare_synthesis_state(state, limits=synthesis_excerpt_limits(normal_chars=snippet))
    context = build_synthesis_context(state=state)
    prepared = prepare_synthesis_inputs(
        state=state, context=context,
        max_turns=6,
    )
    raw = str(prepared.model_messages[-1].content)
    packet = json.loads(raw[raw.index("[", len("[Evidence Packet]")):])
    return prepared, packet


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


def test_final_comparison_citations_must_cover_each_requested_file():
    """A comparison cannot omit a requested file from its final citations."""
    task = RetrievalTask(route="upload", query="compare the setup", k=4, requirement={"file_ids": ["alpha", "beta"]})
    hits = [_upload_hit(task, "alpha", "setup = 1\n"), _upload_hit(task, "beta", "setup = 2\n")]
    packet = [hit.evidence for hit in hits]
    result = finalize_answer(text_document("setup differs", basis="source", refs=[
        item.id for item in packet if item.element.metadata["file_id"] == "alpha"
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

    assert assessment.retry_reason == "missing_route_coverage"
    assert assessment.failed_requirement_ids == {task.requirement_id}


def _prompt_requirements(prepared):
    raw = next(str(message.content) for message in prepared.model_messages
               if str(message.content).startswith("[Retrieval Requirements]"))
    return json.loads(raw[raw.index("\n[") + 1:])


def test_each_file_must_retain_its_own_literal_code_anchors_in_prompt_and_citations():
    """Concrete code anchors from one file cannot conceal absence in another requested file."""
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

    requirements = _prompt_requirements(prepared)[0]
    assert requirements["requirement"]["aspects"] == ["setup", "cleanup"]
    assert requirements["coverage"]["missing_file_ids"] == []
    assert requirements["coverage"]["missing_aspects_by_file"] == {"alpha": [], "beta": ["cleanup"]}
    assert assess_validation(snapshot).failed_requirement_ids == {task.requirement_id}


def test_failed_comparison_does_not_publish_new_excerpts_from_requested_or_other_files():
    """A failed comparison cannot substitute a new source excerpt for its rejected answer."""
    task = RetrievalTask(route="upload", query="compare alpha and beta", k=2,
                         requirement={"file_ids": ["alpha", "beta"]})
    requested = _upload_hit(task, "alpha", "setup = 1\n")
    excluded = _upload_hit(task, "gamma", "excluded_value = 99\n")
    hits = [requested, excluded]
    packet = [requested.evidence]
    contract = RequestContract()
    state = build_graph_state_input(
        user_input=task.query, request_contract=contract,
        planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task])),
        retrieval=RetrievalState(hit_log=hits),
        response=ResponseState(
            result=finalize_answer(text_document("The comparison is incomplete."), packet),
            evidence_packet=packet,
            evidence_requirement_map={item.id: [task.requirement_id] for item in packet},
            request_id=contract.request_id, contract_revision=contract.revision,
        ),
        retry=RetryState(max_retries=0, max_content_repairs=0),
    )

    response = make_post_synthesis_validation_node(False, decisions_client=make_decisions_client())(state)["response"]

    assert response.kind == "failure"
    assert response.result.citations == []
    assert response.evidence_packet == []
    assert all(hit.evidence.excerpt not in response.result.content.blocks[0].content[0].text for hit in hits)
