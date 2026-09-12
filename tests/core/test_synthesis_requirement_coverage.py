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
    context = build_synthesis_context(state=state, has_default_slack_destination=False)
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


def test_tight_prompt_budget_keeps_evidence_for_each_requested_library():
    """The first docs source cannot consume the budget reserved for another comparison target."""
    alpha, beta = _task("alpha", "rows"), _task("beta", "columns")
    alpha_hit = _hit(alpha, "alpha rows are combined. " * 35)
    beta_hit = _hit(beta, "beta columns are combined. " * 35)

    prepared, packet = _prepare([alpha, beta], [alpha_hit, beta_hit], budget=600, snippet=960, max_items=2)

    assert len(packet) == 2
    assert sum(len(item.excerpt) for item in prepared.evidence_packet) <= 600
    assert {requirement for item in packet for requirement in item["requirement_ids"]} == {alpha.requirement_id, beta.requirement_id}


def test_requirement_aspect_selects_its_passage_and_reaches_the_model():
    """Structured aspects survive retrieval into the selected passage and model requirements."""
    task = _task("numpy", "order")
    explanation = "**order**\nC visits the last axis first. F visits the first axis first. A follows the input layout."
    hit = _hit(task, ("Navigation and release notes. " * 15 + "\n\n") * 8 + explanation)

    prepared, packet = _prepare([task], [hit])

    assert explanation in packet[0]["excerpt"]
    assert packet[0]["requirement_ids"] == [task.requirement_id]
    requirement_message = next(str(message.content) for message in prepared.model_messages if str(message.content).startswith("[Retrieval Requirements]"))
    requirements = json.loads(requirement_message[requirement_message.index("\n[") + 1:])
    assert requirements == [{
        "id": task.requirement_id, "route": "docs", "query": task.query,
        "requirement": task.requirement.model_dump(mode="json"),
        "coverage": {"evidence_ids": [packet[0]["id"]], "present_aspects": ["order"], "missing_aspects": [], "is_partial": False},
    }]


def test_shared_document_retains_separate_passages_for_separate_requirements():
    """Different aspects of one document remain searchable as separate source ranges."""
    order, copy = _task("numpy", "order"), _task("numpy", "copy")
    order_text = "**order**\nControls axis traversal."
    copy_text = "**copy**\nControls allocation of an independent array."
    text = order_text + "\n\n" + ("Unrelated notes. " * 80) + "\n\n" + copy_text
    order_hit = _hit(order, text)
    copy_hit = order_hit.model_copy(update={"requirement_id": copy.requirement_id})

    _, packet = _prepare([order, copy], [order_hit, copy_hit], snippet=120, max_items=2)

    assert len(packet) == 2
    assert any(order_text in item["excerpt"] for item in packet)
    assert any(copy_text in item["excerpt"] for item in packet)


def test_distant_aspects_of_one_requirement_use_separate_bounded_passages():
    """A multi-aspect requirement can retain both requested parameter explanations within the budget."""
    task = RetrievalTask(
        route="docs", query="numpy order copy", k=3,
        requirement=RetrievalRequirement(library="numpy", aspects=["order", "copy"]),
    )
    order_text = "**order**\nControls axis traversal."
    copy_text = "**copy**\nControls allocation of an independent array."
    hit = _hit(task, order_text + "\n\n" + ("Unrelated notes. " * 80) + "\n\n" + copy_text)

    prepared, packet = _prepare([task], [hit], snippet=120, max_items=2)

    assert len(packet) == 2
    assert any(order_text in item["excerpt"] for item in packet)
    assert any(copy_text in item["excerpt"] for item in packet)
    assert sum(len(item.excerpt) for item in prepared.evidence_packet) <= 240


def test_one_complete_source_can_support_multiple_requirements_without_duplicate_tokens():
    """A shared complete source retains every search association even when only one item fits."""
    order, copy = _task("numpy", "order"), _task("numpy", "copy")
    hit = _hit(order, "order controls traversal. copy controls allocation.")

    packet, requirement_ids = select_evidence_packet(
        [hit.evidence], max_items=1, snippet_char_limit=960, evidence_char_budget=len(hit.evidence.excerpt),
        requirements_by_evidence={hit.evidence.id: [order, copy]},
    )

    assert packet == [hit.evidence]
    assert requirement_ids == {hit.evidence.id: [order.requirement_id, copy.requirement_id]}


def test_item_budget_exposes_an_omitted_aspect_in_the_model_requirements():
    """A requirement ID never hides an aspect missing from the actual bounded model input."""
    task = RetrievalTask(
        route="docs", query="numpy order copy", k=3,
        requirement=RetrievalRequirement(library="numpy", aspects=["order", "copy"]),
    )
    text = (
        "**order**\nControls traversal of a copyable view.\n\n"
        + "Unrelated notes. " * 80 + "\n\n**copy**\nControls allocation."
    )
    prepared, packet = _prepare([task], [_hit(task, text)], snippet=120, max_items=1)
    raw = next(str(message.content) for message in prepared.model_messages if str(message.content).startswith("[Retrieval Requirements]"))
    requirements = json.loads(raw[raw.index("\n[") + 1:])

    assert len(packet) == 1
    assert "**copy**" not in packet[0]["excerpt"]
    assert requirements[0]["coverage"] == {
        "evidence_ids": [packet[0]["id"]], "present_aspects": ["order"],
        "missing_aspects": ["copy"], "is_partial": True,
    }
    assert prepared.evidence_requirement_map == {prepared.reference_aliases[packet[0]["id"]]: [task.requirement_id]}


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


def test_ten_file_comparison_preserves_sources_without_increasing_text_budget():
    """The task limit does not prevent ten-file comparisons from retaining one passage per file."""
    file_ids = [f"file-{index}" for index in range(10)]
    task = RetrievalTask(route="upload", query="compare the setup", k=4, requirement={"file_ids": file_ids})
    plan = PlannerOutput(use_retrieval=True, tasks=[task])
    profile = resolve_synthesis_budget_profile(user_input=task.query, planner_output=plan, snippet_char_limit=960)
    hits = [_upload_hit(task, file_id, "setup = 123\n" * 100, rank=index + 1) for index, file_id in enumerate(file_ids)]

    prepared, _ = _prepare([task], hits, budget=600, snippet=960, max_items=profile.max_evidence_items)

    assert len(prepared.evidence_packet) == 10
    assert {item.element.metadata["file_id"] for item in prepared.evidence_packet} == set(file_ids)
    assert sum(len(item.excerpt) for item in prepared.evidence_packet) <= 600
    assert profile.evidence_chars == 6000


def test_missing_comparison_file_is_visible_in_model_coverage():
    """Evidence from one file cannot conceal another requested file that was not retrieved."""
    task = RetrievalTask(route="upload", query="compare the setup", k=4, requirement={"file_ids": ["alpha", "beta"]})
    prepared, _ = _prepare([task], [_upload_hit(task, "alpha", "setup = 1\n")])
    raw = next(str(message.content) for message in prepared.model_messages if str(message.content).startswith("[Retrieval Requirements]"))
    requirements = json.loads(raw[raw.index("\n[") + 1:])

    assert requirements[0]["coverage"]["requested_file_ids"] == ["alpha", "beta"]
    assert requirements[0]["coverage"]["covered_file_ids"] == ["alpha"]
    assert requirements[0]["coverage"]["missing_file_ids"] == ["beta"]
    assert requirements[0]["coverage"]["is_partial"]


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


def test_repeated_file_sets_preserve_each_independent_requirement_in_the_default_budget():
    """Independent symbols in the same files retain every requirement/file pair through validation."""
    file_ids = [f"file-{index}" for index in range(5)]
    tasks = [RetrievalTask(route="upload", query=f"compare {symbol}", k=5,
                           requirement={"file_ids": file_ids, "symbols": [symbol], "match": "definition"})
             for symbol in ("setup", "cleanup")]
    plan = PlannerOutput(use_retrieval=True, tasks=tasks)
    profile = resolve_synthesis_budget_profile(user_input="compare", planner_output=plan, snippet_char_limit=1800)
    hits = [_upload_hit(task, file_id, f"def {task.requirement.symbols[0]}():\n    return {index}\n")
            for task in tasks for index, file_id in enumerate(file_ids)]

    prepared, _ = _prepare(tasks, hits, max_items=profile.max_evidence_items,
                           budget=profile.evidence_chars, snippet=profile.snippet_chars)

    assert all(not record["coverage"]["is_partial"] for record in _prompt_requirements(prepared))
    result = finalize_answer(text_document("The implementations differ.", basis="inference",
                                         refs=[item.id for item in prepared.evidence_packet]),
                             prepared.evidence_packet, retrieval_required=True)
    contract = prepared.request_contract
    snapshot = build_validation_snapshot(
        user_input="compare", planner_output=plan, parsed_hits=hits,
        current_attempt_retrieval_errors=[], current_attempt_retrieval_diagnostics=[],
        response_result=result, evidence_packet=prepared.evidence_packet,
        evidence_requirement_map=prepared.evidence_requirement_map, request_contract=contract,
        response_request_id=contract.request_id, response_contract_revision=contract.revision,
    )
    assert assess_validation(snapshot).retry_reason is None


def test_required_aspects_precede_optional_neighboring_context_across_files():
    """A feasible set of short statements is preserved before unrelated source context consumes the budget."""
    file_ids = [f"file-{index}" for index in range(5)]
    task = RetrievalTask(route="upload", query="compare setup cleanup", k=5,
                         requirement={"file_ids": file_ids, "symbols": ["run"],
                                      "aspects": ["setup", "cleanup"], "match": "definition"})
    text = "def run():\n    setup()\n" + "    unrelated()\n" * 80 + "    cleanup()\n"
    hits = [_upload_hit(task, file_id, text) for file_id in file_ids]
    profile = resolve_synthesis_budget_profile(
        user_input=task.query, planner_output=PlannerOutput(use_retrieval=True, tasks=[task]), snippet_char_limit=1800)

    prepared, _ = _prepare([task], hits, max_items=profile.max_evidence_items, budget=400, snippet=1800)

    assert _prompt_requirements(prepared)[0]["coverage"]["missing_aspects_by_file"] == {
        file_id: [] for file_id in file_ids}
    assert sum(len(item.excerpt) for item in prepared.evidence_packet) <= 400
    assert all(item.excerpt == item.element.text[item.selection.start:item.selection.end]
               for item in prepared.evidence_packet)


def test_shared_shortened_range_preserves_both_requirement_associations_at_the_item_limit():
    """Sharing an already selected cropped range costs no additional item or text budget."""
    order, copy = _task("numpy", "order"), _task("numpy", "copy")
    passage = "**order copy**\norder and copy control arrays.\n"
    hit = _hit(order, passage + "\n" + "Unrelated notes. " * 100)
    shared = hit.model_copy(update={"requirement_id": copy.requirement_id})

    prepared, _ = _prepare([order, copy], [hit, shared], max_items=1, snippet=100,
                           budget=len(passage))

    assert len(prepared.evidence_packet) == 1
    assert prepared.evidence_packet[0].excerpt == passage
    assert set(prepared.evidence_requirement_map[prepared.evidence_packet[0].id]) == {
        order.requirement_id, copy.requirement_id}
    assert all(not record["coverage"]["is_partial"] for record in _prompt_requirements(prepared))


def test_shared_passage_expansion_stays_inside_each_associated_retrieved_range():
    """Optional context cannot move a shared reference beyond another requirement's retrieved bounds."""
    order, copy = _task("numpy", "order"), _task("numpy", "copy")
    passage = "**order copy**\norder and copy control arrays.\n"
    broad = _hit(order, passage + "\nAdditional context.\n")
    narrow = broad.model_copy(update={
        "requirement_id": copy.requirement_id,
        "evidence": build_evidence(snapshot=broad.evidence.snapshot, element=broad.evidence.element, end=len(passage)),
    })

    prepared, _ = _prepare([order, copy], [broad, narrow], max_items=1, snippet=100, budget=100)

    assert prepared.evidence_packet == [narrow.evidence]
    assert set(prepared.evidence_requirement_map[narrow.evidence.id]) == {order.requirement_id, copy.requirement_id}


def test_a_large_required_statement_uses_capacity_left_by_a_smaller_statement():
    """Unequal required statement sizes can share a feasible total budget after initial fair allocation."""
    task = RetrievalTask(route="upload", query="alpha beta", k=1,
                         requirement={"symbols": ["run"], "aspects": ["alpha", "beta"]})
    alpha = "    alpha(" + "x" * 80 + ")\n"
    beta = "    beta()\n"
    hit = _upload_hit(task, "file", "def run():\n" + alpha + "    unrelated()\n" * 80 + beta)

    prepared, _ = _prepare([task], [hit], max_items=8, snippet=1800, budget=120)

    assert not _prompt_requirements(prepared)[0]["coverage"]["is_partial"]
    assert any(alpha in item.excerpt for item in prepared.evidence_packet)
    assert any(beta in item.excerpt for item in prepared.evidence_packet)
    assert sum(len(item.excerpt) for item in prepared.evidence_packet) <= 120


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


@pytest.mark.parametrize("clipped", [False, True])
def test_partial_code_reserves_each_file_and_aspect_before_optional_context(clipped):
    """Every file and aspect retains a partial anchor when whole lines exceed the shared budget."""
    files = ["alpha", "beta"]
    task = RetrievalTask(route="upload", query="compare setup cleanup", k=4,
                         requirement={"file_ids": files, "aspects": ["setup", "cleanup"]})
    hits = []
    for file_id in files:
        for aspect in task.requirement.aspects:
            hit = _upload_hit(task, file_id, f'{aspect} = "' + "context " * 80 + '"\n')
            if clipped:
                hit = hit.model_copy(update={"evidence": build_evidence(
                    snapshot=hit.evidence.snapshot, element=hit.evidence.element, end=500)})
            hits.append(hit)

    prepared, _ = _prepare([task], hits, max_items=4, snippet=1800, budget=160)

    coverage = _prompt_requirements(prepared)[0]["coverage"]
    assert coverage["missing_aspects_by_file"] == {file_id: [] for file_id in files}
    assert len(prepared.evidence_packet) == 4
    assert sum(len(item.excerpt) for item in prepared.evidence_packet) <= 160
    assert all(any(item.snapshot == hit.evidence.snapshot and item.element == hit.evidence.element
                   and hit.evidence.selection.start <= item.selection.start < item.selection.end <= hit.evidence.selection.end
                   for hit in hits) for item in prepared.evidence_packet)


def test_partial_code_does_not_turn_a_truncated_anchor_into_covered_evidence():
    """A budget too small for the literal anchor retains the requirement's missing status."""
    task = RetrievalTask(route="upload", query="Explain target_call", k=1,
                         requirement={"aspects": ["target_call"]})
    hit = _upload_hit(task, "alpha", 'target_call = "' + "context " * 100 + '"\n')

    prepared, _ = _prepare([task], [hit], max_items=1, snippet=4, budget=4)

    assert prepared.evidence_packet
    assert sum(len(item.excerpt) for item in prepared.evidence_packet) <= 4
    assert _prompt_requirements(prepared)[0]["coverage"]["missing_aspects"] == ["target_call"]


def test_whole_statement_redistribution_keeps_room_for_another_file():
    """When only one whole statement fits, both requested files retain partial evidence."""
    task = RetrievalTask(route="upload", query="compare setup", k=2,
                         requirement={"file_ids": ["alpha", "beta"], "aspects": ["setup"]})
    text = 'setup = "' + "x" * 108 + '"\n'
    hits = [_upload_hit(task, file_id, text) for file_id in task.requirement.file_ids]

    prepared, _ = _prepare([task], hits, max_items=2, snippet=1800, budget=len(text))

    assert _prompt_requirements(prepared)[0]["coverage"]["missing_aspects_by_file"] == {"alpha": [], "beta": []}
    assert len(prepared.evidence_packet) == 2
    assert sum(len(item.excerpt) for item in prepared.evidence_packet) <= len(text)


def test_whole_statement_redistribution_finishes_before_partial_fallback():
    """Later small selections release enough space to retain every feasible whole statement."""
    texts = {f"file-{index}": "setup = " + repr("x" * (length - 11)) + "\n"
             for index, length in enumerate([200, 190, 180, 170, 11])}
    task = RetrievalTask(route="upload", query="compare setup", k=5,
                         requirement={"file_ids": list(texts), "aspects": ["setup"]})
    hits = [_upload_hit(task, file_id, text) for file_id, text in texts.items()]

    prepared, _ = _prepare([task], hits, max_items=5, snippet=1800, budget=sum(map(len, texts.values())))

    assert {item.element.metadata["file_id"]: item.excerpt for item in prepared.evidence_packet} == texts
