import json

from src.core.contracts import PlannerState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalRequirement, RetrievalTask
from src.runtime.nodes.synthesis.budgets import SynthesisBudgetProfile
from src.runtime.nodes.synthesis.context import build_synthesis_context, prepare_synthesis_inputs
from src.runtime.nodes.synthesis.evidence_selection import select_evidence_hits
from src.runtime.nodes.synthesis.prompt_builder import select_evidence_packet


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
