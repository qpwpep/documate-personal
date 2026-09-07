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
        budget_profile=SynthesisBudgetProfile("docs", 900, snippet, budget, max_items),
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












