import json


from langchain_core.messages import AIMessage, HumanMessage

from src.core.answer_schema import export_answer_text, iter_content_units, text_document
from src.core.contracts import PlannerState, RetrievalState, DebugState, RetrievalDiagnostic
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import ExtractBody, RequestContract
from src.runtime.nodes.synthesis import make_synthesize_node
from tests.synthesis_fixtures import synthesis_excerpt_limits


def _hit(text="Default mode is safe.", *, source="official", rank=1):
    snapshot = build_snapshot(
        source_uri="https://docs.example.com/settings" if source == "official" else "uploads/example.py",
        title="Settings", media_type="text/plain", source_type=source,
        content=text, parser="test", parser_version="1",
    )
    element = DocumentElement(
        element_id="settings", kind="paragraph" if source == "official" else "code", text=text,
        anchors=[SourceAnchor(kind="web" if source == "official" else "code", start=0, end=len(text), precision="exact")],
    )
    return SearchHit(evidence=build_evidence(snapshot=snapshot, element=element), rank=rank,
                     score=RetrievalScore(metric="test", raw=0.01, normalized=0.01, direction="higher"))


def _state(hits, query="Explain the setting"):
    routes = list(dict.fromkeys(hit.evidence.route for hit in hits))
    tasks = [RetrievalTask(route=route, query=query, k=4) for route in routes]
    task_ids = {task.route: task.requirement_id for task in tasks}
    hits = [hit if hit.requirement_id else hit.model_copy(update={"requirement_id": task_ids[hit.evidence.route]}) for hit in hits]
    return build_graph_state_input(
        user_input=query, messages=[HumanMessage(content=query)],
        request_contract=RequestContract(),
        planner=PlannerState(output=PlannerOutput(use_retrieval=bool(routes), tasks=tasks)),
        retrieval=RetrievalState(hit_log=[hit.model_dump(mode="json") for hit in hits]),
    )


class ModelBoundary:
    def __init__(self, *, error=None, malformed=None):
        self.error = error
        self.malformed = malformed
        self.packet = []

    def with_structured_output(self, *args, **kwargs):
        return self

    def invoke(self, messages, *, timeout=None):
        packet_text = next(str(message.content) for message in messages if str(message.content).startswith("[Evidence Packet]"))
        self.packet = json.loads(packet_text[packet_text.index("[", len("[Evidence Packet]")):])
        if self.error:
            raise self.error
        if self.malformed is not None:
            return self.malformed
        if self.packet:
            document = text_document(self.packet[0]["excerpt"], basis="excerpt", refs=[self.packet[0]["id"]])
        else:
            document = text_document("현재 답변입니다.")
        return {"parsed": document.model_dump(mode="json"), "raw": AIMessage(content=""), "parsing_error": None}


def test_malformed_generation_becomes_explicit_original_source_fallback():
    """Invalid model text is never promoted to an answer, while original evidence stays available."""
    hit = _hit()
    updates = make_synthesize_node(ModelBoundary(malformed={"answer": "UNVALIDATED WRONG ANSWER"}), excerpt_limits=synthesis_excerpt_limits())(_state([hit]))
    result = updates["response"].result
    assert "UNVALIDATED WRONG ANSWER" not in export_answer_text(result)
    assert hit.evidence.excerpt in export_answer_text(result)
    assert any(unit.basis == "excerpt" for _, unit in iter_content_units(result.content))
    assert updates["debug"].synthesis_errors
    assert all(check.support_status != "unsupported" for check in result.checks)


def test_exact_upload_extraction_needs_no_model_generation():
    """Explicit source extraction uses the same checked document contract without rewriting code."""
    hit = _hit("retries = 3\n", source="upload")
    state = _state([hit], query="retries 코드를 원문 그대로 발췌해줘")
    state["runtime"] = state["runtime"].model_copy(update={
        "request_contract": RequestContract(body=ExtractBody()),
    })
    task = state["planner"].output.tasks[0]
    state["retrieval"] = RetrievalState(hit_log=[hit.model_copy(update={"requirement_id": task.requirement_id}).model_dump(mode="json")])
    state["debug"] = DebugState(retrieval_diagnostics=[RetrievalDiagnostic(
        route="upload", requirement_id=task.requirement_id, status="success", answerability="covered", evidence_count=1,
    )])
    updates = make_synthesize_node(ModelBoundary(error=AssertionError("model should not run")), excerpt_limits=synthesis_excerpt_limits())(state)
    assert updates["debug"].synthesis_errors == []
    assert any(check.support_status == "exact_match" for check in updates["response"].result.checks)
    assert updates["response"].result.citations[0].evidence.excerpt == hit.evidence.excerpt
    assert updates["response"].normal_evidence_missing_requirement_ids == []
    assert updates["response"].result.retrieval_required
    assert updates["response"].result.issues == []
