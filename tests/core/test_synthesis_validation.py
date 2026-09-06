import json

from langchain_core.messages import AIMessage, HumanMessage

from src.core.answer_schema import export_answer_text, iter_content_units, text_document
from src.core.contracts import PlannerState, RetrievalState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.runtime.nodes.synthesis import make_synthesize_node


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
    return build_graph_state_input(
        user_input=query, messages=[HumanMessage(content=query)],
        planner=PlannerState(output=PlannerOutput(use_retrieval=bool(routes), tasks=[RetrievalTask(route=route, query=query, k=4) for route in routes])),
        retrieval=RetrievalState(hit_log=[hit.model_dump(mode="json") for hit in hits]),
    )


class ModelBoundary:
    def __init__(self, *, error=None, malformed=None):
        self.error = error
        self.malformed = malformed
        self.packet = []

    def with_structured_output(self, *args, **kwargs):
        return self

    def invoke(self, messages):
        packet_text = str(messages[-1].content)
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


def test_synthesis_displays_and_checks_the_same_units():
    """The model's sole body is preserved through the result, validation checks, and export."""
    hit = _hit()
    result = make_synthesize_node(ModelBoundary())(_state([hit]))["response"].result
    units = list(iter_content_units(result.content))
    assert len(units) == 1
    assert units[0][1].text == hit.evidence.excerpt
    assert [check.unit_id for check in result.checks] == [path for path, _ in units]
    assert result.checks[0].support_status == "exact_match"
    assert export_answer_text(result).count(hit.evidence.excerpt) == 1
    assert result.citations[0].evidence == hit.evidence
    assert set(result.model_dump()) == {"content", "citations", "checks", "issues", "actions", "content_hash", "retrieval_required"}


def test_only_the_packet_provided_to_the_model_is_accepted_for_citations():
    """Prompt truncation changes the allowed reference rather than silently authorizing unseen text."""
    hit = _hit("First sentence.\nSecond sentence with further detail.")
    model = ModelBoundary()
    response = make_synthesize_node(model, prompt_snippet_char_limit=16)(_state([hit]))["response"]
    assert [item.id for item in response.evidence_packet] == [item["id"] for item in model.packet]
    assert response.evidence_packet[0].id != hit.evidence.id
    assert response.result.citations[0].evidence.excerpt == hit.evidence.element.text[:16]
    assert response.result.citations[0].evidence.selection.end == 16


def test_malformed_generation_becomes_explicit_original_source_fallback():
    """Invalid model text is never promoted to an answer, while original evidence stays available."""
    hit = _hit()
    updates = make_synthesize_node(ModelBoundary(malformed={"answer": "UNVALIDATED WRONG ANSWER"}))(_state([hit]))
    result = updates["response"].result
    assert "UNVALIDATED WRONG ANSWER" not in export_answer_text(result)
    assert hit.evidence.excerpt in export_answer_text(result)
    assert any(unit.basis == "excerpt" for _, unit in iter_content_units(result.content))
    assert updates["debug"].synthesis_errors
    assert all(check.support_status != "unsupported" for check in result.checks)


def test_timeout_uses_compact_model_and_its_actual_source_ranges():
    """A successful compact retry retains only the evidence ranges provided to that attempt."""
    hit = _hit("source detail " * 300)
    compact = ModelBoundary()
    updates = make_synthesize_node(ModelBoundary(error=TimeoutError("structured timeout")), compact)(_state([hit]))
    response = updates["response"]
    assert response.evidence_packet[0].excerpt == compact.packet[0]["excerpt"]
    assert response.evidence_packet[0].id == compact.packet[0]["id"]
    assert len(response.evidence_packet[0].excerpt) == 900
    assert response.result.citations[0].evidence.id == compact.packet[0]["id"]
    assert "SYNTHESIS_TIMEOUT" in updates["debug"].error_codes


def test_exhausted_timeout_fallback_keeps_source_version_and_location():
    """Both model failures still leave an honest excerpt tied to the immutable source revision."""
    hit = _hit("retries = 3\n", source="upload")
    updates = make_synthesize_node(
        ModelBoundary(error=TimeoutError("timeout")), ModelBoundary(error=TimeoutError("timeout")),
    )(_state([hit]))
    evidence = updates["response"].result.citations[0].evidence
    assert evidence.snapshot == hit.evidence.snapshot
    assert evidence.element.anchors == hit.evidence.element.anchors
    assert evidence.excerpt == "retries = 3\n"


def test_exact_upload_extraction_needs_no_model_generation():
    """Explicit source extraction uses the same checked document contract without rewriting code."""
    hit = _hit("retries = 3\n", source="upload")
    updates = make_synthesize_node(ModelBoundary(error=AssertionError("model should not run")))(_state([hit], query="retries 코드를 원문 그대로 발췌해줘"))
    assert updates["debug"].synthesis_errors == []
    assert any(check.support_status == "exact_match" for check in updates["response"].result.checks)
    assert updates["response"].result.citations[0].evidence.excerpt == hit.evidence.excerpt


def test_synthesis_draft_is_not_added_to_conversation_before_validation():
    """Unvalidated content cannot survive a later repair as a stale assistant message."""
    updates = make_synthesize_node(ModelBoundary())(_state([_hit()]))
    assert "messages" not in updates
