import unittest

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.core.conversation_memory import ConversationMemoryPolicy
from src.core.contracts import GraphState, LLMCallMetadata, PlannerState, ResponseState
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.runtime.graph_builder import _instrument_stage_node
from src.runtime.make_graph import build_graph
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.session import add_user_message
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from src.core.planner_schema import PlannerOutput, RetrievalTask

from .helpers import (
    _CapturePlannerLLM,
    _tool_payload,
)


def _official_hit(*, uri, title, excerpt, score):
    snapshot = build_snapshot(
        source_uri=uri, title=title, media_type="text/html", source_type="official",
        content=excerpt, parser="fixture", parser_version="1",
    )
    evidence = build_evidence(snapshot=snapshot, element=DocumentElement(element_id="body", kind="paragraph", text=excerpt))
    return SearchHit(evidence=evidence, score=RetrievalScore(metric="relevance", raw=score, direction="higher"), rank=1).model_dump(mode="json")


def _response(text, *, state=None, attempt=1):
    evidence = [SearchHit.model_validate(hit).evidence for hit in get_retrieval_state(state).hit_log] if state else []
    document = text_document(text, basis="source" if evidence else "interaction", refs=[item.id for item in evidence])
    return ResponseState(result=finalize_answer(document, evidence), evidence_packet=evidence, synthesis_attempt=attempt)


class GraphRoutingTest(unittest.TestCase):
    pass








if __name__ == "__main__":
    unittest.main()




def _assert_repair_flow(routes, defect, max_retries, persistent):
    import json

    from src.core.answer_schema import AnswerDocument
    from src.runtime.nodes.synthesis import make_synthesize_node
    from tests.core.test_answer_validation import _evidence, _hit

    hits = {
        route: _hit(_evidence(source_type="official" if route == "docs" else "upload")).model_dump(mode="json")
        for route in routes
    }
    searches = {"docs": 0, "upload": 0}
    packets = []

    def search(route, query):
        searches[route] += 1
        return _tool_payload(
            [hits[route]], tool="tavily_search" if route == "docs" else "upload_search",
            route=route, status="success", message="", query=query,
        )

    class RepairingLLM:
        def with_structured_output(self, *args, **kwargs):
            return self

        def invoke(self, messages):
            raw = next(str(message.content) for message in messages if str(message.content).startswith("[Evidence Packet]"))
            packet = json.loads(raw.split("\n", 2)[2])
            packets.append(packet)
            refs = [item["id"] for item in packet]
            broken = len(packets) == 1 or persistent
            if broken and defect == "references":
                refs = ["unknown-ref"]
            if broken and defect == "coverage":
                refs = refs[:1]
            if defect == "code" and not broken:
                document = {"blocks": [{
                    "type": "code", "language": "python",
                    "content": {"text": "retries = 5", "basis": "example", "refs": refs},
                }]}
            else:
                document = {"blocks": [{
                    "type": "paragraph",
                    "content": [{"text": "수정 전 본문" if broken else "확인한 설명", "basis": "source", "refs": refs}],
                }]}
            return AnswerDocument.model_validate(document).model_dump(mode="json")

    plan = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route=route, query=route, k=3) for route in routes])
    planner_llm = _CapturePlannerLLM(plan)
    request = "업로드 코드 예시를 설명해줘" if defect == "code" else (
        "공식 문서와 업로드 자료를 비교해줘" if len(routes) == 2 else "공식 문서를 설명해줘"
    )
    graph = build_graph(
        state_type=GraphState, add_user_node=add_user_message,
        summarize_node=lambda state: {},
        planner_node=make_planner_node(planner_llm, verbose=False),
        retrieve_dispatch_node=make_retrieve_dispatch_node(
            lambda query: search("docs", query),
            lambda query, k, retriever=None: search("upload", query),
            verbose=False,
        ),
        synthesize_node=make_synthesize_node(RepairingLLM(), verbose=False),
        pre_synthesis_validation_node=make_pre_synthesis_validation_node(False),
        post_synthesis_validation_node=make_post_synthesis_validation_node(False),
        action_postprocess_node=lambda state: {},
        memory_policy=ConversationMemoryPolicy(),
    )

    state = graph.invoke(build_graph_state_input(
        user_input=request, messages=[], retriever=object() if "upload" in routes else None,
        retry={"max_retries": max_retries},
    ))

    expected_attempts = 1 if max_retries == 0 else 2
    assert len(packets) == expected_attempts, (routes, defect, max_retries)
    assert all(packet == packets[0] for packet in packets)
    assert planner_llm.call_count == 1
    assert searches == {route: int(route in routes) for route in searches}
    result = state["response"].result
    assert all(check.reference_status != "missing" for check in result.checks)
    assert [message.content for message in state["messages"] if isinstance(message, AIMessage)] == [export_answer_text(result)]
    if persistent:
        assert any(issue.code == "source_excerpt_fallback" for issue in result.issues)
        assert "수정 전 본문" not in export_answer_text(result)
    else:
        assert not any(issue.code == "source_excerpt_fallback" for issue in result.issues)
