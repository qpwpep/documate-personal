"""Browser recovery retains uncertain requests without scheduling them again."""
import json

from src.app.client import AgentRequestContext, AgentSessionClient
from src.app.web.chat_response_task import ChatResponseTask
from src.app.web.composer_draft import ComposerDraft
from src.app.web.conversation_snapshot import dump_conversations, load_conversations
from src.app.web.streamlit_requests import ChatSubmission
from src.app.web.streamlit_state import Conversation


CONVERSATION_ID = "fc719189-39ce-426f-98b2-782d4625d639"
API_URL = "http://127.0.0.1:8000"


def test_receiving_response_roundtrip_keeps_next_draft_and_never_requeues_question():
    prompt = "  이미 전송한 질문\n두 번째 줄  "
    next_draft = "  기다리는 동안 쓴 다음 질문\n아직 보내지 않음  "
    client = AgentSessionClient(AgentRequestContext(fastapi_url=API_URL, session_id=CONVERSATION_ID))
    submission = ChatSubmission(prompt)
    conversation = Conversation(
        id=CONVERSATION_ID,
        client=client,
        messages=[{"role": "user", "content": prompt}],
        draft=ComposerDraft(text=next_draft, revision=1),
        active_submission=submission,
        operation_in_progress=True,
        chat_task=ChatResponseTask(operation_id=submission.operation_id, client=client),
    )

    restored, active_id = load_conversations(
        dump_conversations([conversation], CONVERSATION_ID), fastapi_url=API_URL,
    )
    recovered = restored[active_id]

    assert active_id == CONVERSATION_ID
    assert recovered.messages == [
        {"role": "user", "content": prompt},
        {"role": "assistant", "result": None, "interrupted": True},
    ]
    assert recovered.draft.text == next_draft
    assert recovered.draft.revision == 1
    assert recovered.draft.needs_restore
    assert recovered.pending_request is None
    assert recovered.active_submission is None
    assert recovered.chat_task is None
    assert recovered.saved_prompt is None
    assert not recovered.is_busy

    reloaded, _ = load_conversations(
        dump_conversations([recovered], active_id), fastapi_url=API_URL,
    )
    assert reloaded[active_id].messages == recovered.messages
    assert reloaded[active_id].draft.text == next_draft
    assert reloaded[active_id].pending_request is None


def test_legacy_queued_snapshot_retains_question_without_replaying_or_consuming_next_draft():
    snapshot = json.dumps({
        "version": 1,
        "active_conversation_id": CONVERSATION_ID,
        "conversations": [{
            "id": CONVERSATION_ID,
            "messages": [],
            "draft": {"text": "다음에 보낼 질문", "revision": 1},
            "pending_prompt": "결과를 확인하지 못한 질문",
        }],
    })

    restored, active_id = load_conversations(snapshot, fastapi_url=API_URL)
    conversation = restored[active_id]

    assert conversation.messages == [
        {"role": "user", "content": "결과를 확인하지 못한 질문"},
        {"role": "assistant", "result": None, "interrupted": True},
    ]
    assert conversation.draft.text == "다음에 보낼 질문"
    assert conversation.pending_request is None
    assert conversation.active_submission is None
    assert conversation.chat_task is None
    assert not conversation.is_busy


def test_legacy_finished_snapshot_does_not_add_an_interruption():
    messages = [
        {"role": "user", "content": "이전 질문"},
        {"role": "assistant", "result": None, "transport_errors": ["응답 연결 실패"]},
    ]
    snapshot = json.dumps({
        "version": 1,
        "active_conversation_id": CONVERSATION_ID,
        "conversations": [{
            "id": CONVERSATION_ID,
            "messages": messages,
            "draft": {"text": "수정 중인 질문", "revision": 2},
        }],
    })

    restored, active_id = load_conversations(snapshot, fastapi_url=API_URL)
    conversation = restored[active_id]

    assert conversation.messages == messages
    assert not conversation.has_interrupted_response
    assert conversation.draft.text == "수정 중인 질문"
    assert conversation.draft.needs_restore
    assert not conversation.is_busy
