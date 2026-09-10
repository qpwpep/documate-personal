from langchain_core.messages import AIMessage, HumanMessage
import pytest

from src.core.contracts.boundary.graph import build_graph_state_input, normalize_graph_update
from src.core.conversation_memory import ConversationMemoryPolicy, build_durable_conversation_memory
from src.core.request_contracts import UserTurnSnapshot
from src.runtime.agent_runtime.session_context import SessionContext
from src.runtime.nodes.session import add_user_message


def test_user_message_retry_preserves_the_ingress_id_and_exact_original() -> None:
    state = build_graph_state_input(user_input="  원문\n그대로  ", current_turn_id="turn-current")
    first = add_user_message(state)
    retry = add_user_message({**state, **first})
    assert first["messages"][0].id == retry["messages"][0].id == "turn-current"
    assert retry["runtime"].user_turns == (
        UserTurnSnapshot(turn_id="turn-current", text="  원문\n그대로  "),
    )


def test_durable_trimming_preserves_user_ids_without_replacing_the_original_snapshot() -> None:
    original = "앞부분\n" + "가" * 2000 + "\n참조할 마지막 문장"
    policy = ConversationMemoryPolicy(
        high_water_turns=3, low_water_turns=2,
        high_water_tokens=400, low_water_tokens=200,
        high_water_bytes=1500, low_water_bytes=900,
        summary_max_tokens=24, summary_max_bytes=96, hard_max_bytes=2048,
    )
    memory = build_durable_conversation_memory(
        [HumanMessage(id="turn-original", content=original), AIMessage(content="답변" * 2000)],
        memory_summary=None, policy=policy,
    )
    assert memory.messages[0].id == "turn-original"
    assert memory.messages[0].content != original
    session = SessionContext()
    session.commit_conversation_memory(
        messages=memory.messages, memory_summary=memory.memory_summary,
        user_turns=(UserTurnSnapshot(turn_id="turn-original", text=original),),
    )
    restored = session.snapshot_conversation_memory()
    assert restored.messages[0].id == "turn-original"
    assert restored.user_turns == (UserTurnSnapshot(turn_id="turn-original", text=original),)


def test_runtime_json_roundtrip_keeps_stable_reference_context() -> None:
    state = build_graph_state_input(
        user_input="번역해줘", current_turn_id="turn-next",
        user_turns=(UserTurnSnapshot(turn_id="turn-before", text="original"),),
    )
    restored = normalize_graph_update({"runtime": state["runtime"].model_dump(mode="json")})
    assert restored["runtime"] == state["runtime"]


def test_session_reset_discards_original_turn_snapshots() -> None:
    session = SessionContext()
    session.commit_conversation_memory(
        messages=[HumanMessage(id="turn-old", content="old")], memory_summary=None,
        user_turns=(UserTurnSnapshot(turn_id="turn-old", text="old"),),
    )
    session.reset_conversation_memory()
    assert session.snapshot_conversation_memory().user_turns == ()


def test_session_retains_pending_original_beyond_recent_turn_limit() -> None:
    session = SessionContext()
    turns = tuple(UserTurnSnapshot(turn_id=f"turn-{index}", text=f"원문 {index}") for index in range(40))
    session.commit_conversation_memory(
        messages=[HumanMessage(id="turn-39", content="원문 39")], memory_summary="오래된 대화의 요약",
        user_turns=turns, preserve_turn_ids={"turn-0"},
    )
    retained = session.snapshot_conversation_memory().user_turns
    assert retained == (turns[0], *turns[-16:])


def test_legacy_message_gets_an_id_once_when_entering_session_storage() -> None:
    session = SessionContext()
    session.messages = [HumanMessage(content="기존 대화")]
    first = session.snapshot_conversation_memory()
    session.memory_summary = "요약"
    second = session.snapshot_conversation_memory()
    assert first.messages[0].id == second.messages[0].id
    assert first.user_turns == second.user_turns
    assert first.user_turns[0].turn_id == first.messages[0].id


def test_runtime_rejects_ambiguous_duplicate_turn_id() -> None:
    with pytest.raises(ValueError, match="duplicate user turn ID"):
        build_graph_state_input(
            user_input="new", user_turns=(
                UserTurnSnapshot(turn_id="same-id", text="original"),
                UserTurnSnapshot(turn_id="same-id", text="replacement"),
            ),
        )


def test_session_cannot_replace_an_owned_original_using_the_same_id() -> None:
    session = SessionContext()
    session.messages = [HumanMessage(id="owned-id", content="original")]
    before = session.snapshot_conversation_memory()
    with pytest.raises(ValueError, match="original user text cannot change"):
        session.commit_conversation_memory(
            messages=[], memory_summary="new summary",
            user_turns=(UserTurnSnapshot(turn_id="owned-id", text="replacement"),),
        )
    assert session.snapshot_conversation_memory() == before
