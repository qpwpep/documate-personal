from __future__ import annotations


from hypothesis import example, given, settings as hypothesis_settings, strategies as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.core.conversation_memory import (
    ConversationMemoryPolicy,
    build_durable_conversation_memory,
)


def _policy(**overrides: int) -> ConversationMemoryPolicy:
    values = {
        "high_water_turns": 3,
        "low_water_turns": 2,
        "high_water_tokens": 1_000,
        "low_water_tokens": 800,
        "high_water_bytes": 8_192,
        "low_water_bytes": 4_096,
        "high_water_messages": 16,
        "low_water_messages": 8,
        "summary_max_tokens": 32,
        "summary_max_bytes": 256,
        "hard_max_bytes": 12_288,
    }
    values.update(overrides)
    return ConversationMemoryPolicy(**values)


_unicode_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=0,
    max_size=120,
)


@hypothesis_settings(max_examples=80, deadline=None)
@given(st.lists(_unicode_text, min_size=1, max_size=24))
@example(turn_texts=["\x00" * 120])
def test_durable_projection_stays_within_every_hard_bound_for_unicode_history(
    turn_texts: list[str],
) -> None:
    policy = _policy(
        high_water_turns=4,
        low_water_turns=3,
        high_water_tokens=300,
        low_water_tokens=180,
        high_water_bytes=900,
        low_water_bytes=700,
        high_water_messages=10,
        low_water_messages=8,
        summary_max_tokens=24,
        summary_max_bytes=192,
        hard_max_bytes=1_024,
    )
    messages = []
    for index, text in enumerate(turn_texts):
        messages.extend(
            [
                HumanMessage(content=f"user-{index}:{text}"),
                ToolMessage(
                    content=f'{{"tool-{index}":"{text * 3}"}}',
                    name="search",
                    tool_call_id=f"tool-{index}",
                ),
                AIMessage(content=f"assistant-{index}:{text}"),
            ]
        )

    memory = build_durable_conversation_memory(
        messages,
        memory_summary="요약 " * 20,
        policy=policy,
        canonical_assistant_text=f"visible:{turn_texts[-1]}",
    )
    projected = list(memory.messages)
    usage = memory.usage

    assert len(projected) < policy.high_water_messages
    assert usage.estimated_tokens < policy.high_water_tokens
    assert usage.serialized_bytes < policy.high_water_bytes
    assert usage.serialized_bytes <= policy.hard_max_bytes
    assert all(isinstance(message, (HumanMessage, AIMessage)) for message in projected)
    assert any(isinstance(message, HumanMessage) for message in projected)
    projected_again = build_durable_conversation_memory(
        projected,
        memory_summary=memory.memory_summary,
        policy=policy,
    ).messages
    assert [
        (type(message), str(message.content)) for message in projected_again
    ] == [(type(message), str(message.content)) for message in projected]
