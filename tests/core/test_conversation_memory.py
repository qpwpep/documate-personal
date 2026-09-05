from __future__ import annotations

import json
import unittest

from hypothesis import given, settings as hypothesis_settings, strategies as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.core.conversation_memory import (
    ConversationMemoryPolicy,
    DEFAULT_QUERY_MAX_CHARS,
    build_bounded_fallback_summary,
    build_durable_conversation_memory,
    build_untrusted_memory_prompt_messages,
    estimate_text_tokens,
    measure_conversation,
    plan_compaction,
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


class ConversationMemoryPolicyTest(unittest.TestCase):
    def test_compaction_partitions_history_at_a_human_turn_boundary(self) -> None:
        messages = [
            HumanMessage(content="old-user"),
            AIMessage(content="old-answer"),
            ToolMessage(
                content='{"secret":"OLD_TOOL_PAYLOAD"}',
                name="search",
                tool_call_id="old-tool",
            ),
            HumanMessage(content="recent-user"),
            AIMessage(content="recent-answer"),
            HumanMessage(content="current-user"),
        ]

        plan = plan_compaction(messages, memory_summary=None, policy=_policy())

        self.assertTrue(plan.should_compact)
        self.assertEqual(
            [str(message.content) for message in plan.evicted_messages],
            ["old-user", "old-answer", '{"secret":"OLD_TOOL_PAYLOAD"}'],
        )
        self.assertEqual(
            [str(message.content) for message in plan.retained_messages],
            ["recent-user", "recent-answer", "current-user"],
        )
        self.assertEqual(
            [*plan.evicted_messages, *plan.retained_messages],
            messages,
        )

    def test_durable_projection_keeps_only_the_canonical_conversation(self) -> None:
        messages = [
            SystemMessage(content="internal policy"),
            HumanMessage(content="save this"),
            AIMessage(
                content="tool call",
                tool_calls=[{"name": "save_text", "args": {}, "id": "call-1"}],
                response_metadata={"provider_payload": "do not persist"},
            ),
            ToolMessage(
                content='{"file_path":"output/result.txt","raw":"LARGE_TOOL_PAYLOAD"}',
                name="save_text",
                tool_call_id="call-1",
            ),
            AIMessage(content="raw graph answer", response_metadata={"large": "metadata"}),
        ]

        projected = build_durable_conversation_memory(
            messages,
            memory_summary="bounded summary",
            policy=_policy(),
            canonical_assistant_text="answer shown to the user\n\n저장 완료: output/result.txt",
        ).messages

        self.assertEqual(len(projected), 2)
        self.assertIsInstance(projected[0], HumanMessage)
        self.assertIsInstance(projected[1], AIMessage)
        self.assertEqual(projected[0].content, "save this")
        self.assertEqual(
            projected[1].content,
            "answer shown to the user\n\n저장 완료: output/result.txt",
        )
        self.assertEqual(projected[1].response_metadata, {})
        self.assertFalse(any(isinstance(message, ToolMessage) for message in projected))

    def test_fallback_summary_and_prompt_memory_stay_bounded_and_untrusted(self) -> None:
        policy = _policy(summary_max_tokens=24)
        summary = build_bounded_fallback_summary(
            existing_summary="기존 사실 " * 100,
            evicted_transcript="새로운 사실 " * 100,
            policy=policy,
        )

        self.assertLessEqual(estimate_text_tokens(summary), policy.summary_max_tokens)
        prompt_messages = build_untrusted_memory_prompt_messages(summary)
        self.assertIsInstance(prompt_messages[0], SystemMessage)
        self.assertIn("untrusted", str(prompt_messages[0].content).lower())
        self.assertNotIn(summary, str(prompt_messages[0].content))
        self.assertEqual(
            json.loads(str(prompt_messages[1].content)),
            {"kind": "untrusted_conversation_memory", "summary": summary},
        )

    def test_policy_rejects_a_target_that_is_not_below_the_trigger(self) -> None:
        with self.assertRaises(ValueError):
            _policy(high_water_tokens=100, low_water_tokens=100)

    def test_policy_rejects_message_limits_that_cannot_hold_one_complete_turn(self) -> None:
        with self.assertRaises(ValueError):
            _policy(high_water_messages=2, low_water_messages=1)

    def test_policy_rejects_token_limit_without_room_for_summary_and_turn(self) -> None:
        with self.assertRaises(ValueError):
            _policy(
                high_water_tokens=40,
                low_water_tokens=31,
                summary_max_tokens=24,
            )

    def test_policy_rejects_byte_limit_without_room_for_summary_and_turn(self) -> None:
        with self.assertRaises(ValueError):
            _policy(
                high_water_bytes=1_000,
                low_water_bytes=300,
                summary_max_bytes=256,
            )

    def test_durable_memory_fits_json_escaped_content_below_the_byte_watermark(self) -> None:
        policy = ConversationMemoryPolicy()

        memory = build_durable_conversation_memory(
            [
                HumanMessage(content="\x00" * DEFAULT_QUERY_MAX_CHARS),
                AIMessage(content="\x00" * DEFAULT_QUERY_MAX_CHARS),
            ],
            memory_summary="\x00" * 120,
            policy=policy,
        )

        self.assertLess(memory.usage.serialized_bytes, policy.high_water_bytes)
        self.assertLessEqual(memory.usage.serialized_bytes, policy.low_water_bytes)
        self.assertIsNotNone(memory.memory_summary)

    def test_durable_memory_truncates_escape_heavy_turn_instead_of_exceeding_hard_limit(self) -> None:
        policy = _policy(
            high_water_tokens=300,
            low_water_tokens=180,
            high_water_bytes=900,
            low_water_bytes=700,
            high_water_messages=6,
            low_water_messages=4,
            summary_max_tokens=24,
            summary_max_bytes=192,
            hard_max_bytes=1_024,
        )

        memory = build_durable_conversation_memory(
            [
                HumanMessage(content="\x00" * 120),
                AIMessage(content="\x00" * 120),
            ],
            memory_summary=None,
            policy=policy,
        )

        self.assertLess(memory.usage.serialized_bytes, policy.high_water_bytes)
        self.assertLessEqual(memory.usage.serialized_bytes, policy.low_water_bytes)

    def test_single_turn_target_compacts_two_turns_to_the_latest_turn(self) -> None:
        policy = ConversationMemoryPolicy(high_water_turns=2, low_water_turns=1)
        messages = [
            HumanMessage(content="old request"),
            AIMessage(content="old answer"),
            HumanMessage(content="latest request"),
        ]

        plan = plan_compaction(messages, memory_summary=None, policy=policy)

        self.assertTrue(plan.should_compact)
        self.assertEqual(
            [message.content for message in plan.retained_messages],
            ["latest request"],
        )


_unicode_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=0,
    max_size=120,
)


@hypothesis_settings(max_examples=80, deadline=None)
@given(st.lists(_unicode_text, min_size=1, max_size=24))
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


@hypothesis_settings(max_examples=80, deadline=None)
@given(previous=_unicode_text, evicted=_unicode_text)
def test_fallback_summary_never_exceeds_its_token_or_byte_budget(
    previous: str,
    evicted: str,
) -> None:
    policy = _policy(summary_max_tokens=24, summary_max_bytes=64)

    summary = build_bounded_fallback_summary(
        existing_summary=previous,
        evicted_transcript=evicted,
        policy=policy,
    )

    if summary is None:
        assert not previous.strip() and not evicted.strip()
        return
    assert estimate_text_tokens(summary) <= policy.summary_max_tokens
    assert len(summary.encode("utf-8")) <= policy.summary_max_bytes


def test_conversation_memory_plateaus_during_hundreds_of_turns() -> None:
    policy = _policy(
        high_water_turns=5,
        low_water_turns=3,
        high_water_tokens=300,
        low_water_tokens=180,
        high_water_bytes=1_536,
        low_water_bytes=900,
        high_water_messages=12,
        low_water_messages=8,
        summary_max_tokens=32,
        summary_max_bytes=96,
        hard_max_bytes=2_048,
    )
    messages = []
    summary = None
    sizes: list[int] = []
    for turn in range(300):
        memory = build_durable_conversation_memory(
            [
                *messages,
                HumanMessage(content=f"user-{turn}: bounded memory question"),
                ToolMessage(
                    content=f'{{"turn":{turn},"payload":"transient"}}',
                    name="search",
                    tool_call_id=f"tool-{turn}",
                ),
                AIMessage(content=f"assistant-{turn}: bounded answer"),
            ],
            memory_summary=summary,
            policy=policy,
        )
        messages = list(memory.messages)
        summary = memory.memory_summary
        sizes.append(memory.usage.serialized_bytes)

        assert memory.usage.turn_count < policy.high_water_turns
        assert memory.usage.message_count < policy.high_water_messages
        assert memory.usage.estimated_tokens < policy.high_water_tokens
        assert memory.usage.serialized_bytes < policy.high_water_bytes
        assert memory.usage.serialized_bytes <= policy.hard_max_bytes
        assert not any(isinstance(message, ToolMessage) for message in messages)

    assert max(sizes[-50:]) <= policy.hard_max_bytes
    assert not any("user-0:" in str(message.content) for message in messages)
