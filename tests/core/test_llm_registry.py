import unittest
from unittest.mock import patch

from src.infra.llm import build_llm_registry
from src.core.planner_schema import PlannerOutput
from src.infra.settings import AppSettings


class _FakeChatOpenAI:
    created_kwargs: list[dict] = []
    structured_args: list[tuple] = []
    structured_kwargs: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.created_kwargs.append(kwargs)

    def with_structured_output(self, *_args, **_kwargs):
        self.__class__.structured_args.append(_args)
        self.__class__.structured_kwargs.append(_kwargs)
        return self


class LLMRegistryTest(unittest.TestCase):
    def test_planner_schema_matches_strict_json_schema_shape(self) -> None:
        schema = PlannerOutput.model_json_schema()

        self.assertEqual(set(schema["required"]), {"use_retrieval", "tasks"})
        self.assertIs(schema["additionalProperties"], False)

        task_schema = schema["$defs"]["RetrievalTask"]
        self.assertEqual(set(task_schema["required"]), {"route", "query", "k"})
        self.assertIs(task_schema["additionalProperties"], False)
        self.assertEqual(set(task_schema["properties"]["route"]["enum"]), {"docs", "upload", "local"})

    @patch("src.infra.llm.ChatOpenAI", new=_FakeChatOpenAI)
    def test_build_llm_registry_applies_explicit_synthesis_policy(self) -> None:
        _FakeChatOpenAI.created_kwargs = []
        _FakeChatOpenAI.structured_args = []
        _FakeChatOpenAI.structured_kwargs = []
        settings = AppSettings(
            openai_api_key="test-key",
            tavily_api_key="test-tavily",
            planner_max_tokens=654,
            synthesis_timeout_seconds=9,
            synthesis_max_retries=1,
            synthesis_max_tokens=777,
            synthesis_reasoning_effort=None,
            verbose=False,
        )

        registry = build_llm_registry(settings)

        self.assertEqual(len(_FakeChatOpenAI.created_kwargs), 4)
        synthesizer_kwargs = _FakeChatOpenAI.created_kwargs[0]
        self.assertEqual(synthesizer_kwargs["temperature"], 0)
        self.assertEqual(synthesizer_kwargs["timeout"], 9)
        self.assertEqual(synthesizer_kwargs["max_retries"], 1)
        self.assertEqual(synthesizer_kwargs["max_tokens"], 777)
        self.assertTrue(synthesizer_kwargs["use_responses_api"])
        self.assertEqual(synthesizer_kwargs["output_version"], "responses/v1")
        self.assertEqual(synthesizer_kwargs["verbose"], False)
        self.assertNotIn("reasoning", synthesizer_kwargs)
        compact_kwargs = _FakeChatOpenAI.created_kwargs[1]
        self.assertEqual(compact_kwargs["temperature"], 0)
        self.assertEqual(compact_kwargs["timeout"], 4)
        self.assertEqual(compact_kwargs["max_retries"], 0)
        self.assertEqual(compact_kwargs["max_tokens"], 388)
        self.assertTrue(compact_kwargs["use_responses_api"])
        self.assertEqual(compact_kwargs["output_version"], "responses/v1")
        self.assertNotIn("reasoning", compact_kwargs)
        planner_kwargs = _FakeChatOpenAI.created_kwargs[2]
        self.assertEqual(planner_kwargs["max_tokens"], 654)
        self.assertNotIn("reasoning", planner_kwargs)
        self.assertIsNotNone(registry.llm_synthesizer_compact)
        self.assertEqual(_FakeChatOpenAI.structured_args[0][0], PlannerOutput)
        self.assertEqual(_FakeChatOpenAI.structured_kwargs[0]["include_raw"], True)

    @patch("src.infra.llm.ChatOpenAI", new=_FakeChatOpenAI)
    def test_build_llm_registry_can_apply_reasoning_effort_for_synthesis_only(self) -> None:
        _FakeChatOpenAI.created_kwargs = []
        _FakeChatOpenAI.structured_args = []
        _FakeChatOpenAI.structured_kwargs = []
        settings = AppSettings(
            openai_api_key="test-key",
            tavily_api_key="test-tavily",
            synthesis_reasoning_effort="xhigh",
        )

        build_llm_registry(settings)

        self.assertEqual(_FakeChatOpenAI.created_kwargs[0]["reasoning"], {"effort": "xhigh"})
        self.assertEqual(_FakeChatOpenAI.created_kwargs[1]["reasoning"], {"effort": "xhigh"})
        self.assertNotIn("reasoning", _FakeChatOpenAI.created_kwargs[2])
        self.assertNotIn("reasoning", _FakeChatOpenAI.created_kwargs[3])

    def test_app_settings_normalizes_blank_reasoning_effort_to_none(self) -> None:
        settings = AppSettings(
            openai_api_key="test-key",
            tavily_api_key="test-tavily",
            synthesis_reasoning_effort="",
        )

        self.assertIsNone(settings.synthesis_reasoning_effort)

    @patch("src.infra.llm.ChatOpenAI", new=_FakeChatOpenAI)
    def test_build_llm_registry_preserves_explicit_none_reasoning_override(self) -> None:
        _FakeChatOpenAI.created_kwargs = []
        _FakeChatOpenAI.structured_args = []
        _FakeChatOpenAI.structured_kwargs = []
        settings = AppSettings(
            openai_api_key="test-key",
            tavily_api_key="test-tavily",
            synthesis_reasoning_effort="none",
        )

        build_llm_registry(settings)

        self.assertEqual(settings.synthesis_reasoning_effort, "none")
        self.assertEqual(_FakeChatOpenAI.created_kwargs[0]["reasoning"], {"effort": "none"})
        self.assertEqual(_FakeChatOpenAI.created_kwargs[1]["reasoning"], {"effort": "none"})


if __name__ == "__main__":
    unittest.main()
