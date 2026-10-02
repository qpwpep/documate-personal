import unittest
from unittest.mock import patch

from src.infra.llm import build_llm_registry
from src.infra.settings import AppSettings


class _FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def with_structured_output(self, *_args, **_kwargs):
        return self


class LLMRegistryTest(unittest.TestCase):
    @patch("src.infra.llm.ChatOpenAI", new=_FakeChatOpenAI)
    def test_build_llm_registry_applies_role_specific_transport_policy(self) -> None:
        """Keep client timeout/retry checks here; request bodies are tested at HTTP."""
        settings = AppSettings(
            _env_file=None,
            openai_api_key="test-key",
            tavily_api_key="test-tavily",
            synthesis_timeout_seconds=9,
            verbose=False,
        )

        registry = build_llm_registry(settings)

        for role, client, timeout, retries in (
            ("synthesis", registry.llm_synthesizer, 9, 0),
            ("compact", registry.llm_synthesizer_compact, 4, 0),
            ("planner", registry.llm_planner.runnable, 30, 0),
            ("replan", registry.llm_planner_retry.runnable, 30, 0),
            ("summary", registry.llm_summarizer, 60, 2),
        ):
            with self.subTest(role=role):
                self.assertEqual(client.kwargs["timeout"], timeout)
                self.assertEqual(client.kwargs["max_retries"], retries)
                self.assertEqual(client.kwargs["temperature"], 0)
                self.assertFalse(client.kwargs["verbose"])


if __name__ == "__main__":
    unittest.main()
