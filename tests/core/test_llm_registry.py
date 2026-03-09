import unittest
from unittest.mock import patch

from src.llm import build_llm_registry
from src.settings import AppSettings


class _FakeChatOpenAI:
    created_kwargs: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.created_kwargs.append(kwargs)

    def with_structured_output(self, *_args, **_kwargs):
        return self


class LLMRegistryTest(unittest.TestCase):
    @patch("src.llm.ChatOpenAI", new=_FakeChatOpenAI)
    def test_build_llm_registry_applies_explicit_synthesis_policy(self) -> None:
        _FakeChatOpenAI.created_kwargs = []
        settings = AppSettings(
            openai_api_key="test-key",
            tavily_api_key="test-tavily",
            synthesis_timeout_seconds=9,
            synthesis_max_retries=1,
            synthesis_max_tokens=777,
            verbose=False,
        )

        _ = build_llm_registry(settings)

        self.assertEqual(len(_FakeChatOpenAI.created_kwargs), 3)
        synthesizer_kwargs = _FakeChatOpenAI.created_kwargs[0]
        self.assertEqual(synthesizer_kwargs["temperature"], 0)
        self.assertEqual(synthesizer_kwargs["timeout"], 9)
        self.assertEqual(synthesizer_kwargs["max_retries"], 1)
        self.assertEqual(synthesizer_kwargs["max_tokens"], 777)
        self.assertEqual(synthesizer_kwargs["verbose"], False)


if __name__ == "__main__":
    unittest.main()
