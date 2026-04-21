import unittest

from src.runtime.nodes.synthesis.schema_adapter import build_structured_synthesizer


class _CaptureStructuredWrapperLLM:
    def __init__(self):
        self.args = None
        self.kwargs = None

    def with_structured_output(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self


class SynthesisPayloadBuilderTest(unittest.TestCase):
    def test_build_structured_synthesizer_uses_strict_json_schema_dict(self) -> None:
        llm = _CaptureStructuredWrapperLLM()

        result = build_structured_synthesizer(llm)

        self.assertIs(result, llm)
        self.assertIsNotNone(llm.args)
        self.assertIsNotNone(llm.kwargs)
        schema = llm.args[0]
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["name"], "SynthesisOutput")
        self.assertTrue(schema["strict"])
        self.assertIn("schema", schema)
        self.assertEqual(llm.kwargs["method"], "json_schema")
        self.assertTrue(llm.kwargs["include_raw"])
        self.assertTrue(llm.kwargs["strict"])


if __name__ == "__main__":
    unittest.main()
