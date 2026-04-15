import unittest
from unittest.mock import patch

from src.settings import AppSettings
from src.tools import build_tool_registry


class DocsSearchTest(unittest.TestCase):
    @patch("src.tools.docs_search.client.request_tavily_search")
    def test_docs_search_applies_bare_library_hints_for_common_libraries(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {"results": []}
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        cases = [
            ("pandas official docs", "pandas.pydata.org", "pandas user guide"),
            ("numpy official docs", "numpy.org", "numpy user guide"),
            ("fastapi official docs", "fastapi.tiangolo.com", "fastapi tutorial"),
        ]

        for query, expected_domain, expected_fallback in cases:
            with self.subTest(query=query):
                mock_request_tavily_search.reset_mock()
                registry.tavily_search_tool.func(query=query)

                self.assertGreaterEqual(len(mock_request_tavily_search.call_args_list), 2)
                first_kwargs = mock_request_tavily_search.call_args_list[0].kwargs
                second_kwargs = mock_request_tavily_search.call_args_list[1].kwargs
                self.assertEqual(first_kwargs["include_domains"], [expected_domain])
                self.assertEqual(first_kwargs["query"], query)
                self.assertEqual(second_kwargs["query"], expected_fallback)

    @patch("src.tools.docs_search.client.request_tavily_search")
    def test_docs_search_uses_fallback_when_first_batch_is_cross_library_only(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.side_effect = [
            {
                "results": [
                    {
                        "url": "https://pandas.pydata.org/docs/reference/api/pandas.merge.html",
                        "title": "pandas.merge",
                        "content": "pandas merge docs",
                        "score": 0.92,
                    }
                ]
            },
            {
                "results": [
                    {
                        "url": "https://numpy.org/doc/stable/user/basics.broadcasting.html",
                        "title": "Broadcasting",
                        "content": "NumPy broadcasting stretches compatible array dimensions.",
                        "score": 0.88,
                    }
                ]
            },
        ]

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool.func(query="numpy official docs")

        self.assertEqual(len(mock_request_tavily_search.call_args_list), 2)
        self.assertEqual(
            [item["url_or_path"] for item in result["evidence"]],
            ["https://numpy.org/doc/stable/user/basics.broadcasting.html"],
        )

    @patch("src.tools.docs_search.client.request_tavily_search")
    def test_docs_search_adopts_only_same_library_domain_results(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {
            "results": [
                {
                    "url": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
                    "title": "train_test_split",
                    "content": "Split arrays or matrices into random train and test subsets.",
                    "score": 0.95,
                },
                {
                    "url": "https://numpy.org/doc/stable/reference/generated/numpy.split.html",
                    "title": "numpy.split",
                    "content": "Split an array into multiple sub-arrays.",
                    "score": 0.9,
                },
            ]
        }

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool.func(query="train_test_split official docs")

        self.assertEqual(
            [item["url_or_path"] for item in result["evidence"]],
            [
                "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"
            ],
        )

    @patch("src.tools.docs_search.client.request_tavily_search")
    def test_docs_search_does_not_treat_concatenate_as_pandas_concat(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {
            "results": [
                {
                    "url": "https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
                    "title": "numpy.concatenate",
                    "content": "Join a sequence of arrays along an existing axis.",
                    "score": 0.94,
                }
            ]
        }

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool.func(query="numpy concatenate official docs")

        self.assertEqual(
            [item["url_or_path"] for item in result["evidence"]],
            ["https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html"],
        )
        first_kwargs = mock_request_tavily_search.call_args_list[0].kwargs
        self.assertEqual(first_kwargs["include_domains"], ["numpy.org"])

    @patch("src.tools.docs_search.client.request_tavily_search")
    def test_docs_search_uses_fallback_when_first_batch_is_docs_chrome_only(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.side_effect = [
            {
                "results": [
                    {
                        "url": "https://numpy.org/doc/stable/user/basics.broadcasting.html",
                        "title": "Broadcasting",
                        "content": "Home > Docs > API\nTable of contents\nPrevious: Intro",
                        "score": 0.91,
                    }
                ]
            },
            {
                "results": [
                    {
                        "url": "https://numpy.org/doc/stable/user/basics.broadcasting.html",
                        "title": "Broadcasting",
                        "content": "Broadcasting stretches compatible array dimensions.",
                        "score": 0.88,
                    }
                ]
            },
        ]

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool.func(query="numpy official docs")

        self.assertGreaterEqual(len(mock_request_tavily_search.call_args_list), 2)
        self.assertEqual(
            [item["url_or_path"] for item in result["evidence"]],
            ["https://numpy.org/doc/stable/user/basics.broadcasting.html"],
        )
        self.assertIn("Broadcasting stretches compatible array dimensions.", result["evidence"][0]["snippet"])


if __name__ == "__main__":
    unittest.main()
