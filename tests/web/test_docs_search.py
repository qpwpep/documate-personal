import unittest
from unittest.mock import patch

from src.infra.settings import AppSettings
from src.infra.tools import build_tool_registry


class DocsSearchTest(unittest.TestCase):
    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_applies_bare_library_hints_for_common_libraries(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {"results": []}
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        cases = [
            ("python official docs", "docs.python.org", "python asyncio task documentation"),
            ("git official docs", "git-scm.com", "git rebase documentation"),
            ("LangChain official docs", "python.langchain.com", "LangChain retrievers docs"),
            ("pandas official docs", "pandas.pydata.org", "pandas user guide"),
            ("numpy official docs", "numpy.org", "numpy user guide"),
            ("matplotlib official docs", "matplotlib.org", "matplotlib api reference"),
            ("PyTorch official docs", "docs.pytorch.org", "PyTorch torch.Tensor documentation"),
            ("Hugging Face official docs", "huggingface.co", "Hugging Face Transformers tokenizer padding docs"),
            ("fastapi official docs", "fastapi.tiangolo.com", "fastapi tutorial"),
            ("BeautifulSoup official docs", "crummy.com", "BeautifulSoup find find_all select official docs"),
            ("streamlit official docs", "docs.streamlit.io", "streamlit api reference"),
            ("gradio official docs", "gradio.app", "gradio docs"),
            ("scikit-learn official docs", "scikit-learn.org", "scikit-learn user guide"),
            ("Pydantic official docs", "docs.pydantic.dev", "pydantic concepts documentation"),
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

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_applies_user_like_feature_hints_for_supported_sites(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {"results": []}
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        cases = [
            (
                "python asyncio task cancellation official docs",
                "docs.python.org",
                "python asyncio task documentation",
            ),
            (
                "git rebase conflict resolution official docs",
                "git-scm.com",
                "git rebase documentation",
            ),
            (
                "LangChain retriever usage official docs",
                "python.langchain.com",
                "LangChain retrievers docs",
            ),
            (
                "matplotlib histogram bins official docs",
                "matplotlib.org",
                "matplotlib.pyplot.hist api reference",
            ),
            (
                "NumPy reshape array official docs",
                "numpy.org",
                "numpy reshape documentation",
            ),
            (
                "pandas groupby aggregate official docs",
                "pandas.pydata.org",
                "pandas groupby user guide",
            ),
            (
                "PyTorch DataLoader dataset official docs",
                "docs.pytorch.org",
                "torch.utils.data Dataset DataLoader",
            ),
            (
                "Hugging Face tokenizer padding official docs",
                "huggingface.co",
                "Hugging Face Transformers tokenizer padding docs",
            ),
            (
                "FastAPI Depends dependency official docs",
                "fastapi.tiangolo.com",
                "FastAPI Depends reference",
            ),
            (
                "BeautifulSoup find_all CSS selector official docs",
                "crummy.com",
                "BeautifulSoup find find_all select official docs",
            ),
            (
                "streamlit st.session_state widget official docs",
                "docs.streamlit.io",
                "streamlit st.session_state docs",
            ),
            (
                "gradio Blocks click event official docs",
                "gradio.app",
                "gradio Blocks docs",
            ),
            (
                "scikit-learn train_test_split stratify official docs",
                "scikit-learn.org",
                "train_test_split sklearn.model_selection",
            ),
            (
                "Pydantic BaseModel model_validate official docs",
                "docs.pydantic.dev",
                "pydantic model_validate documentation",
            ),
        ]

        for query, expected_domain, expected_fallback in cases:
            with self.subTest(query=query):
                mock_request_tavily_search.reset_mock()
                registry.tavily_search_tool.func(query=query)

                self.assertGreaterEqual(len(mock_request_tavily_search.call_args_list), 2)
                first_kwargs = mock_request_tavily_search.call_args_list[0].kwargs
                second_kwargs = mock_request_tavily_search.call_args_list[1].kwargs
                self.assertEqual(first_kwargs["include_domains"], [expected_domain])
                self.assertEqual(second_kwargs["query"], expected_fallback)

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_applies_specific_fallback_hints(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {"results": []}
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        cases = [
            (
                "PyTorch Dataset DataLoader official docs",
                "docs.pytorch.org",
                "torch.utils.data Dataset DataLoader",
            ),
            (
                "Pydantic v2 Field validation official docs",
                "docs.pydantic.dev",
                "pydantic Field validation validator",
            ),
            (
                "pandas concat official docs",
                "pandas.pydata.org",
                "pandas.concat api reference",
            ),
            (
                "matplotlib pie official docs",
                "matplotlib.org",
                "matplotlib.pyplot.pie parameters",
            ),
            (
                "BeautifulSoup으로 특정 태그 찾는 예제를 보여줘",
                "crummy.com",
                "BeautifulSoup find find_all select official docs",
            ),
            (
                "find_all official docs",
                "crummy.com",
                "BeautifulSoup find find_all select official docs",
            ),
        ]

        for query, expected_domain, expected_fallback in cases:
            with self.subTest(query=query):
                mock_request_tavily_search.reset_mock()
                registry.tavily_search_tool.func(query=query)

                self.assertGreaterEqual(len(mock_request_tavily_search.call_args_list), 2)
                first_kwargs = mock_request_tavily_search.call_args_list[0].kwargs
                second_kwargs = mock_request_tavily_search.call_args_list[1].kwargs
                self.assertEqual(first_kwargs["include_domains"], [expected_domain])
                self.assertEqual(second_kwargs["query"], expected_fallback)

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_continues_fallback_until_identifier_coverage_is_complete(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.side_effect = [
            {
                "results": [
                    {
                        "url": "https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset",
                        "title": "torch.utils.data.Dataset",
                        "content": "Dataset represents a dataset.",
                        "score": 0.91,
                    }
                ]
            },
            {
                "results": [
                    {
                        "url": "https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader",
                        "title": "torch.utils.data.DataLoader",
                        "content": "DataLoader loads data from a Dataset.",
                        "score": 0.9,
                    }
                ]
            },
        ]

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool.func(query="PyTorch Dataset DataLoader official docs")

        self.assertEqual(len(mock_request_tavily_search.call_args_list), 2)
        combined = " ".join(item["snippet"] for item in result["evidence"])
        self.assertEqual(result["diagnostics"]["status"], "success")
        self.assertIn("Dataset", combined)
        self.assertIn("DataLoader", combined)

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_returns_no_result_when_identifier_coverage_stays_incomplete(self, mock_request_tavily_search) -> None:
        dataset_only = {
            "results": [
                {
                    "url": "https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset",
                    "title": "torch.utils.data.Dataset",
                    "content": "Dataset represents a dataset.",
                    "score": 0.91,
                }
            ]
        }
        mock_request_tavily_search.side_effect = [dataset_only, dataset_only, dataset_only, dataset_only, dataset_only]

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool.func(query="PyTorch Dataset DataLoader official docs")

        self.assertEqual(result["diagnostics"]["status"], "no_result")
        self.assertEqual(result["evidence"], [])
        self.assertIn("identifier_coverage_incomplete", result["diagnostics"]["warnings"])

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
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

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
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

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
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

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_canonicalizes_numpy_versioned_urls_to_stable(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {
            "results": [
                {
                    "url": "https://numpy.org/doc/2.3/reference/generated/numpy.reshape.html",
                    "title": "numpy.reshape - NumPy v2.3 Manual",
                    "content": "Gives a new shape to an array without changing its data.",
                    "score": 0.93,
                }
            ]
        }

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool.func(query="NumPy reshape array official docs")

        self.assertEqual(result["diagnostics"]["status"], "success")
        self.assertEqual(
            [item["url_or_path"] for item in result["evidence"]],
            ["https://numpy.org/doc/stable/reference/generated/numpy.reshape.html"],
        )

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
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

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_adopts_beautifulsoup_official_docs_for_korean_example_request(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {
            "results": [
                {
                    "url": "https://www.crummy.com/software/BeautifulSoup/bs4/doc/#searching-the-tree",
                    "title": "Beautiful Soup Documentation - Searching the tree",
                    "content": "The find_all method looks through a tag's descendants and retrieves matching tags.",
                    "score": 0.93,
                }
            ]
        }

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool.func(query="BeautifulSoup으로 특정 태그 찾는 예제를 보여줘")

        first_kwargs = mock_request_tavily_search.call_args_list[0].kwargs
        self.assertEqual(first_kwargs["include_domains"], ["crummy.com"])
        self.assertEqual(result["diagnostics"]["status"], "success")
        self.assertEqual(
            [item["url_or_path"] for item in result["evidence"]],
            ["https://www.crummy.com/software/BeautifulSoup/bs4/doc/#searching-the-tree"],
        )


if __name__ == "__main__":
    unittest.main()
