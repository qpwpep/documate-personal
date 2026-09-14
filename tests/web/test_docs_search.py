import unittest
from unittest.mock import Mock, patch

import requests

from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.infra.settings import AppSettings
from src.infra.tools import build_tool_registry
from src.infra.tools.docs_search.client import request_tavily_search
from src.infra.tools.docs_search.policy import canonicalize_doc_url, is_allowed_doc_url
from src.infra.tools.docs_search.ranking import extract_exact_identifier_terms, has_exact_identifier_coverage
from src.infra.tools.docs_search.url_validation import DocUrlValidationResult


class DocsSearchTest(unittest.TestCase):
    def _hit(self, *, title: str, url: str, content: str) -> SearchHit:
        snapshot = build_snapshot(
            source_uri=url, title=title, media_type="text/plain", source_type="official",
            content=content, parser="test", parser_version="1", capture_scope="provider_excerpt",
        )
        element = DocumentElement(
            element_id=f"{snapshot.snapshot_id}:body", kind="paragraph", text=content,
            anchors=[SourceAnchor(kind="web", start=0, end=len(content), precision="exact")],
        )
        return SearchHit(
            evidence=build_evidence(snapshot=snapshot, element=element), rank=1,
            score=RetrievalScore(metric="provider_score", raw=0.8, normalized=0.8, direction="higher"),
        )

    def setUp(self) -> None:
        self._url_validation_patcher = patch("src.infra.tools.docs_search.serialization.validate_doc_url")
        self.mock_validate_doc_url = self._url_validation_patcher.start()
        self.mock_validate_doc_url.side_effect = lambda url: DocUrlValidationResult(
            ok=True,
            final_url=url,
            status_code=200,
        )
        self.addCleanup(self._url_validation_patcher.stop)

    def _provider_response(self, body: object, *, status_code: int = 200) -> Mock:
        response = Mock()
        response.status_code = status_code
        response.json.return_value = body
        return response

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_preserves_library_when_reformulating_general_query(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {"results": []}
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        cases = [
            ("python official docs", "docs.python.org"),
            ("git official docs", "git-scm.com"),
            ("LangChain official docs", "python.langchain.com"),
            ("pandas official docs", "pandas.pydata.org"),
            ("numpy official docs", "numpy.org"),
            ("matplotlib official docs", "matplotlib.org"),
            ("PyTorch official docs", "docs.pytorch.org"),
            ("Hugging Face official docs", "huggingface.co"),
            ("fastapi official docs", "fastapi.tiangolo.com"),
            ("BeautifulSoup official docs", "crummy.com"),
            ("streamlit official docs", "docs.streamlit.io"),
            ("gradio official docs", "gradio.app"),
            ("scikit-learn official docs", "scikit-learn.org"),
            ("Pydantic official docs", ["docs.pydantic.dev", "pydantic.dev"]),
        ]

        for query, expected_domain in cases:
            with self.subTest(query=query):
                mock_request_tavily_search.reset_mock()
                registry.tavily_search_tool(query=query)

                self.assertGreaterEqual(len(mock_request_tavily_search.call_args_list), 2)
                first_kwargs = mock_request_tavily_search.call_args_list[0].kwargs
                second_kwargs = mock_request_tavily_search.call_args_list[1].kwargs
                expected_domains = expected_domain if isinstance(expected_domain, list) else [expected_domain]
                self.assertEqual(first_kwargs["include_domains"], expected_domains)
                self.assertEqual(first_kwargs["query"], query)
                self.assertIn(first_kwargs["query"], second_kwargs["query"].replace('"', ""))
                self.assertEqual(len(mock_request_tavily_search.call_args_list), 2)

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_preserves_feature_subject_when_reformulating_query(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {"results": []}
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        cases = [
            ("python asyncio task cancellation official docs", "docs.python.org"),
            ("git rebase conflict resolution official docs", "git-scm.com"),
            ("LangChain retriever usage official docs", "python.langchain.com"),
            ("matplotlib histogram bins official docs", "matplotlib.org"),
            ("NumPy reshape array official docs", "numpy.org"),
            ("pandas groupby aggregate official docs", "pandas.pydata.org"),
            ("PyTorch DataLoader dataset official docs", "docs.pytorch.org"),
            ("Hugging Face tokenizer padding official docs", "huggingface.co"),
            ("FastAPI Depends dependency official docs", "fastapi.tiangolo.com"),
            ("BeautifulSoup find_all CSS selector official docs", "crummy.com"),
            ("streamlit st.session_state widget official docs", "docs.streamlit.io"),
            ("gradio Blocks click event official docs", "gradio.app"),
            ("scikit-learn train_test_split stratify official docs", "scikit-learn.org"),
            ("Pydantic BaseModel model_validate official docs", ["docs.pydantic.dev", "pydantic.dev"]),
        ]

        for query, expected_domain in cases:
            with self.subTest(query=query):
                mock_request_tavily_search.reset_mock()
                registry.tavily_search_tool(query=query)

                self.assertGreaterEqual(len(mock_request_tavily_search.call_args_list), 2)
                first_kwargs = mock_request_tavily_search.call_args_list[0].kwargs
                second_kwargs = mock_request_tavily_search.call_args_list[1].kwargs
                expected_domains = expected_domain if isinstance(expected_domain, list) else [expected_domain]
                self.assertEqual(first_kwargs["include_domains"], expected_domains)
                self.assertIn(first_kwargs["query"], second_kwargs["query"].replace('"', ""))
                self.assertEqual(len(mock_request_tavily_search.call_args_list), 2)

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_preserves_subject_when_reformulating_symbol_query(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {"results": []}
        registry = build_tool_registry(
            AppSettings(
                openai_api_key="test",
                tavily_api_key="test",
            )
        )

        cases = [
            ("PyTorch Dataset DataLoader official docs", "docs.pytorch.org"),
            ("Pydantic v2 Field validation official docs", ["docs.pydantic.dev", "pydantic.dev"]),
            ("pandas concat official docs", "pandas.pydata.org"),
            ("matplotlib pie official docs", "matplotlib.org"),
            ("BeautifulSoup으로 특정 태그 찾는 예제를 보여줘", "crummy.com"),
            ("find_all official docs", "crummy.com"),
        ]

        for query, expected_domain in cases:
            with self.subTest(query=query):
                mock_request_tavily_search.reset_mock()
                registry.tavily_search_tool(query=query)

                self.assertGreaterEqual(len(mock_request_tavily_search.call_args_list), 2)
                first_kwargs = mock_request_tavily_search.call_args_list[0].kwargs
                second_kwargs = mock_request_tavily_search.call_args_list[1].kwargs
                expected_domains = expected_domain if isinstance(expected_domain, list) else [expected_domain]
                self.assertEqual(first_kwargs["include_domains"], expected_domains)
                self.assertIn(first_kwargs["query"], second_kwargs["query"].replace('"', ""))
                self.assertEqual(len(mock_request_tavily_search.call_args_list), 2)

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_canonicalizes_spaced_dotted_query_tokens(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {"results": []}
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        cases = [
            ("pandas. concat official docs", "pandas concat official docs", ["pandas.pydata.org"]),
            ("pandas. DataFrame. merge official docs", "pandas.DataFrame.merge official docs", ["pandas.pydata.org"]),
            ("matplotlib. pyplot. hist official docs", "matplotlib.pyplot.hist official docs", ["matplotlib.org"]),
            ("Standard. Scaler official docs", "StandardScaler official docs scikit-learn", ["scikit-learn.org"]),
            ("Git. Reset official docs", "Git Reset official docs", ["git-scm.com"]),
        ]

        for query, expected_query, expected_domains in cases:
            with self.subTest(query=query):
                mock_request_tavily_search.reset_mock()

                registry.tavily_search_tool(query=query)

                first_kwargs = mock_request_tavily_search.call_args_list[0].kwargs
                self.assertEqual(first_kwargs["query"], expected_query)
                self.assertEqual(first_kwargs["include_domains"], expected_domains)

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
        result = registry.tavily_search_tool(query="PyTorch Dataset DataLoader official docs")

        self.assertEqual(len(mock_request_tavily_search.call_args_list), 2)
        combined = " ".join(SearchHit.model_validate(item).evidence.excerpt for item in result["hits"])
        self.assertEqual(result["diagnostics"]["status"], "success")
        self.assertIn("Dataset", combined)
        self.assertIn("DataLoader", combined)

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_preserves_partial_evidence_when_identifier_coverage_stays_incomplete(self, mock_request_tavily_search) -> None:
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
        result = registry.tavily_search_tool(query="PyTorch Dataset DataLoader official docs")

        self.assertEqual(result["diagnostics"]["answerability"], "partial")
        self.assertEqual(len(result["hits"]), 1)
        self.assertEqual(result["diagnostics"]["missing_requirements"], ["symbol:DataLoader"])
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
        result = registry.tavily_search_tool(query="numpy official docs")

        self.assertEqual(len(mock_request_tavily_search.call_args_list), 2)
        self.assertEqual(
            [item["evidence"]["snapshot"]["source_uri"] for item in result["hits"]],
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
        result = registry.tavily_search_tool(query="train_test_split official docs")

        self.assertEqual(
            [item["evidence"]["snapshot"]["source_uri"] for item in result["hits"]],
            [
                "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"
            ],
        )

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_keeps_scikit_learn_version_alias_results(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {
            "results": [
                {
                    "url": "https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.train_test_split.html",
                    "title": "train_test_split",
                    "content": "Split arrays or matrices into random train and test subsets.",
                    "score": 0.95,
                }
            ]
        }

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool(query="train_test_split official docs")

        self.assertEqual(result["diagnostics"]["status"], "success")
        self.assertEqual(
            [item["evidence"]["snapshot"]["source_uri"] for item in result["hits"]],
            [
                "https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.train_test_split.html"
            ],
        )

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_keeps_validated_original_url_when_canonical_alias_fails(self, mock_request_tavily_search) -> None:
        original_url = "https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.StandardScaler.html"
        stable_url = "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html"
        mock_request_tavily_search.return_value = {
            "results": [
                {
                    "url": original_url,
                    "title": "StandardScaler",
                    "content": "Standardize features by removing the mean and scaling to unit variance.",
                    "score": 0.95,
                }
            ]
        }
        self.mock_validate_doc_url.side_effect = [
            DocUrlValidationResult(
                ok=False,
                final_url=stable_url,
                status_code=404,
                reason="http_error",
            ),
            DocUrlValidationResult(
                ok=True,
                final_url=original_url,
                status_code=200,
            ),
        ]

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool(query="Standard. Scaler official docs")

        self.assertEqual(result["diagnostics"]["status"], "success")
        self.assertEqual([item["evidence"]["snapshot"]["source_uri"] for item in result["hits"]], [original_url])

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
        result = registry.tavily_search_tool(query="numpy concatenate official docs")

        self.assertEqual(
            [item["evidence"]["snapshot"]["source_uri"] for item in result["hits"]],
            ["https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html"],
        )
        self.assertIn("provider_ms", result["diagnostics"])
        self.assertIn("url_validation_ms", result["diagnostics"])
        self.assertIn("post_filter_ms", result["diagnostics"])
        self.assertFalse(result["diagnostics"]["include_raw_content_requested"])
        first_kwargs = mock_request_tavily_search.call_args_list[0].kwargs
        self.assertEqual(first_kwargs["include_domains"], ["numpy.org"])

    @patch("src.infra.tools.docs_search.client.requests.post")
    def test_docs_search_sends_one_provider_request_when_initial_evidence_is_sufficient(
        self,
        mock_post,
    ) -> None:
        mock_post.return_value = self._provider_response(
            {
                "results": [
                    {
                        "url": "https://numpy.org/doc/stable/user/basics.broadcasting.html",
                        "title": "Broadcasting",
                        "content": "NumPy broadcasting stretches compatible array dimensions.",
                        "score": 0.88,
                    }
                ],
            }
        )
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        result = registry.tavily_search_tool(
            query="numpy broadcasting official docs",
            include_domains=["numpy.org"],
        )

        self.assertEqual(result["diagnostics"]["status"], "success")
        self.assertEqual(len(mock_post.call_args_list), 1)
        self.assertEqual(mock_post.call_args.kwargs["json"]["query"], "numpy broadcasting official docs")
        self.assertEqual(mock_post.call_args.kwargs["json"]["include_domains"], ["numpy.org"])

    @patch("src.infra.tools.docs_search.client.requests.post")
    def test_docs_search_reports_timeout_when_every_quality_fallback_times_out(
        self,
        mock_post,
    ) -> None:
        mock_post.side_effect = [
            self._provider_response({"results": []}),
            requests.Timeout("first fallback timed out"),
            requests.Timeout("second fallback timed out"),
        ]
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        result = registry.tavily_search_tool(query="numpy official docs")

        self.assertEqual(len(mock_post.call_args_list), 2)
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["diagnostics"]["status"], "error")
        self.assertEqual(result["diagnostics"]["error_code"], "RETRIEVAL_DOCS_TIMEOUT")
        self.assertIn("timed out", result["diagnostics"]["message"])

    @patch("src.infra.tools.docs_search.client.requests.post")
    def test_docs_search_reports_provider_failure_when_every_quality_fallback_is_invalid(
        self,
        mock_post,
    ) -> None:
        mock_post.side_effect = [
            self._provider_response({"results": []}),
            self._provider_response({"unexpected": []}),
            self._provider_response({"results": "invalid"}),
        ]
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        result = registry.tavily_search_tool(query="numpy official docs")

        self.assertEqual(len(mock_post.call_args_list), 2)
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["diagnostics"]["status"], "error")
        self.assertEqual(result["diagnostics"]["error_code"], "RETRIEVAL_DOCS_FAILED")
        self.assertIn("invalid Tavily results payload", result["diagnostics"]["message"])

    @patch("src.infra.tools.docs_search.client.requests.post")
    def test_docs_search_classifies_initial_provider_timeout(self, mock_post) -> None:
        mock_post.side_effect = requests.Timeout("provider timeout")
        registry = build_tool_registry(AppSettings(
            openai_api_key="test", tavily_api_key="test", docs_search_timeout_seconds=5,
        ))

        result = registry.tavily_search_tool(
            query="custom docs",
            include_domains=["numpy.org"],
        )

        self.assertEqual(len(mock_post.call_args_list), 1)
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["diagnostics"]["status"], "error")
        self.assertEqual(result["diagnostics"]["error_code"], "RETRIEVAL_DOCS_TIMEOUT")
        self.assertIn("timed out after 5s", result["diagnostics"]["message"])

    @patch("src.infra.tools.docs_search.client.requests.post")
    def test_docs_search_classifies_initial_transport_failure(self, mock_post) -> None:
        mock_post.side_effect = requests.ConnectionError("connection refused")
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        result = registry.tavily_search_tool(
            query="custom docs",
            include_domains=["numpy.org"],
        )

        self.assertEqual(len(mock_post.call_args_list), 1)
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["diagnostics"]["status"], "error")
        self.assertEqual(result["diagnostics"]["error_code"], "RETRIEVAL_DOCS_FAILED")
        self.assertIn("Tavily request failed", result["diagnostics"]["message"])

    @patch("src.infra.tools.docs_search.client.requests.post")
    def test_docs_search_classifies_initial_invalid_provider_payload(self, mock_post) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"results": "invalid"}
        mock_post.return_value = response
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        result = registry.tavily_search_tool(
            query="custom docs",
            include_domains=["numpy.org"],
        )

        self.assertEqual(len(mock_post.call_args_list), 1)
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["diagnostics"]["status"], "error")
        self.assertEqual(result["diagnostics"]["error_code"], "RETRIEVAL_DOCS_FAILED")
        self.assertIn("invalid Tavily results payload", result["diagnostics"]["message"])

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_preserves_numpy_provider_version_when_stable_alias_validates(self, mock_request_tavily_search) -> None:
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
        result = registry.tavily_search_tool(query="NumPy reshape array official docs")

        self.assertEqual(result["diagnostics"]["status"], "success")
        self.assertEqual(
            [item["evidence"]["snapshot"]["source_uri"] for item in result["hits"]],
            ["https://numpy.org/doc/2.3/reference/generated/numpy.reshape.html"],
        )
        self.assertEqual(result["hits"][0]["evidence"]["snapshot"]["title"], "numpy.reshape - NumPy v2.3 Manual")

    def test_docs_search_canonicalizes_pytorch_versioned_docs_urls_to_stable(self) -> None:
        self.assertEqual(
            canonicalize_doc_url("https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.html"),
            "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.html",
        )
        self.assertEqual(
            canonicalize_doc_url("https://docs.pytorch.org/docs/2.9/_sources/generated/torch.Tensor.rst.txt"),
            "https://docs.pytorch.org/docs/stable/_sources/generated/torch.Tensor.rst.txt",
        )

    def test_docs_search_canonicalizes_stable_root_version_aliases(self) -> None:
        self.assertEqual(
            canonicalize_doc_url(
                "https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.train_test_split.html"
            ),
            "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
        )
        self.assertEqual(
            canonicalize_doc_url(
                "https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.StandardScaler.html"
            ),
            "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html",
        )
        self.assertTrue(
            is_allowed_doc_url(
                "https://scikit-learn.org/1.x/modules/generated/sklearn.pipeline.Pipeline.html"
            )
        )

    def test_docs_search_allows_pytorch_tutorial_urls(self) -> None:
        self.assertTrue(
            is_allowed_doc_url("https://docs.pytorch.org/tutorials/beginner/basics/intro.html")
        )

    def test_docs_search_canonicalizes_pydantic_v2_urls_to_latest(self) -> None:
        self.assertEqual(
            canonicalize_doc_url("https://docs.pydantic.dev/2.10/concepts/models/"),
            "https://docs.pydantic.dev/latest/concepts/models/",
        )
        self.assertEqual(
            canonicalize_doc_url("https://docs.pydantic.dev/2.x/concepts/validators/"),
            "https://docs.pydantic.dev/latest/concepts/validators/",
        )

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_filters_http_error_urls(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {
            "results": [
                {
                    "url": "https://docs.pydantic.dev/latest/usage/types/uuids/",
                    "title": "Types/uuids",
                    "content": "UUID parsing details.",
                    "score": 0.92,
                }
            ]
        }
        self.mock_validate_doc_url.side_effect = lambda url: DocUrlValidationResult(
            ok=False,
            final_url="https://pydantic.dev/docs/validation/latest/usage/types/uuids/",
            status_code=404,
            reason="http_error",
        )

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool(
            query="Pydantic UUID official docs",
            include_domains=["docs.pydantic.dev"],
        )

        self.assertEqual(result["diagnostics"]["status"], "no_result")
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["diagnostics"]["filtered_http_error_count"], 1)
        self.assertIn("url_http_error_filtered", result["diagnostics"]["warnings"])

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_records_allowed_redirect_without_reassigning_provider_content(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {
            "results": [
                {
                    "url": "https://docs.pydantic.dev/latest/concepts/fields/",
                    "title": "Fields",
                    "content": "Field customizes model fields, default values, and validation constraints.",
                    "score": 0.93,
                }
            ]
        }
        self.mock_validate_doc_url.side_effect = lambda url: DocUrlValidationResult(
            ok=True,
            final_url="https://pydantic.dev/docs/validation/latest/concepts/fields/",
            status_code=200,
        )

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool(query="Pydantic fields official docs")

        self.assertEqual(result["diagnostics"]["status"], "success")
        self.assertEqual(
            [item["evidence"]["snapshot"]["source_uri"] for item in result["hits"]],
            ["https://docs.pydantic.dev/latest/concepts/fields/"],
        )
        self.assertEqual(
            result["hits"][0]["evidence"]["element"]["metadata"]["validated_url"],
            "https://pydantic.dev/docs/validation/latest/concepts/fields/",
        )
        self.assertEqual(result["diagnostics"]["validated_url_count"], 1)

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_rejects_redirects_outside_allowed_docs(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {
            "results": [
                {
                    "url": "https://docs.pydantic.dev/latest/concepts/fields/",
                    "title": "Fields",
                    "content": "Field customizes model fields.",
                    "score": 0.93,
                }
            ]
        }
        self.mock_validate_doc_url.side_effect = lambda url: DocUrlValidationResult(
            ok=False,
            final_url="https://example.com/docs/fields/",
            status_code=200,
            reason="redirect_policy",
        )

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool(
            query="Pydantic Field official docs",
            include_domains=["docs.pydantic.dev"],
        )

        self.assertEqual(result["diagnostics"]["status"], "no_result")
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["diagnostics"]["filtered_redirect_policy_count"], 1)
        self.assertIn("url_redirect_policy_filtered", result["diagnostics"]["warnings"])

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_rejects_single_identifier_mismatch(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {
            "results": [
                {
                    "url": "https://docs.pydantic.dev/latest/usage/types/uuids/",
                    "title": "Types/uuids",
                    "content": "UUID values are parsed from strings and bytes.",
                    "score": 0.95,
                }
            ]
        }

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool(query="Pydantic v2 Field validation official docs")

        self.assertEqual(result["diagnostics"]["status"], "no_result")
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["diagnostics"]["filtered_identifier_mismatch_count"], 1)
        self.assertIn("identifier_coverage_incomplete", result["diagnostics"]["warnings"])

    def test_exact_identifier_coverage_ignores_trailing_dot_tokens(self) -> None:
        self.assertEqual(
            extract_exact_identifier_terms("pandas. concat official docs", library_name="pandas"),
            [],
        )
        self.assertTrue(
            has_exact_identifier_coverage(
                "pandas. DataFrame. merge official docs",
                [
                    self._hit(
                        title="pandas.DataFrame.merge",
                        url="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html",
                        content="Merge DataFrame objects.",
                    )
                ],
                library_name="pandas",
            )
        )
        self.assertEqual(
            extract_exact_identifier_terms("Standard. Scaler official docs", library_name="scikit-learn"),
            ["StandardScaler"],
        )
        self.assertTrue(
            has_exact_identifier_coverage(
                "Standard. Scaler official docs",
                [
                    self._hit(
                        title="StandardScaler",
                        url="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html",
                        content="Standardize features.",
                    )
                ],
                library_name="scikit-learn",
            )
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
        result = registry.tavily_search_tool(query="numpy official docs")

        self.assertGreaterEqual(len(mock_request_tavily_search.call_args_list), 2)
        self.assertEqual(
            [item["evidence"]["snapshot"]["source_uri"] for item in result["hits"]],
            ["https://numpy.org/doc/stable/user/basics.broadcasting.html"],
        )
        self.assertIn("Broadcasting stretches compatible array dimensions.", SearchHit.model_validate(result["hits"][0]).evidence.excerpt)

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
        result = registry.tavily_search_tool(query="BeautifulSoup으로 특정 태그 찾는 예제를 보여줘")

        first_kwargs = mock_request_tavily_search.call_args_list[0].kwargs
        self.assertEqual(first_kwargs["include_domains"], ["crummy.com"])
        self.assertEqual(result["diagnostics"]["status"], "success")
        self.assertEqual(
            [item["evidence"]["snapshot"]["source_uri"] for item in result["hits"]],
            ["https://www.crummy.com/software/BeautifulSoup/bs4/doc/#searching-the-tree"],
        )

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_extracts_structured_api_metadata_from_raw_content(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {
            "results": [
                {
                    "url": "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html",
                    "title": "matplotlib.pyplot.pie — Matplotlib documentation",
                    "content": "Plot a pie chart.",
                    "raw_content": "\n".join(
                        [
                            "# matplotlib.pyplot.pie",
                            "matplotlib.pyplot.pie(x, *, labels=None, autopct=None, startangle=0, wedgeprops=None)",
                            "Plot a pie chart.",
                            "Parameters:",
                            "x 1D array-like",
                            "The wedge sizes.",
                            "labels list, default: None",
                            "A sequence of strings providing the labels for each wedge.",
                            "autopct None or str or callable, default: None",
                            "If not None, autopct is used to label the wedges with their numeric value.",
                            "startangle float, default: 0 degrees",
                            "The angle by which the start of the pie is rotated.",
                            "wedgeprops dict, default: None",
                            "Dict of arguments passed to each Wedge of the pie.",
                            "Returns:",
                            "patches list",
                            "A sequence of Wedge instances.",
                        ]
                    ),
                    "score": 0.95,
                }
            ]
        }

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool(query="matplotlib pie 차트 옵션을 정리해줘")

        first_kwargs = mock_request_tavily_search.call_args_list[0].kwargs
        self.assertEqual(first_kwargs["include_raw_content"], "markdown")
        self.assertTrue(result["diagnostics"]["include_raw_content_requested"])
        self.assertEqual(result["diagnostics"]["status"], "success")
        metadata = result["hits"][0]["evidence"]["element"]["metadata"]["doc_metadata"]
        self.assertEqual(metadata["doc_family"], "sphinx_api")
        self.assertEqual(metadata["symbol"], "matplotlib.pyplot.pie")
        parameter_names = [item["name"] for item in metadata["parameters"]]
        self.assertIn("labels", parameter_names)
        self.assertIn("autopct", parameter_names)
        self.assertIn("wedgeprops", parameter_names)
        self.assertIn("autopct None or str or callable", SearchHit.model_validate(result["hits"][0]).evidence.excerpt)

    @patch("src.infra.tools.docs_search.client.request_tavily_search")
    def test_docs_search_prefers_api_reference_for_matplotlib_option_requests(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {
            "results": [
                {
                    "url": "https://matplotlib.org/stable/plot_types/stats/pie.html",
                    "title": "pie(x) — Matplotlib documentation",
                    "content": "Plot a pie chart. See pie.",
                    "score": 0.99,
                },
                {
                    "url": "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html",
                    "title": "matplotlib.pyplot.pie — Matplotlib documentation",
                    "content": "Plot a pie chart.",
                    "raw_content": "\n".join(
                        [
                            "# matplotlib.pyplot.pie",
                            "matplotlib.pyplot.pie(x, *, labels=None, autopct=None, startangle=0, wedgeprops=None)",
                            "Parameters:",
                            "x 1D array-like",
                            "The wedge sizes.",
                            "autopct None or str or callable, default: None",
                            "If not None, autopct is used to label the wedges with their numeric value.",
                            "wedgeprops dict, default: None",
                            "Dict of arguments passed to each Wedge of the pie.",
                        ]
                    ),
                    "score": 0.82,
                },
            ]
        }

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool(query="matplotlib pie 차트 옵션을 정리해줘")

        first_kwargs = mock_request_tavily_search.call_args_list[0].kwargs
        self.assertEqual(first_kwargs["include_raw_content"], "markdown")
        self.assertEqual(
            result["hits"][0]["evidence"]["snapshot"]["source_uri"],
            "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html",
        )
        self.assertIn("autopct", SearchHit.model_validate(result["hits"][0]).evidence.excerpt)


class TavilyClientTest(unittest.TestCase):
    def _response(self, *, status_code: int = 200, body: object) -> Mock:
        response = Mock()
        response.status_code = status_code
        response.json.return_value = body
        return response

    @patch("src.infra.tools.docs_search.client.requests.post")
    def test_tavily_client_returns_valid_results_after_one_provider_request(self, mock_post) -> None:
        provider_body = {"results": [{"url": "https://numpy.org/doc/stable/"}]}
        mock_post.return_value = self._response(body=provider_body)

        result = request_tavily_search(
            query="numpy documentation",
            tavily_api_key="test-key",
            include_domains=["numpy.org"],
            search_depth="basic",
            timeout_seconds=5,
        )

        self.assertEqual(result, provider_body)
        self.assertEqual(len(mock_post.call_args_list), 1)
        self.assertEqual(mock_post.call_args.kwargs["timeout"], 5)
        self.assertEqual(mock_post.call_args.kwargs["json"]["query"], "numpy documentation")

    @patch("src.infra.tools.docs_search.client.requests.post")
    def test_tavily_client_rejects_invalid_json_after_one_provider_request(self, mock_post) -> None:
        response = self._response(body={})
        response.json.side_effect = ValueError("invalid json")
        mock_post.return_value = response

        with self.assertRaisesRegex(RuntimeError, "invalid JSON response"):
            request_tavily_search(
                query="numpy documentation",
                tavily_api_key="test-key",
                include_domains=["numpy.org"],
                search_depth="basic",
                timeout_seconds=5,
            )

        self.assertEqual(len(mock_post.call_args_list), 1)


if __name__ == "__main__":
    unittest.main()
