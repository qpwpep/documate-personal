from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.rules import load_rules_config


DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "src/infra/config/agent_rules.toml"


def test_load_default_rules_config() -> None:
    rules = load_rules_config(str(DEFAULT_RULES_PATH))

    assert rules.docs_search.query_hints
    assert rules.docs_search.allowed_doc_path_prefixes["numpy.org"] == ["/doc/stable/"]
    assert rules.docs_search.docs_identifier_stopwords == [
        "from", "official", "docs", "doc", "documentation", "reference", "with", "the", "it", "and",
    ]


@pytest.mark.parametrize(
    ("section", "expected_location"),
    [
        (None, ("unknown_setting",)),
        ("[docs_search]", ("docs_search", "unknown_setting")),
        ("[[docs_search.query_hints]]", ("docs_search", "query_hints", 0, "unknown_setting")),
    ],
    ids=["root", "docs-search", "query-hint"],
)
def test_load_rules_config_rejects_unknown_keys(tmp_path, section, expected_location) -> None:
    payload = DEFAULT_RULES_PATH.read_text(encoding="utf-8")
    extra = "unknown_setting = true\n"
    if section is None:
        payload = extra + payload
    else:
        payload = payload.replace(section + "\n", section + "\n" + extra, 1)
    rules_path = tmp_path / "agent_rules.toml"
    rules_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValidationError) as error:
        load_rules_config(str(rules_path))

    assert any(
        item["loc"] == expected_location and item["type"] == "extra_forbidden"
        for item in error.value.errors()
    )


@pytest.mark.parametrize(
    ("payload", "expected_location"),
    [
        (
            '[intents]\ndocs_patterns = ["docs"]\n[docs_search]\n',
            ("intents",),
        ),
        (
            '[planner]\ncompare_clause_pattern = "compare"\n[docs_search]\n',
            ("planner",),
        ),
        (
            '[planner]\ndocs_identifier_stopwords = ["docs"]\n[docs_search]\n',
            ("planner",),
        ),
        (
            '[docs_search]\ncompare_clause_pattern = "compare"\n',
            ("docs_search", "compare_clause_pattern"),
        ),
        (
            '[docs_search]\n[[docs_search.query_hints]]\nlibrary_name = "Python"\n'
            'fallback_queries = ["python docs"]\n',
            ("docs_search", "query_hints", 0, "fallback_queries"),
        ),
    ],
    ids=["intents", "planner-pattern", "planner-stopwords", "comparison-pattern", "fallback-queries"],
)
def test_load_rules_config_rejects_retired_keys(tmp_path, payload, expected_location) -> None:
    rules_path = tmp_path / "agent_rules.toml"
    rules_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValidationError) as error:
        load_rules_config(str(rules_path))

    assert any(
        item["loc"] == expected_location and item["type"] == "extra_forbidden"
        for item in error.value.errors()
    )


def test_load_rules_config_accepts_docs_search_settings(tmp_path) -> None:
    rules_path = tmp_path / "agent_rules.toml"
    rules_path.write_text(
        '[docs_search]\ndocs_identifier_stopwords = ["example"]\n'
        'error_page_markers = ["missing page"]\n'
        '[docs_search.allowed_doc_path_prefixes]\n"docs.example.org" = ["/api/"]\n'
        '[[docs_search.query_hints]]\nidentifiers = ["example"]\nlibrary_name = "Example"\n'
        'domains = ["docs.example.org"]\nmatch_mode = "word"\n',
        encoding="utf-8",
    )

    rules = load_rules_config(str(rules_path))

    assert rules.docs_search.docs_identifier_stopwords == ["example"]
    assert rules.docs_search.error_page_markers == ["missing page"]
    assert rules.docs_search.allowed_doc_path_prefixes == {"docs.example.org": ["/api/"]}
    hint = rules.docs_search.query_hints[0]
    assert (hint.identifiers, hint.library_name, hint.domains, hint.match_mode) == (
        ["example"], "Example", ["docs.example.org"], "word",
    )
