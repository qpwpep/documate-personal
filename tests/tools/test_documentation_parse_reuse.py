"""Repeated evidence ranges reuse document parsing without sharing their scopes."""
import cProfile

from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import build_evidence
from src.infra.tools.docs_search.sections import clean_markup, parse_doc_sections, scan_declarations
from src.runtime.source_scope import documentation_declaration_scope


def test_complete_single_line_declarations_do_not_scan_following_prose():
    text = "".join(f"method_{index}(value, *, flag=True)¶\nDescription {index}.\n" for index in range(100))

    with cProfile.Profile() as profile:
        declarations = scan_declarations(text)

    assert [item.name for item in declarations] == [f"method_{index}" for index in range(100)]
    assert all(text[item.start:item.end] == f"{item.name}(value, *, flag=True)¶" for item in declarations)
    cleaned_lines = sum(item.callcount for item in profile.getstats() if item.code is clean_markup.__code__)
    assert cleaned_lines <= len(text.splitlines()) * 2


def test_documentation_scopes_parse_one_document_for_distinct_evidence_ranges():
    text = "alpha()¶\nAlpha behavior.\n\nbeta(flag=True)¶\nBeta behavior.\n"
    snapshot = build_snapshot(
        source_uri="https://docs.python.org/3/library/reuse_example.html", title="reuse_example",
        media_type="text/html", source_type="official", content=text, parser="official_html", parser_version="2",
    )
    element = DocumentElement(element_id="body", kind="paragraph", text=text)
    beta_start = text.index("beta(")
    ranges = [
        build_evidence(snapshot=snapshot, element=element, end=beta_start),
        build_evidence(snapshot=snapshot, element=element, start=beta_start),
        build_evidence(snapshot=snapshot, element=element, start=beta_start, end=beta_start + 8),
    ]

    with cProfile.Profile() as profile:
        scopes = [documentation_declaration_scope(evidence) for evidence in ranges * 4]

    assert [scope["declarations"][0]["name"] for scope in scopes[:3]] == [
        "reuse_example.alpha", "reuse_example.beta", "reuse_example.beta",
    ]
    assert [scope["declarations"][0]["fully_selected"] for scope in scopes[:3]] == [True, True, False]
    assert scopes[0]["declarations"][0]["keyword_parameters"] == []
    assert scopes[1]["declarations"][0]["keyword_parameters"] == ["flag"]
    assert "keyword_parameters" not in scopes[2]["declarations"][0]
    parses = sum(item.callcount for item in profile.getstats() if item.code is parse_doc_sections.__code__)
    assert parses == 1


def test_documentation_scope_keeps_definition_owners_and_returned_values_independent():
    text = "method(value)¶\nA method.\n"
    snapshot = build_snapshot(
        source_uri="https://docs.python.org/3/library/scope_owners.html", title="scope_owners",
        media_type="text/html", source_type="official", content=text, parser="official_html", parser_version="2",
    )
    evidence = [build_evidence(
        snapshot=snapshot, element=DocumentElement(
            element_id="body", kind="paragraph", text=text, metadata={"definition_owners": {"0": owner}},
        ),
    ) for owner in ("First.method", "Second.method")]

    first = documentation_declaration_scope(evidence[0])
    second = documentation_declaration_scope(evidence[1])
    first["declarations"][0]["name"] = "changed by a caller"
    repeated = documentation_declaration_scope(evidence[0])

    assert repeated["declarations"][0]["name"] == "First.method"
    assert second["declarations"][0]["name"] == "Second.method"
