"""Source-owned HTML identities survive flattening without entering the excerpt."""
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests

from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.infra.settings import AppSettings
from src.infra.tools.docs_search.official_page import _capture_page
from src.infra.tools.docs_search.sections import parse_doc_sections
from src.infra.tools.docs_search.tool import build_docs_search_tool
from src.runtime.nodes.synthesis.budgets import RetrievedEvidenceBudget
from src.runtime.nodes.synthesis.evidence_selection import prepare_evidence_candidates
from src.runtime.source_scope import documentation_declaration_scope
from tests.tools.test_official_inventory import inventory, response


CAPTURED_DEFINITIONS = (Path(__file__).parents[1] / "fixtures" / "official_builtin_methods.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("symbol, declaration, behavior", [
    ("list.sort", "sort(*, key=None, reverse=False)¶", "This method sorts the list in place"),
    ("dict.update", "update(**kwargs)¶", "Update the dictionary"),
    ("str.removeprefix", "str.removeprefix(prefix, /)¶", "If the string starts with the prefix string"),
])
def test_official_definition_owner_and_offsets_survive_plain_text_capture(monkeypatch, symbol, declaration, behavior):
    def get(url, **kwargs):
        return response(url, inventory([f"{symbol} py:method 1 builtins/stdtypes.html#$ -"])
                        if url.endswith("objects.inv") else CAPTURED_DEFINITIONS)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, '{"results": []}'))
    task = RetrievalTask(route="docs", query=f"Python {symbol} behavior", k=1,
                         requirement={"library": "Python", "symbols": [symbol], "evidence_needs": ["explanation"]})
    result = build_docs_search_tool(AppSettings(_env_file=None, tavily_api_key="test"))(
        task.query, requirement=task.requirement, k=task.k,
    )

    evidence = result.hits[0].evidence
    scope = documentation_declaration_scope(evidence)
    matched = next(item for item in scope["declarations"] if item["name"] == symbol)
    assert evidence.element.text[matched["start"]:matched["end"]] == declaration
    assert behavior in evidence.excerpt
    assert "Link to this definition" not in evidence.element.text
    assert "href=" not in evidence.element.text
    assert "](#" not in evidence.element.text
    prepared = prepare_evidence_candidates(
        user_input=task.query,
        hits=[hit.model_copy(update={"requirement_id": task.requirement_id}) for hit in result.hits],
        planner_output=PlannerOutput(use_retrieval=True, tasks=[task]),
        budget=RetrievedEvidenceBudget(500, 1000, 2),
    )
    assert any(behavior in hit.evidence.excerpt for hit in prepared)
    for hit in prepared:
        item = hit.evidence
        assert item.excerpt == item.element.text[item.selection.start:item.selection.end]


@pytest.mark.parametrize("symbol, behavior", [
    ("list.sort", "This method sorts the list in place"),
    ("dict.update", "Update the dictionary"),
    ("str.removeprefix", "If the string starts with the prefix string"),
])
def test_html_definition_precedes_incidental_mentions_in_a_long_reference(monkeypatch, symbol, behavior):
    unrelated = "".join(
        f"<dl><dt>other_{index}()¶</dt><dd><p>{'Unrelated operations and parameters. ' * 100}</p></dd></dl>"
        for index in range(12)
    )
    page = CAPTURED_DEFINITIONS.replace("<main>", "<main>" + unrelated).replace(
        "</main>", f"<p>{symbol} is mentioned by an unrelated operation.</p></main>",
    )
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: response(
        url, inventory([f"{symbol} py:method 1 builtins/stdtypes.html#$ -"])
        if url.endswith("objects.inv") else page,
    ))
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, '{"results": []}'))
    task = RetrievalTask(route="docs", query=f"Python {symbol} behavior", k=1,
                         requirement={"library": "Python", "symbols": [symbol], "evidence_needs": ["explanation"]})

    result = build_docs_search_tool(AppSettings(_env_file=None, tavily_api_key="test"))(
        task.query, requirement=task.requirement, k=task.k,
    )

    assert behavior in result.hits[0].evidence.excerpt
    assert result.hits[0].evidence.selection.start > 40000
    prepared = prepare_evidence_candidates(
        user_input=task.query,
        hits=[hit.model_copy(update={"requirement_id": task.requirement_id}) for hit in result.hits],
        planner_output=PlannerOutput(use_retrieval=True, tasks=[task]),
        budget=RetrievedEvidenceBudget(500, 1000, 2),
    )
    assert any(behavior in hit.evidence.excerpt for hit in prepared)


def test_identical_short_signatures_keep_their_source_owners_and_exact_display_text(monkeypatch):
    page = (
        '<title>Built-in Types</title><main>'
        '<dl><dt id="First.method">method(value)¶</dt><dd><p>First body.</p></dd></dl>'
        '<dl><dt id="Second.method">method(value)¶</dt><dd><p>Second body.</p></dd></dl></main>'
    )
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: response(
        url, inventory(["First.method py:method 1 builtins/stdtypes.html#$ -"])
        if url.endswith("objects.inv") else page,
    ))
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, '{"results": []}'))
    task = RetrievalTask(route="docs", query="First.method behavior", k=1,
                         requirement={"library": "Python", "symbols": ["First.method"]})

    result = build_docs_search_tool(AppSettings(_env_file=None, tavily_api_key="test"))(
        task.query, requirement=task.requirement,
    )

    evidence = result.hits[0].evidence
    expected_text = "method(value)¶\n\nFirst body.\n\nmethod(value)¶\n\nSecond body."
    assert evidence.element.text == expected_text
    assert [(item["name"], expected_text[item["start"]:item["end"]])
            for item in documentation_declaration_scope(evidence)["declarations"]] == [
        ("First.method", "method(value)¶"), ("Second.method", "method(value)¶"),
    ]


@pytest.mark.parametrize("url, owner, title, definition", [
    (
        "https://docs.python.org/3/library/ast.html", "ast", "ast — Python documentation",
        '<dt class="sig sig-object std" id="cmdoption-ast-mode">'
        '<span class="sig-name descname"><span class="pre">--mode</span></span>'
        '<span class="sig-prename descclassname"> <span class="pre">&lt;mode&gt;</span></span>'
        '<a class="headerlink" href="#cmdoption-ast-mode" title="Link to this definition">¶</a></dt>'
        '<dd><p>Specify what kind of code must be compiled, like the <em>mode</em> argument in parse().</p></dd>',
    ),
    (
        "https://git-scm.com/docs/git-rebase", "git-rebase", "Git - git-rebase Documentation",
        '<dt class="hdlist1" id="Documentation/git-rebase.txt---ontoltnewbasegt">'
        '<a id="Documentation/git-rebase.txt---ontonewbase" class="anchor" '
        'href="#Documentation/git-rebase.txt---ontonewbase"></a>--onto &lt;newbase&gt; </dt>'
        '<dd><p>Starting point at which to create the new commits.</p></dd>',
    ),
], ids=["python-ast-cli", "git-rebase-options"])
def test_cli_option_anchor_ids_keep_the_existing_document_owner(monkeypatch, url, owner, title, definition):
    # The definition markup comes from the actual API page/manual. Its option
    # anchor identifies a link target, not a Python API declaration.
    page = f"<title>{title}</title><main><dl>{definition}</dl></main>"
    monkeypatch.setattr(requests, "get", lambda target, **kwargs: response(target, page))

    evidence = _capture_page(url, timeout_seconds=10, allowed_domains={urlparse(url).hostname})
    sections = parse_doc_sections(
        evidence.element.text, title=evidence.snapshot.title, url=evidence.snapshot.source_uri,
        definition_owners=evidence.element.metadata["definition_owners"],
    )
    assert {section.owner for section in sections} == {owner}
    assert evidence.element.metadata["definition_owners"] == {}
