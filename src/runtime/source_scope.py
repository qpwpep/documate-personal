"""Parser-derived scope shared by synthesis and semantic validation prompts."""
from __future__ import annotations

import ast
from functools import lru_cache
from typing import cast
from urllib.parse import urlparse

from src.core.evidence import EvidenceRef
from src.infra.tools.docs_search.sections import (
    DocDeclaration, DocSection, clean_markup, definition_sections, parse_doc_sections, scan_declarations,
)


def documentation_declaration_scope(evidence: EvidenceRef) -> dict[str, object] | None:
    """Describe official Python ownership and visible syntax without deciding API support."""
    if (evidence.route != "docs" or evidence.snapshot.parser != "official_html"
            or urlparse(evidence.snapshot.source_uri).hostname != "docs.python.org"):
        return None
    element, selection = evidence.element, evidence.selection
    declarations, sections, source_definitions = _document_structure(
        element.text, evidence.snapshot.title, evidence.snapshot.source_uri,
        tuple(sorted((element.metadata.get("definition_owners") or {}).items())),
    )
    definitions = {section.start: section for section in source_definitions}
    selection_end = len(element.text) if selection.end is None else selection.end
    api_owners = []
    defined_owner = None
    for section in sections:
        if section.start in definitions:
            defined_owner = section.owner
        elif section.owner != defined_owner:
            defined_owner = None
        if defined_owner and section.start < selection_end and selection.start < section.end:
            api_owners.append({"name": defined_owner, "section_start": section.start, "section_end": section.end})
    selected = []
    for declaration in declarations:
        if declaration.start >= selection_end or declaration.end <= selection.start:
            continue
        # Match the actual self-linked definition, not a call inside its prose
        # or fenced examples. Owner context alone never establishes a signature.
        section = definitions.get(element.text.rfind("\n", 0, declaration.start) + 1)
        if section is None:
            continue
        original = element.text[declaration.start:declaration.end]
        fully_selected = selection.start <= declaration.start and selection_end >= declaration.end
        record = {"name": section.owner, "start": declaration.start, "end": declaration.end,
                  "fully_selected": fully_selected}
        if fully_selected:
            signature = clean_markup(original).rstrip(" ¶")
            try:
                # The controlled prefix always produces a FunctionDef. Parsing
                # never executes defaults, annotations, or document content.
                function = cast(ast.FunctionDef, ast.parse(
                    "def _documented" + signature[signature.index("("):] + ":\n    pass\n",
                ).body[0])
            except (SyntaxError, ValueError):
                pass
            else:
                if not any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(function)):
                    record.update({"keyword_parameters": [argument.arg for argument in (
                        *function.args.args, *function.args.kwonlyargs,
                    )], "accepts_arbitrary_keywords": function.args.kwarg is not None})
        selected.append(record)
    return {"kind": "documentation_declarations", "declarations": selected,
            "api_owners": api_owners} if selected or api_owners else None


@lru_cache(maxsize=8)
def _document_structure(
    text: str, title: str, url: str, definition_owners: tuple[tuple[str, str], ...],
) -> tuple[tuple[DocDeclaration, ...], tuple[DocSection, ...], tuple[DocSection, ...]]:
    # Several evidence ranges and validation predicates share one captured
    # page. Retain only eight immutable parses; selection-specific scopes stay
    # fresh, and changed text or captured owners produce a different entry.
    declarations = scan_declarations(text)
    sections = parse_doc_sections(
        text, title=title, url=url, declarations=declarations, definition_owners=dict(definition_owners),
    )
    definitions = definition_sections(text, declarations=declarations, sections=sections)
    return tuple(declarations), tuple(sections), tuple(definitions)
