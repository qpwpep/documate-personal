"""Resolve explicit code requirements against preserved source, independently of vector rank."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Literal

from src.core.documents import DocumentElement, DocumentSnapshot, ParsedDocument
from src.core.evidence import EvidenceRef, RetrievalScore, SearchHit, build_evidence, dedupe_search_hits
from src.core.planner_schema import RetrievalRequirement


_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
_DEFINITION_WORDS = r"(?:function|method|class|함수|메서드|클래스)"
_EXTRACTION_WORDS = re.compile(r"extract|quote|verbatim|definition|implementation|발췌|추출|정의|구현|원문|그대로", re.I)
_INSTRUCTION_WORDS = {"a", "an", "the", "extract", "quote", "verbatim", "definition", "implementation",
                      "function", "method", "class", "named", "code", "source", "of"}


def infer_legacy_requirement(query: str) -> RetrievalRequirement | None:
    """Preserve the direct tool API for explicit named-definition requests.

    General prose and library usage queries retain semantic search. Only a code
    name adjacent to a definition noun, or a quoted name in a definition request,
    supplies a deterministic requirement; arbitrary English words are not names.
    """
    if not _EXTRACTION_WORDS.search(query):
        return None
    symbols: list[str] = []
    patterns = (
        rf"(?<![A-Za-z0-9_])({_IDENTIFIER})(?:\(\))?\s*(?:이라는|라는)?\s*{_DEFINITION_WORDS}(?![A-Za-z])",
        rf"\b(?:function|method|class)\s+(?:named\s+)?[`'\"]?({_IDENTIFIER})(?![A-Za-z0-9_])",
        rf"(?:함수|메서드|클래스)\s+[`'\"]?({_IDENTIFIER})(?![A-Za-z0-9_])",
    )
    for pattern in patterns:
        symbols.extend(re.findall(pattern, query, flags=re.I))
    if re.search(rf"definition|implementation|정의|구현|{_DEFINITION_WORDS}", query, re.I):
        symbols.extend(re.findall(rf"[`'\"]({_IDENTIFIER})[`'\"]", query))
    symbols = list(dict.fromkeys(name for name in symbols if name.lower() not in _INSTRUCTION_WORDS))
    return RetrievalRequirement(symbols=symbols, match="definition") if symbols else None


@dataclass(frozen=True)
class SourceRequirementResult:
    hits: list[SearchHit]
    answerability: Literal["covered", "partial", "missing", "unknown"]
    missing_requirements: list[str]
    candidate_count: int
    warnings: list[str]


@dataclass(frozen=True)
class _Source:
    snapshot: DocumentSnapshot
    element: DocumentElement


@dataclass(frozen=True)
class _Occurrence:
    name: str
    node: ast.AST
    definition: bool
    priority: int
    canonical_name: str = ""


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self, tree: ast.AST, aliases: dict[str, str]):
        self.occurrences: list[_Occurrence] = []
        self.scope: list[str] = []
        self.parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        self.aliases = dict(aliases)

    def _reference(self, name: str, node: ast.AST, priority: int) -> None:
        root, separator, suffix = name.partition(".")
        canonical = self.aliases.get(root, root) + (separator + suffix if separator else "")
        self.occurrences.append(_Occurrence(name, self._statement(node), False, priority, canonical))

    def _statement(self, node: ast.AST) -> ast.AST:
        while not isinstance(node, ast.stmt) and node in self.parents:
            node = self.parents[node]
        return node

    def _definition(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self.occurrences.append(_Occurrence(".".join([*self.scope, node.name]), node, True, 1))
        outer_aliases = dict(self.aliases)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()
        self.aliases = outer_aliases

    visit_FunctionDef = _definition
    visit_AsyncFunctionDef = _definition
    visit_ClassDef = _definition

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self.aliases[name] = alias.name if alias.asname else name
            self.occurrences.append(_Occurrence(alias.name, node, False, 3))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            qualified = ".".join(part for part in (node.module, alias.name) if part)
            self.aliases[name] = qualified
            self.occurrences.append(_Occurrence(qualified, node, False, 3))

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted_name(node.func)
        if name:
            self._reference(name, node, 0)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self._reference(node.id, node, 2)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        name = _dotted_name(node)
        if name:
            self._reference(name, node, 2)
        self.generic_visit(node)


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _matches(requested: str, occurrence: _Occurrence) -> bool:
    names = [occurrence.name, occurrence.canonical_name]
    if "." not in requested:
        return any(name.rsplit(".", 1)[-1] == requested for name in names)
    return any(name == requested or (occurrence.definition and name.endswith("." + requested)) for name in names)


def _node_range(source: str, node: ast.AST, *, definition: bool) -> tuple[int, int]:
    """AST columns are UTF-8 byte offsets; evidence offsets count Python characters."""
    lines = source.splitlines(keepends=True)
    start_line = int(node.lineno)
    if definition:
        decorators = getattr(node, "decorator_list", [])
        start_line = min([start_line, *(int(item.lineno) for item in decorators)])
    end_line = int(node.end_lineno)
    start = sum(len(line) for line in lines[:start_line - 1])
    # Include indentation and decorators for a standalone source quotation.
    end = sum(len(line) for line in lines[:end_line - 1])
    end += len(lines[end_line - 1].encode("utf-8")[:int(node.end_col_offset)].decode("utf-8"))
    return start, end


def _code_terms(node: ast.AST) -> set[str]:
    """Concrete code requirements cannot be fulfilled by comments or string prose."""
    terms: set[str] = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name):
            terms.add(item.id)
        elif isinstance(item, ast.Attribute):
            terms.update((item.attr, _dotted_name(item)))
        elif isinstance(item, (ast.arg, ast.keyword)) and item.arg:
            terms.add(item.arg)
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            terms.add(item.name)
    return terms


def _candidate_sources(rows: list[tuple[Any, float | None]]) -> list[_Source]:
    sources: dict[tuple[str, str], _Source] = {}
    for doc, _score in rows:
        try:
            ref = EvidenceRef.model_validate_json(doc.metadata["evidence_ref"])
            if ref.route != "upload" or ref.excerpt != doc.page_content:
                continue
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        sources[(ref.snapshot.snapshot_id, ref.element.element_id)] = _Source(ref.snapshot, ref.element)
    return list(sources.values())


def resolve_source_requirement(
    *, requirement: RetrievalRequirement, source_document: ParsedDocument | None = None,
    candidate_rows: list[tuple[Any, float | None]],
    source_documents: tuple[ParsedDocument, ...] | None = None,
) -> SourceRequirementResult:
    """A complete registry proves absence; vector candidates alone never do."""
    documents = source_documents if source_documents is not None else ((source_document,) if source_document else None)
    if documents is not None and requirement.file_ids:
        documents = tuple(document for document in documents
                          if any(element.metadata.get("file_id") in requirement.file_ids
                                 for element in document.elements))
    exhaustive = bool(documents) and all(document.snapshot.capture_scope == "full_document" for document in documents)
    sources = ([_Source(document.snapshot, element) for document in documents for element in document.elements]
               if documents is not None else _candidate_sources(candidate_rows))
    if requirement.file_ids:
        sources = [source for source in sources if source.element.metadata.get("file_id") in requirement.file_ids]
    symbols = list(dict.fromkeys(requirement.symbols))
    by_symbol: dict[str, list[tuple[int, _Source, _Occurrence, tuple[int, int]]]] = {name: [] for name in symbols}
    warnings: list[str] = []
    aliases: dict[str, str] = {}
    alias_snapshot: str | None = None
    for source in sources:
        if source.snapshot.snapshot_id != alias_snapshot:
            # Notebook cells share imports; separate files never share lexical scope.
            aliases = {}
            alias_snapshot = source.snapshot.snapshot_id
        if source.element.kind != "code":
            continue
        try:
            tree = ast.parse(source.element.text)
        except (SyntaxError, ValueError):
            exhaustive = False
            warnings.append("source_symbol_analysis_unavailable")
            continue
        visitor = _SymbolVisitor(tree, aliases)
        visitor.visit(tree)
        aliases = visitor.aliases
        for symbol in symbols:
            for occurrence in visitor.occurrences:
                if requirement.match == "definition" and not occurrence.definition:
                    continue
                if _matches(symbol, occurrence):
                    bounds = _node_range(source.element.text, occurrence.node, definition=occurrence.definition)
                    by_symbol[symbol].append((occurrence.priority, source, occurrence, bounds))

    hits: list[SearchHit] = []
    covered_aspects: set[str] = set()
    aspects_by_file: dict[str, set[str]] = {file_id: set() for file_id in requirement.file_ids}
    for matches in by_symbol.values():
        if not matches:
            continue
        # Explicit file comparisons keep the best occurrence from each file.
        scoped = bool(requirement.file_ids)
        preferred_by_file = {}
        for priority, source, _occurrence, _bounds in matches:
            key = source.snapshot.snapshot_id if scoped else "all"
            preferred_by_file[key] = min(priority, preferred_by_file.get(key, priority))
        for priority, source, occurrence, (start, end) in matches:
            if priority != preferred_by_file[source.snapshot.snapshot_id if scoped else "all"]:
                continue
            evidence = build_evidence(snapshot=source.snapshot, element=source.element, start=start, end=end)
            hits.append(SearchHit(evidence=evidence, score=RetrievalScore(metric="source_symbol", direction="higher"), rank=len(hits) + 1))
            terms = _code_terms(occurrence.node)
            covered_aspects.update(terms)
            if source.element.metadata.get("file_id") in aspects_by_file:
                aspects_by_file[source.element.metadata["file_id"]].update(terms)
    hits = [hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(dedupe_search_hits(hits), start=1)]
    unresolved = [name for name, matches in by_symbol.items() if not matches]
    if requirement.file_ids:
        unresolved = [f"file:{file_id}:{name}" for file_id in requirement.file_ids
                      for name, matches in by_symbol.items()
                      if not any(source.element.metadata.get("file_id") == file_id
                                 for _priority, source, _occurrence, _bounds in matches)]
    if not unresolved:
        answerability = "covered"
    elif not exhaustive:
        answerability = "unknown"
    else:
        answerability = "partial" if hits else "missing"
    aspects_unverified = any(aspect not in terms for terms in (list(aspects_by_file.values()) or [covered_aspects])
                             for aspect in requirement.aspects)
    if answerability == "covered" and (requirement.version or aspects_unverified):
        answerability = "unknown"
        warnings.append("source_requirement_constraints_unverified")
    return SourceRequirementResult(
        hits=hits,
        answerability=answerability,
        missing_requirements=unresolved if exhaustive else [],
        candidate_count=len(candidate_rows) if documents is None else len(hits),
        warnings=list(dict.fromkeys(warnings)),
    )
