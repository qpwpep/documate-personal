"""Resolve explicit code requirements against preserved source, independently of vector rank."""

from __future__ import annotations

import ast
import io
import symtable
import tokenize
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from src.core.documents import DocumentElement, DocumentSnapshot, ParsedDocument
from src.core.evidence import RetrievalScore, SearchHit, build_evidence, dedupe_search_hits
from src.core.literal_matching import literal_aspect_present
from src.core.planner_schema import RetrievalRequirement
from src.core.python_source import PythonSourceMap


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

    @cached_property
    def source_map(self) -> PythonSourceMap:
        return PythonSourceMap(self.element.text)

    @cached_property
    def code_tokens(self) -> list[tuple[tuple[int, str], int, int]]:
        # Token columns count characters, unlike the AST's UTF-8 byte columns.
        return [((token.type, token.string),
                 self.source_map.offsets[token.start[0] - 1] + token.start[1],
                 self.source_map.offsets[token.end[0] - 1] + token.end[1])
                for token in _fragment_tokens(self.element.text)]


@dataclass(frozen=True)
class _Occurrence:
    name: str
    node: ast.AST
    definition: bool
    priority: int
    canonical_name: str = ""


class _SymbolScopeUnavailable(ValueError):
    """The compiler did not expose a scope required for lexical resolution."""


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self, tree: ast.AST, aliases: dict[str, str], symbols: symtable.SymbolTable):
        self.occurrences: list[_Occurrence] = []
        self.scope: list[str] = []
        self.parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        self.aliases = dict(aliases)
        self.module_aliases = self.aliases
        self.symbols = symbols
        self.class_enclosing_aliases: dict[str, str] | None = None
        self.child_scopes: dict[tuple[int, str, int], Iterator[symtable.SymbolTable]] = {}

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
        # Defaults and decorators use the enclosing scope; generic annotations
        # have a separate compiler scope around the function/class body.
        for decorator in node.decorator_list:
            self.visit(decorator)
        if not isinstance(node, ast.ClassDef):
            for default in (*node.args.defaults, *node.args.kw_defaults):
                if default is not None:
                    self.visit(default)
        annotation_scope = (self._nested_scope(node.name, node.lineno, annotation_scope=True)
                            if node.type_params else nullcontext())
        with annotation_scope:
            for parameter in node.type_params:
                self.visit(parameter)
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    self.visit(base)
                for keyword in node.keywords:
                    self.visit(keyword.value)
            else:
                for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
                                 node.args.vararg, node.args.kwarg):
                    if argument is not None and argument.annotation is not None:
                        self.visit(argument.annotation)
                if node.returns is not None:
                    self.visit(node.returns)
            self.aliases.pop(node.name, None)
            with self._nested_scope(node.name, node.lineno, class_scope=isinstance(node, ast.ClassDef)):
                self.scope.append(node.name)
                for statement in node.body:
                    self.visit(statement)
                self.scope.pop()
        self.aliases.pop(node.name, None)

    @contextmanager
    def _nested_scope(
        self, name: str | None, line: int, *, class_scope: bool = False, annotation_scope: bool = False,
    ) -> Iterator[None]:
        outer_aliases = self.aliases
        outer_symbols = self.symbols
        outer_class_aliases = self.class_enclosing_aliases
        self.aliases = dict(outer_aliases if annotation_scope or outer_class_aliases is None else outer_class_aliases)
        if name is not None:
            self.symbols = self._child_scope(name, line)
        if class_scope:
            # Methods skip the class namespace when resolving unqualified names.
            self.class_enclosing_aliases = dict(self.aliases)
        else:
            self.class_enclosing_aliases = None
        if name is not None:
            for symbol in self.symbols.get_symbols():
                symbol_name = symbol.get_name()
                if symbol.is_declared_global():
                    # An explicit global skips every enclosing lexical binding.
                    self.aliases.pop(symbol_name, None)
                    if symbol_name in self.module_aliases:
                        self.aliases[symbol_name] = self.module_aliases[symbol_name]
                elif not class_scope and symbol.is_local():
                    self.aliases.pop(symbol_name, None)
        try:
            yield
        finally:
            self.aliases = outer_aliases
            self.symbols = outer_symbols
            self.class_enclosing_aliases = outer_class_aliases

    def _child_scope(self, name: str, line: int) -> symtable.SymbolTable:
        key = (self.symbols.get_id(), name, line)
        if key not in self.child_scopes:
            # Several lambdas can share a line and name. Consume each compiler
            # scope once in source traversal order rather than reusing the first.
            self.child_scopes[key] = iter(child for child in self.symbols.get_children()
                                          if child.get_name() == name and child.get_lineno() == line)
        child = next(self.child_scopes[key], None)
        if child is None:
            raise _SymbolScopeUnavailable(f"No compiler scope for {name!r} at line {line}")
        return child

    visit_FunctionDef = _definition
    visit_AsyncFunctionDef = _definition
    visit_ClassDef = _definition

    def visit_TypeAlias(self, node: ast.TypeAlias) -> None:
        self.visit(node.name)
        annotation_scope = (self._nested_scope(node.name.id, node.lineno, annotation_scope=True)
                            if node.type_params else nullcontext())
        with annotation_scope:
            for parameter in node.type_params:
                self.visit(parameter)
            with self._nested_scope(node.name.id, node.lineno, annotation_scope=True):
                self.visit(node.value)

    def visit_TypeVar(self, node: ast.TypeVar) -> None:
        if node.bound is not None:
            with self._nested_scope(node.name, node.lineno, annotation_scope=True):
                self.visit(node.bound)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit(node.args)
        with self._nested_scope("lambda", node.lineno):
            self.visit(node.body)

    def _comprehension(self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp) -> None:
        self.visit(node.generators[0].iter)
        # List/set/dict comprehensions are inlined by Python 3.12+, while a
        # generator keeps its own symbol table. Both isolate their target names.
        name = "genexpr" if isinstance(node, ast.GeneratorExp) else None
        with self._nested_scope(name, node.lineno):
            for index, generator in enumerate(node.generators):
                if index:
                    self.visit(generator.iter)
                self.visit(generator.target)
                for condition in generator.ifs:
                    self.visit(condition)
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
            else:
                self.visit(node.elt)

    visit_ListComp = _comprehension
    visit_SetComp = _comprehension
    visit_DictComp = _comprehension
    visit_GeneratorExp = _comprehension

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
        if not isinstance(node.ctx, ast.Load):
            self.aliases.pop(node.id, None)
        self._reference(node.id, node, 2)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)
            self.visit(node.target)
        elif isinstance(node.target, ast.Name):
            self._reference(node.target.id, node.target, 2)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self.visit(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self.visit(node.target)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        self.visit(node.target)
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    visit_AsyncFor = visit_For

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self.aliases.pop(node.name, None)
        for statement in node.body:
            self.visit(statement)
        if node.name is not None:
            self.aliases.pop(node.name, None)

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


def _node_range(source: PythonSourceMap, node: ast.AST, *, definition: bool) -> tuple[int, int]:
    """AST columns are UTF-8 byte offsets; evidence offsets count Python characters."""
    start_line = source.definition_start(node)[0] if definition else node.lineno
    # Include indentation and decorators for a standalone source quotation.
    return source.position(start_line, 0), source.position(node.end_lineno, node.end_col_offset)


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


def _fragment_tokens(text: str) -> list[tokenize.TokenInfo]:
    """Tokenize source fragments without requiring a complete Python statement."""
    tokens = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text, newline=None).readline):
            if token.type not in {tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}:
                tokens.append(token)
    except tokenize.TokenError as exc:
        # An explicitly requested prefix such as `call(` is still a code anchor.
        # An unterminated string must not silently become its preceding tokens.
        if not exc.args[0].endswith("EOF in multi-line statement"):
            return []
    except IndentationError:
        return []
    if tokens and tokens[-1].type == tokenize.NEWLINE:
        tokens.pop()
    return tokens


def _fragment_terms(source: _Source, occurrence: _Occurrence, aspects: list[str]) -> set[str]:
    """Match literal fragments within actual code tokens of one selected node."""
    # Preserve AST-based matching of names, which excludes string prose.
    fragments = [aspect for aspect in aspects if not aspect.isidentifier()]
    if not fragments:
        return set()
    node = occurrence.node
    start_line, start_column = (source.source_map.definition_start(node) if occurrence.definition
                                else (node.lineno, node.col_offset))
    start = source.source_map.position(start_line, start_column)
    end = source.source_map.position(node.end_lineno, node.end_col_offset)
    tokens = [(key, left, right) for key, left, right in source.code_tokens if start <= left and right <= end]
    keys = [key for key, _left, _right in tokens]
    covered = set()
    for aspect in fragments:
        expected = [(token.type, token.string) for token in _fragment_tokens(aspect)]
        if not expected or any(kind in {tokenize.COMMENT, tokenize.ERRORTOKEN} for kind, _text in expected):
            continue
        for index in range(len(keys) - len(expected) + 1):
            if keys[index:index + len(expected)] != expected:
                continue
            left, right = tokens[index][1], tokens[index + len(expected) - 1][2]
            # Check literal spacing within the matched tokens, keeping this
            # occurrence separate from similar text in surrounding comments.
            context = source.element.text[max(start, left - 1):min(end, right + 1)]
            if literal_aspect_present(aspect, context, case_sensitive=True):
                covered.add(aspect)
                break
    return covered


def resolve_source_requirement(
    *, requirement: RetrievalRequirement, source_documents: tuple[ParsedDocument, ...],
) -> SourceRequirementResult:
    """Only a complete, parseable source registry can establish a missing symbol."""
    documents = source_documents
    if requirement.file_ids:
        documents = tuple(document for document in documents
                          if any(element.metadata.get("file_id") in requirement.file_ids
                                 for element in document.elements))
    exhaustive = bool(documents) and all(document.snapshot.capture_scope == "full_document"
                                        and document.snapshot.parser != "docling" for document in documents)
    sources = [_Source(document.snapshot, element) for document in documents for element in document.elements]
    symbols = requirement.symbols
    by_symbol: dict[str, list[tuple[int, _Source, _Occurrence, tuple[int, int]]]] = {name: [] for name in symbols}
    warnings: list[str] = []
    aliases: dict[str, str] = {}
    alias_snapshot: str | None = None
    for source in sources:
        if source.snapshot.parser == "docling":
            exhaustive = False
            warnings.append("source_symbol_analysis_unavailable")
            continue
        if source.snapshot.snapshot_id != alias_snapshot:
            # Notebook cells share imports; separate files never share lexical scope.
            aliases = {}
            alias_snapshot = source.snapshot.snapshot_id
        if source.element.kind != "code":
            continue
        try:
            tree = ast.parse(source.element.text)
            symbols_table = symtable.symtable(source.element.text, "<uploaded source>", "exec")
        except (SyntaxError, ValueError):
            exhaustive = False
            warnings.append("source_symbol_analysis_unavailable")
            continue
        visitor = _SymbolVisitor(tree, aliases, symbols_table)
        try:
            visitor.visit(tree)
        except _SymbolScopeUnavailable:
            exhaustive = False
            warnings.append("source_symbol_analysis_unavailable")
            aliases = {}
            continue
        aliases = visitor.aliases
        source_map = source.source_map
        for symbol in symbols:
            for occurrence in visitor.occurrences:
                if requirement.match == "definition" and not occurrence.definition:
                    continue
                if _matches(symbol, occurrence):
                    bounds = _node_range(source_map, occurrence.node, definition=occurrence.definition)
                    by_symbol[symbol].append((occurrence.priority, source, occurrence, bounds))

    hits: list[SearchHit] = []
    covered_aspects: set[str] = set()
    aspects_by_file: dict[str, set[str]] = {file_id: set() for file_id in requirement.file_ids}
    for matches in by_symbol.values():
        if not matches:
            continue
        # A call and its implementation establish different facts. Keep both
        # roles for symbol searches: preferring the call globally discarded the
        # default values/body needed to explain that actual example's result.
        # Definition-only requests still contain definitions exclusively.
        scoped = bool(requirement.file_ids)
        preferred_by_file = {}
        for priority, source, occurrence, _bounds in matches:
            key = (source.snapshot.snapshot_id if scoped else "all", occurrence.definition)
            preferred_by_file[key] = min(priority, preferred_by_file.get(key, priority))
        for priority, source, occurrence, (start, end) in matches:
            key = (source.snapshot.snapshot_id if scoped else "all", occurrence.definition)
            if priority != preferred_by_file[key]:
                continue
            evidence = build_evidence(snapshot=source.snapshot, element=source.element, start=start, end=end)
            hits.append(SearchHit(evidence=evidence, score=RetrievalScore(metric="source_symbol", direction="higher"), rank=len(hits) + 1))
            terms = _code_terms(occurrence.node)
            terms.update(_fragment_terms(source, occurrence, requirement.aspects))
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
        candidate_count=len(hits),
        warnings=list(dict.fromkeys(warnings)),
    )
