"""Resolve explicit code requirements against preserved source, independently of vector rank."""

from __future__ import annotations

import ast
import io
import symtable
import tokenize
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from src.core.documents import DocumentElement, DocumentSnapshot, ParsedDocument
from src.core.evidence import EvidenceRef, RetrievalScore, SearchHit, build_evidence, dedupe_search_hits
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
class _SourceRange:
    snapshot_id: str
    element_id: str
    start: int
    end: int


@dataclass(frozen=True)
class _Binding:
    canonical_name: str | None
    origin: _SourceRange | None = None


@dataclass(frozen=True)
class _Occurrence:
    name: str
    node: ast.AST
    definition: bool
    priority: int
    canonical_names: frozenset[str | None] = frozenset()
    context_ranges: frozenset[_SourceRange] = frozenset()


class _SymbolScopeUnavailable(ValueError):
    """The compiler did not expose a scope required for lexical resolution."""


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self, tree: ast.AST, aliases: dict[str, frozenset[_Binding]], symbols: symtable.SymbolTable,
                 source: _Source):
        self.occurrences: list[_Occurrence] = []
        self.source = source
        self.scope: list[str] = []
        self.parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        self.aliases = dict(aliases)
        self.module_aliases = self.aliases
        self.symbols = symbols
        self.class_enclosing_aliases: dict[str, frozenset[_Binding]] | None = None
        self.child_scopes: dict[tuple[int, str, int], Iterator[symtable.SymbolTable]] = {}

    def _reference(self, name: str, node: ast.AST, priority: int) -> None:
        root, separator, suffix = name.partition(".")
        bindings = self.aliases.get(root, frozenset((_Binding(root),)))
        canonical = frozenset(binding.canonical_name + (separator + suffix if separator else "")
                              if binding.canonical_name is not None else None for binding in bindings)
        context = {binding.origin for binding in bindings if binding.origin is not None}
        statement = self._statement(node)
        enclosing = self._context_node(statement)
        if enclosing is not statement:
            context.add(self._source_range(enclosing))
        self.occurrences.append(_Occurrence(name, statement, False, priority, canonical, frozenset(context)))

    def _source_range(self, node: ast.AST) -> _SourceRange:
        start, end = _node_range(self.source.source_map, node, definition=False)
        return _SourceRange(self.source.snapshot.snapshot_id, self.source.element.element_id, start, end)

    def _context_node(self, node: ast.AST) -> ast.AST:
        selected = self._statement(node)
        parent = self.parents.get(selected)
        while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            if isinstance(parent, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.TryStar,
                                   ast.With, ast.AsyncWith, ast.Match)):
                selected = parent
            parent = self.parents.get(parent)
        return selected

    def _shadow(self, name: str, node: ast.AST | None = None) -> None:
        if name in self.aliases:
            # A local value is not the imported module, even when the module
            # was imported without an alias and its spelling is unchanged.
            origin = self._source_range(self._context_node(node)) if node is not None else None
            self.aliases[name] = frozenset((_Binding(None, origin),))

    def _visit_branches(self, branches: Sequence[Sequence[ast.AST]]) -> None:
        entry = dict(self.aliases)
        exits = []
        for branch in branches:
            # Keep the dictionary identity: explicit globals in a nested scope
            # must see the current branch's module bindings.
            self.aliases.clear()
            self.aliases.update(entry)
            for node in branch:
                self.visit(node)
            exits.append(dict(self.aliases))
        self.aliases.clear()
        self.aliases.update(_merge_aliases(exits))

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
            self._shadow(node.name)
            with self._nested_scope(node.name, node.lineno, class_scope=isinstance(node, ast.ClassDef)):
                self.scope.append(node.name)
                for statement in node.body:
                    self.visit(statement)
                self.scope.pop()
        self._shadow(node.name)

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
                    self._shadow(symbol_name)
                    if symbol_name in self.module_aliases:
                        self.aliases[symbol_name] = self.module_aliases[symbol_name]
                elif not class_scope and symbol.is_local():
                    self._shadow(symbol_name)
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
            self.aliases[name] = frozenset((_Binding(
                alias.name if alias.asname else name, self._source_range(self._context_node(node)),
            ),))
            self.occurrences.append(_Occurrence(alias.name, node, False, 3))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            qualified = ".".join(part for part in (node.module, alias.name) if part)
            self.aliases[name] = frozenset((_Binding(qualified, self._source_range(self._context_node(node))),))
            self.occurrences.append(_Occurrence(qualified, node, False, 3))

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted_name(node.func)
        if name:
            self._reference(name, node, 0)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            self._shadow(node.id, node)
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

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        self._visit_branches([node.body, node.orelse])

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        self._visit_branches([[node.body], [node.orelse]])

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        unmatched = dict(self.aliases)
        exits = []
        has_unmatched_path = True
        for case in node.cases:
            self.aliases.clear()
            self.aliases.update(unmatched)
            self.visit(case.pattern)
            if case.guard is not None:
                self.visit(case.guard)
                # A matched pattern binds its captures even when its guard is
                # false, so the next case may see those changed names.
                unmatched = (dict(self.aliases) if _irrefutable_pattern(case.pattern)
                             else _merge_aliases([unmatched, self.aliases]))
            elif _irrefutable_pattern(case.pattern):
                has_unmatched_path = False
            for statement in case.body:
                self.visit(statement)
            exits.append(dict(self.aliases))
        if has_unmatched_path:
            exits.append(unmatched)
        self.aliases.clear()
        self.aliases.update(_merge_aliases(exits))

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.pattern is not None:
            self.visit(node.pattern)
        if node.name is not None:
            self._shadow(node.name, node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name is not None:
            self._shadow(node.name, node)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        self.generic_visit(node)
        if node.rest is not None:
            self._shadow(node.rest, node)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._visit_branches([[], [node.target, *node.body]])
        self._visit_branches([[], node.orelse])

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        self._visit_branches([[], node.body])
        self._visit_branches([[], node.orelse])

    def visit_Try(self, node: ast.Try | ast.TryStar) -> None:
        interrupted = [dict(self.aliases)]
        for statement in node.body:
            self.visit(statement)
            interrupted.append(dict(self.aliases))
        for statement in node.orelse:
            self.visit(statement)
        exits = [dict(self.aliases)]
        handler_entry = _merge_aliases(interrupted)
        for handler in node.handlers:
            self.aliases.clear()
            self.aliases.update(handler_entry)
            self.visit(handler)
            exits.append(dict(self.aliases))
        self.aliases.clear()
        self.aliases.update(_merge_aliases([*exits, *interrupted] if node.finalbody else exits))
        for statement in node.finalbody:
            self.visit(statement)

    visit_TryStar = visit_Try

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self._shadow(node.name, node)
        for statement in node.body:
            self.visit(statement)
        if node.name is not None:
            self._shadow(node.name, node)

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


def _merge_aliases(branches: list[dict[str, frozenset[_Binding]]]) -> dict[str, frozenset[_Binding]]:
    names = {name for branch in branches for name in branch}
    return {name: frozenset(binding for branch in branches
                            for binding in branch.get(name, frozenset((_Binding(None),))))
            for name in names}


def _irrefutable_pattern(pattern: ast.pattern) -> bool:
    if isinstance(pattern, ast.MatchAs):
        return pattern.pattern is None or _irrefutable_pattern(pattern.pattern)
    return isinstance(pattern, ast.MatchOr) and any(_irrefutable_pattern(item) for item in pattern.patterns)


def _matches(requested: str, occurrence: _Occurrence, *, certain: bool = False) -> bool:
    names = [occurrence.name, *(name for name in occurrence.canonical_names if name is not None)]
    if "." not in requested:
        return any(name.rsplit(".", 1)[-1] == requested for name in names)
    if not certain and requested == occurrence.name:
        # Finding the requested source spelling does not prove which object it
        # names. Keep its original binding context for semantic verification.
        return True
    if occurrence.canonical_names == frozenset((None,)):
        return False
    if certain and len(occurrence.canonical_names) > 1:
        return False
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
    sources_by_key = {(source.snapshot.snapshot_id, source.element.element_id): source for source in sources}
    symbols = requirement.symbols
    by_symbol: dict[str, list[tuple[int, _Source, _Occurrence, tuple[int, int]]]] = {name: [] for name in symbols}
    warnings: list[str] = []
    aliases: dict[str, frozenset[_Binding]] = {}
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
        visitor = _SymbolVisitor(tree, aliases, symbols_table, source)
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
    supporting_evidence: dict[_SourceRange, EvidenceRef] = {}
    covered_aspects: set[str] = set()
    aspects_by_file: dict[str, set[str]] = {file_id: set() for file_id in requirement.file_ids}
    for symbol, matches in by_symbol.items():
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
            preference = (not _matches(symbol, occurrence, certain=True), priority)
            preferred_by_file[key] = min(preference, preferred_by_file.get(key, preference))
        for priority, source, occurrence, (start, end) in matches:
            key = (source.snapshot.snapshot_id if scoped else "all", occurrence.definition)
            if (not _matches(symbol, occurrence, certain=True), priority) != preferred_by_file[key]:
                continue
            evidence = build_evidence(snapshot=source.snapshot, element=source.element, start=start, end=end)
            for selected in occurrence.context_ranges:
                if selected not in supporting_evidence:
                    context_source = sources_by_key[(selected.snapshot_id, selected.element_id)]
                    supporting_evidence[selected] = build_evidence(
                        snapshot=context_source.snapshot, element=context_source.element,
                        start=selected.start, end=selected.end,
                    )
            hits.append(SearchHit(
                evidence=evidence, score=RetrievalScore(metric="source_symbol", direction="higher"), rank=len(hits) + 1,
                supporting_evidence_ids=sorted({supporting_evidence[selected].id for selected in occurrence.context_ranges
                                               if supporting_evidence[selected].id != evidence.id}),
            ))
            terms = _code_terms(occurrence.node)
            terms.update(_fragment_terms(source, occurrence, requirement.aspects))
            covered_aspects.update(terms)
            if source.element.metadata.get("file_id") in aspects_by_file:
                aspects_by_file[source.element.metadata["file_id"]].update(terms)
    source_order = {key: index for index, key in enumerate(sources_by_key)}
    for selected in sorted(supporting_evidence, key=lambda item: (
        source_order[(item.snapshot_id, item.element_id)], item.start, item.end,
    )):
        hits.append(SearchHit(evidence=supporting_evidence[selected], role="context",
                              score=RetrievalScore(metric="source_binding", direction="higher"), rank=len(hits) + 1))
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
    uncertain = False
    for symbol, matches in by_symbol.items():
        for file_id in requirement.file_ids or [None]:
            scoped_matches = [occurrence for _priority, source, occurrence, _bounds in matches
                              if file_id is None or source.element.metadata.get("file_id") == file_id]
            if scoped_matches and not any(_matches(symbol, occurrence, certain=True) for occurrence in scoped_matches):
                uncertain = True
    if uncertain:
        answerability = "unknown"
        warnings.append("source_symbol_binding_unverified")
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
