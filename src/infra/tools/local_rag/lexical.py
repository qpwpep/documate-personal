"""Korean lexical indexing keeps search tokens separate from captured source text."""
from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock

import bm25s
from kiwipiepy import Kiwi
from langchain_core.documents import Document

from src.core.evidence import SourceSelection


_LATIN_OR_NUMBER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*|\d+(?:[./:-]\d+)*")
_HANGUL = re.compile(r"[가-힣]")
_CONTENT_POS = ("NN", "VV", "VA", "MAG", "XR")
_kiwi: Kiwi | None = None
_kiwi_lock = Lock()


def tokenize_texts(texts: Sequence[str]) -> list[list[str]]:
    """Apply the same morphology and identifier policy to documents and queries."""
    global _kiwi
    normalized = [unicodedata.normalize("NFC", text) for text in texts]
    with _kiwi_lock:
        if _kiwi is None:
            _kiwi = Kiwi(num_workers=1)
        korean = [[token.form for token in tokens
                   if token.tag.startswith(_CONTENT_POS) and _HANGUL.search(token.form)]
                  for tokens in _kiwi.tokenize(normalized)]
    for text, tokens in zip(normalized, korean, strict=True):
        for match in _LATIN_OR_NUMBER.finditer(text):
            term = match.group().casefold()
            tokens.append(term)
            if not term[0].isdigit():
                parts = [part for part in re.split(r"[._]+", term) if part]
                if len(parts) > 1:
                    tokens.extend(parts)
    return korean


@dataclass(frozen=True)
class ChunkReference:
    snapshot_id: str
    element_id: str
    selection: SourceSelection
    file_id: str


@dataclass(frozen=True)
class LexicalMatch:
    reference: ChunkReference
    score: float
    query_coverage: float


class KoreanBM25Index:
    def __init__(self, chunks: Sequence[Document]) -> None:
        self.references = tuple(ChunkReference(
            snapshot_id=chunk.metadata["snapshot_id"], element_id=chunk.metadata["element_id"],
            selection=(SourceSelection(cell_ids=json.loads(chunk.metadata["cell_ids_json"]))
                       if chunk.metadata.get("selection_kind") == "table_cells"
                       else SourceSelection(start=chunk.metadata["start"], end=chunk.metadata["end"])),
            file_id=chunk.metadata.get("file_id", ""),
        ) for chunk in chunks)
        tokens = tokenize_texts([chunk.page_content for chunk in chunks])
        self._terms = tuple(frozenset(document_tokens) for document_tokens in tokens)
        self._index = None
        if any(tokens):
            self._index = bm25s.BM25(method="lucene")
            self._index.index(tokens, show_progress=False)

    def search(
        self, query: str, k: int, *, file_ids: Sequence[str] = (),
    ) -> list[LexicalMatch]:
        if self._index is None:
            return []
        tokens = tokenize_texts([query])[0]
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        allowed = set(file_ids)
        rows = [(index, float(score)) for index, score in enumerate(scores)
                if score > 0 and (not allowed or self.references[index].file_id in allowed)]
        rows.sort(key=lambda row: -row[1])
        if file_ids:
            reserved = {}
            for index, score in rows:
                file_id = self.references[index].file_id
                if file_id not in reserved:
                    reserved[file_id] = (index, score)
            selected = list(reserved.values())
            remaining = [row for row in rows if row not in selected]
            selected.extend(remaining[:max(0, max(k, len(allowed)) - len(selected))])
            rows = sorted(selected, key=lambda row: -row[1])
        else:
            rows = rows[:k]
        # Missing vocabulary terms still describe unmet parts of the query.
        query_terms = set(tokens)
        return [LexicalMatch(self.references[index], score,
                             len(query_terms & self._terms[index]) / len(query_terms))
                for index, score in rows]

    def release(self) -> None:
        self._index = None
        self.references = ()
        self._terms = ()
