"""Map Python parser coordinates back to unchanged source text."""

from __future__ import annotations

import ast
import io
import tokenize
from bisect import bisect_right
from functools import cached_property
from itertools import accumulate


class PythonSourceMap:
    def __init__(self, text: str):
        self.text = text
        # Python physical lines end only in LF, CRLF or CR. Unlike splitlines,
        # universal-newline reading does not split formfeeds or Unicode separators.
        self.lines = io.StringIO(text, newline="").readlines()
        self.offsets = list(accumulate(map(len, self.lines), initial=0))

    @property
    def line_count(self) -> int:
        return max(1, len(self.lines))

    def position(self, line: int, byte_column: int) -> int:
        """Convert one-based AST lines and UTF-8 columns to original character offsets."""
        return self.offsets[line - 1] + len(self.lines[line - 1].encode("utf-8")[:byte_column].decode("utf-8"))

    def line_at_offset(self, offset: int) -> int:
        """Return the one-based physical line containing an original character offset."""
        if not 0 <= offset <= len(self.text):
            raise ValueError("Source offset is outside the original text")
        line = bisect_right(self.offsets, offset)
        # EOF is a new empty line only when the source ends with a line break.
        if offset == len(self.text) and self.lines and not self.lines[-1].endswith(("\r", "\n")):
            line -= 1
        return line

    @cached_property
    def _at_tokens(self) -> list[tuple[int, int]]:
        # ast.parse normalizes physical line endings; tokenize needs the same
        # normalization in its readline input. Only this analysis stream changes.
        return [token.start for token in tokenize.generate_tokens(io.StringIO(self.text, newline=None).readline)
                if token.type == tokenize.OP and token.string == "@"]

    def definition_start(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> tuple[int, int]:
        """Include the actual @ even when a decorator expression starts on a later line."""
        if not node.decorator_list:
            return node.lineno, node.col_offset
        first = min(node.decorator_list, key=lambda part: (part.lineno, part.col_offset))
        return max(position for position in self._at_tokens if position < (first.lineno, first.col_offset))
