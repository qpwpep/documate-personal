"""Unsent text, separate from a request held after a server/upload failure."""
from dataclasses import dataclass


@dataclass
class ComposerDraft:
    text: str = ""
    revision: int = 0
    needs_restore: bool = False

    def record(self, text: str, *, revision: int) -> None:
        if revision == self.revision:
            self.text = text

    def consume(self) -> None:
        self.text = ""
        self.revision += 1
        self.needs_restore = False

    def edit(self, text: str) -> None:
        self.text = text
        self.revision += 1
        self.needs_restore = True
