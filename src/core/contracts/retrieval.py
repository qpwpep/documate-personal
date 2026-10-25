"""The single internal result contract shared by retrieval tools and execution."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from src.core.contracts.debug import RetrievalDiagnostic
from src.core.documents import ParsedDocument
from src.core.evidence import EvidenceRef, SearchHit
from src.core.uploads import UploadFileInfo


class EvidenceVectorStore(Protocol):
    def similarity_search_with_score(
        self, query: str, k: int = 4, *, file_ids: Sequence[str] = (),
    ) -> list[tuple[EvidenceRef, float]]: ...


@runtime_checkable
class UploadRetriever(Protocol):
    @property
    def source_documents(self) -> tuple[ParsedDocument, ...]: ...

    @property
    def upload_files(self) -> tuple[UploadFileInfo, ...]: ...

    @property
    def vectorstore(self) -> EvidenceVectorStore: ...


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hits: list[SearchHit]
    diagnostics: RetrievalDiagnostic
