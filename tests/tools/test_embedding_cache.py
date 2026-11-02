from pathlib import Path

from langchain_core.embeddings import Embeddings

from src.infra.embedding_cache import CachedEmbeddings


class DocumentEmbeddings(Embeddings):
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vectors[text]


def cached_embeddings(directory: Path, vectors: dict[str, list[float]]) -> CachedEmbeddings:
    return CachedEmbeddings(DocumentEmbeddings(vectors), directory,
        namespace={"embedding": {"model": "test", "dimensions": None}},
        max_bytes=1024 * 1024, ttl_seconds=1800)


def test_cached_document_vectors_preserve_order_and_duplicate_inputs(tmp_path):
    original = {"first": [1.0, 2.0], "second": [3.0, 4.0]}
    cached_embeddings(tmp_path, original).embed_documents(["first", "second"])

    restored = cached_embeddings(tmp_path, {}).embed_documents(["second", "first", "second"])

    assert restored == [[3.0, 4.0], [1.0, 2.0], [3.0, 4.0]]


def test_cached_document_with_different_dimensions_is_recomputed(tmp_path):
    cached_embeddings(tmp_path, {"first": [1.0, 2.0]}).embed_documents(["first"])
    cached_embeddings(tmp_path, {"second": [3.0, 4.0, 5.0]}).embed_documents(["second"])

    restored = cached_embeddings(tmp_path, {"second": [6.0, 7.0]}).embed_documents(["first", "second"])

    assert restored == [[1.0, 2.0], [6.0, 7.0]]
    assert cached_embeddings(tmp_path, {}).embed_documents(["second"]) == [[6.0, 7.0]]
