import io
import json
from collections.abc import Iterable

import requests

from src.core.answer_schema import AnswerDocument, finalize_answer, text_document
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence


def sse_frame(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def sse_http_response(
    status_code: int,
    payload: dict | None = None,
    *,
    chunks: Iterable[bytes] | None = None,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    """Supply a real requests response at the HTTP boundary with an SSE body."""
    response = requests.Response()
    response.status_code = status_code
    response.headers.update({"content-type": "text/event-stream; charset=utf-8"})
    response.headers.update(headers or {})
    if status_code != 200:
        response.headers["content-type"] = "application/json"
        response._content = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
        response._content_consumed = True
        return response

    class StreamBody(io.BytesIO):
        def stream(self, chunk_size, decode_content=True):
            yield from chunks if chunks is not None else (
                sse_frame("final_response", payload or {}),
                sse_frame("done", {}),
            )

    response.raw = StreamBody()
    return response


def plain_response(text: str | list[str]) -> dict:
    paragraphs = [text] if isinstance(text, str) else text
    document = AnswerDocument.model_validate({
        "blocks": [
            {"type": "paragraph", "content": [{"text": paragraph, "basis": "interaction", "refs": []}]}
            for paragraph in paragraphs
        ],
    })
    return finalize_answer(document, []).model_dump(mode="json")


def source_evidence(*, official: bool = True, text: str = "공식 설명", source_uri: str | None = None):
    snapshot = build_snapshot(
        source_uri=source_uri or ("https://numpy.org/doc/stable/" if official else "uploads/demo/sample.ipynb"),
        title="NumPy Docs" if official else "Notebook",
        media_type="text/plain",
        source_type="official" if official else "upload",
        content=text,
        parser="eval-fixture",
        parser_version="1",
    )
    element = DocumentElement(
        element_id="body",
        kind="paragraph" if official else "code",
        text=text,
        anchors=[SourceAnchor(kind="web" if official else "notebook", start=0, end=len(text))],
    )
    return build_evidence(snapshot=snapshot, element=element)


def source_hit(*, official: bool = True, text: str = "공식 설명", source_uri: str | None = None) -> SearchHit:
    return SearchHit(evidence=source_evidence(official=official, text=text, source_uri=source_uri), score=RetrievalScore(metric="rank", raw=1, direction="lower"), rank=1)


def comparison_response() -> dict:
    official = source_evidence()
    upload = source_evidence(official=False, text="업로드 비교")
    document = AnswerDocument.model_validate({
        "blocks": [
            {"type": "paragraph", "content": [{"text": "공식 설명", "basis": "source", "refs": [official.id]}]},
            {"type": "paragraph", "content": [{"text": "업로드 비교", "basis": "source", "refs": [upload.id]}]},
        ],
    })
    return finalize_answer(document, [official, upload]).model_dump(mode="json")
