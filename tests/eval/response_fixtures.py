import io
import json
from collections.abc import Iterable
from pathlib import Path

import requests

from src.core.answer_schema import AnswerDocument, AnswerResponse, finalize_answer, text_document
from src.core.answer_schema import export_answer_text
from src.core.save_contract import SaveOperation
from src.core.contracts.provenance import AnswerProvenance, AnswerSource
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


def saved_receipt(response: dict, root: Path, *, session_id: str, operation_id: str = "save-operation",
                  target_kind: str = "compose", source_hash: str | None = None) -> dict:
    """Commit a real artifact; only HTTP and model boundaries are faked by callers."""
    from src.infra.saved_artifacts import save_artifact

    answer = AnswerResponse.model_validate(response)
    text = export_answer_text(answer, include_sources=True)
    operation = SaveOperation.for_text(
        text, operation_id=operation_id, session_id=session_id, request_id="contract-request",
        contract_revision=1, target_kind=target_kind, source_hash=source_hash or answer.content_hash,
        answer_hash=answer.content_hash,
    )
    manifest, path = save_artifact(root, text, operation, ttl_seconds=86400)
    return {"kind": "save_text", "status": "success", "file_path": str(path),
            "operation": operation.model_dump(mode="json"), "artifact": manifest.artifact.model_dump(mode="json"),
            "verification": "verified"}


def artifact_http_response(root: Path, filename: str) -> requests.Response:
    """Represent the real storage reader's output at an external HTTP boundary."""
    from src.infra.saved_artifacts import ArtifactError, read_saved_artifact

    response = requests.Response()
    try:
        manifest, payload = read_saved_artifact(root, filename)
    except ArtifactError as exc:
        response.status_code = {"manifest_missing": 404, "artifact_missing": 404,
                                "artifact_expired": 410, "artifact_mismatch": 409}.get(exc.code, 503)
        response._content = json.dumps({"detail": {"code": exc.code, "message": str(exc)}}).encode()
    else:
        response.status_code = 200
        response._content = payload
        response.headers["X-Save-Binding-SHA256"] = manifest.operation.binding_sha256
        response.headers["X-Artifact-Id"] = manifest.artifact.artifact_id
    return response


def answer_provenance(response, *, source=None, body_kind="compose", evidence_packet=None,
                      request_id=None, contract_revision=None, save_operation_binding_sha256=None) -> dict:
    """Explicitly declare the construction inputs of a valid HTTP fixture answer."""
    answer = response if isinstance(response, AnswerResponse) else AnswerResponse.model_validate(response)
    parent = AnswerSource.model_validate(source) if source is not None else None
    return AnswerProvenance(
        body_kind=body_kind, response_hash=answer.content_hash, source=parent,
        request_id=request_id, contract_revision=contract_revision,
        save_operation_binding_sha256=save_operation_binding_sha256,
        evidence_packet=([citation.evidence for citation in answer.citations]
                         if evidence_packet is None else evidence_packet),
    ).model_dump(mode="json")


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
