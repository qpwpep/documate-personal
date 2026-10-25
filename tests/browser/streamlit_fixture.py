"""Real Streamlit UI with a deterministic, local HTTP boundary for browser checks.

Run from the repository root::

    uv run python -m streamlit run tests/browser/streamlit_fixture.py --server.port 8502

Any question returns a cited text answer. A question starting with ``재시도:`` or
``수정:`` fails on its first submission in that conversation, then succeeds.
For ``저장실패:`` questions, the first TXT export returns HTTP 503 and the next
export succeeds. ``지연:`` questions return their normal answer after a three-second
HTTP delay, allowing pending UI events to be tested while a request is in flight.
``부분:`` questions return a partial answer with one cited excerpt and three
limitations (two answer-wide and one attached to the excerpt).
``문서:`` questions return an explicitly synthetic Markdown source for checking
headings, lists, links, code fences and the retained raw-source view.
No LLM, search provider, or Slack service is called.

The process logs its ephemeral API URL as DOCUMATE_UX_FIXTURE_API. Its read-only
``/_fixture/metrics`` endpoint exposes submitted questions and export attempts.
"""
from __future__ import annotations

import atexit
import json
import tempfile
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import streamlit as st

from src.core.answer_export import build_answer_export_operation
from src.core.answer_schema import ActionReceipt, AnswerResponse, ResponseIssue, export_answer_text, finalize_answer, text_document
from src.core.contracts.outcome import AnswerExportResult, TurnResult
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import build_evidence
from src.core.llm_errors import make_problem
from src.core.save_contract import SaveArtifact
from src.core.uploads import UploadManifest


SOURCE_TEXT = "회귀 검증용 공식 문서 원문입니다. 이 문장은 라이트 모드에서도 선명하게 읽혀야 합니다."
MARKDOWN_SOURCE_TEXT = """# 문서 표시 검증용 합성 자료

이 내용은 실제 공식 문서를 검색한 결과가 아니라 **브라우저 회귀 검증용 자료**입니다.

## 배열 모양 예시

- `shape`는 배열의 모양을 나타냅니다.
- [NumPy 문서 링크](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)는 링크 표시를 확인하기 위한 예시입니다.

```python
values.reshape((2, 3))
```
"""
PARTIAL_LIMITS = (
    "요청한 비교 중 공식 문서의 확인 가능한 내용만 제공합니다.",
    "업로드한 코드의 근거가 없어 공식 문서와 코드의 비교를 완료하지 못했습니다.",
    "이 문단은 공식 문서의 발췌이며 업로드 코드의 동작은 확인하지 못했습니다.",
)


class FixtureState:
    """Thread-safe scripted HTTP outcomes; production UI and client stay real."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.questions: list[dict[str, str]] = []
        self.exports: list[dict[str, str]] = []
        self.attempts: Counter[tuple[str, str]] = Counter()
        self.export_attempts: Counter[tuple[str, str]] = Counter()
        self.answers: dict[tuple[str, str], tuple[AnswerResponse, str]] = {}
        self.downloads: dict[str, bytes] = {}

    @staticmethod
    def manifest(session_id: str) -> dict:
        return UploadManifest(epoch=f"fixture-{session_id}", revision=0, files=[]).model_dump(mode="json")

    def submit(self, payload: dict) -> dict:
        with self.lock:
            session_id, query = payload["session_id"], payload["query"]
            self.questions.append({"session_id": session_id, "query": query})
            self.attempts[(session_id, query)] += 1
            request_id = f"fixture-request-{len(self.questions)}"
            if query.startswith(("재시도:", "수정:")) and self.attempts[(session_id, query)] == 1:
                result = TurnResult(
                    status="failed", request_id=request_id,
                    problem=make_problem("evidence_insufficient", "pre_synthesis_validation"),
                )
            else:
                markdown_source = query.startswith("문서:")
                source_text = MARKDOWN_SOURCE_TEXT if markdown_source else SOURCE_TEXT
                snapshot = build_snapshot(
                    source_uri="https://docs.example.org/fixtures/markdown" if markdown_source else "https://docs.python.org/3/tutorial/",
                    title="문서 표시 검증용 합성 자료" if markdown_source else "브라우저 회귀 검증 자료",
                    media_type="text/markdown" if markdown_source else "text/html", source_type="official", content=source_text,
                    parser="browser-fixture", parser_version="1",
                )
                evidence = build_evidence(
                    snapshot=snapshot,
                    element=DocumentElement(element_id="fixture-paragraph", kind="paragraph", text=source_text),
                )
                partial = query.startswith("부분:")
                issues = [
                    ResponseIssue(code="answer_incomplete", message=PARTIAL_LIMITS[0]),
                    ResponseIssue(code="retrieved_evidence_incomplete", message=PARTIAL_LIMITS[1]),
                    ResponseIssue(code="comparison_scope_limited", message=PARTIAL_LIMITS[2], unit_id="b0.content.0"),
                ] if partial else []
                document = text_document(
                    SOURCE_TEXT if partial else f"회귀 검증 답변: {query}",
                    basis="excerpt" if partial else "source", refs=[evidence.id],
                )
                answer = finalize_answer(document, [evidence], issues=issues)
                answer_id = f"fixture-answer-{len(self.questions)}"
                self.answers[(session_id, answer_id)] = (answer, query)
                result = TurnResult(
                    status="partial" if partial else "completed",
                    response=answer, answer_id=answer_id, request_id=request_id,
                    problem=make_problem("evidence_insufficient", "pre_synthesis_validation") if partial else None,
                )
            return {**result.model_dump(mode="json"), "trace": "", "upload_manifest": self.manifest(session_id)}

    def save(self, session_id: str, answer_id: str) -> tuple[int, dict]:
        with self.lock:
            key = session_id, answer_id
            self.exports.append({"session_id": session_id, "answer_id": answer_id})
            self.export_attempts[key] += 1
            if key not in self.answers:
                return 404, {"detail": {"message": "검증용 답변이 없습니다."}}
            answer, query = self.answers[key]
            if query.startswith("저장실패:") and self.export_attempts[key] == 1:
                return 503, {"detail": {"message": "검증용 저장 서버가 잠시 중단되었습니다."}}
            operation = build_answer_export_operation(answer, session_id=session_id)
            payload = export_answer_text(answer, include_sources=True).encode("utf-8-sig")
            assert operation.matches_bytes(payload)
            filename = f"fixture_{operation.binding_sha256}.txt"
            self.downloads[filename] = payload
            created = time.time()
            artifact = SaveArtifact(
                artifact_id=operation.binding_sha256, filename=filename,
                sha256=operation.payload_sha256, byte_count=len(payload),
                created_at=created, expires_at=created + 3600,
            )
            receipt = ActionReceipt(
                kind="save_text", status="success", verification="verified",
                file_path=f"fixture-memory/{filename}", operation=operation, artifact=artifact,
            )
            result = AnswerExportResult(
                status="completed", answer_id=answer_id, request_id=f"fixture-export-{len(self.exports)}",
                response=answer.model_copy(update={"actions": [receipt]}), message="저장했습니다.",
            )
            return 200, result.model_dump(mode="json")

    def metrics(self) -> dict:
        with self.lock:
            return {"questions": list(self.questions), "exports": list(self.exports)}


@st.cache_resource
def fixture_boundary() -> tuple[str, Path]:
    state = FixtureState()
    temporary = tempfile.TemporaryDirectory(prefix="documate-browser-fixture-")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args) -> None:
            pass

        def respond(self, status: int, payload: bytes, content_type: str = "application/json; charset=utf-8") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def json_response(self, status: int, data: dict) -> None:
            self.respond(status, json.dumps(data, ensure_ascii=False).encode("utf-8"))

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            parts = path.strip("/").split("/")
            if path == "/_fixture/metrics":
                self.json_response(200, state.metrics())
            elif len(parts) == 3 and parts[0] == "sessions" and parts[2] == "uploads":
                self.json_response(200, state.manifest(parts[1]))
            elif len(parts) == 2 and parts[0] == "download" and unquote(parts[1]) in state.downloads:
                self.respond(200, state.downloads[unquote(parts[1])], "text/plain; charset=utf-8")
            else:
                self.json_response(404, {"detail": "Unknown browser-fixture route"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            parts = path.strip("/").split("/")
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            if path == "/agent/stream":
                result = state.submit(payload)
                if payload["query"].startswith("지연:"):
                    time.sleep(3)
                event = "event: final_response\ndata: " + json.dumps(result, ensure_ascii=False) + "\n\n"
                self.respond(200, event.encode("utf-8"), "text/event-stream; charset=utf-8")
            elif len(parts) == 5 and parts[0] == "sessions" and parts[2] == "answers" and parts[4] == "save":
                status, result = state.save(parts[1], parts[3])
                self.json_response(status, result)
            else:
                self.json_response(404, {"detail": "Unknown browser-fixture route"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="documate-browser-fixture").start()
    atexit.register(server.server_close)
    atexit.register(temporary.cleanup)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"DOCUMATE_UX_FIXTURE_API={url}", flush=True)
    return url, Path(temporary.name)


def main() -> None:
    from src.infra.settings import get_settings
    from src.app.web import streamlit_state

    api_url, temporary_root = fixture_boundary()
    get_settings().fastapi_url = api_url
    streamlit_state.get_uploads_dir = lambda: temporary_root / "uploads"
    from src.app.web import streamlit_app

    streamlit_app.SETTINGS = get_settings()
    streamlit_app.main()


if __name__ == "__main__":
    main()
