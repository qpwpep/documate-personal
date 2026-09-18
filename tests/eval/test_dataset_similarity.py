import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.eval.dataset_similarity import build_report, embed_records, extract_prompt_records, load_case_records


@contextmanager
def embedding_endpoint():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(request)
            # Reverse response order so the client must honor API indices.
            payload = {"model": "test-embed", "data": [
                {"index": index, "embedding": [float(index + 1), 1.0]}
                for index in reversed(range(len(request["input"])))
            ], "usage": {"prompt_tokens": 9, "total_tokens": 9}}
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/embeddings", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def record(identifier, group, text):
    return {"id": identifier, "group": group, "text": text, "query": text}


def test_embedding_client_uses_query_mode_and_restores_api_order(monkeypatch):
    monkeypatch.setenv("SIMILARITY_TEST_KEY", "test-boundary-key")
    records = [record("a", "candidate", "Alpha"), record("b", "legacy", "Beta")]
    with embedding_endpoint() as (endpoint, requests):
        vectors, metadata = embed_records(records, endpoint=endpoint, model="test-embed",
                                          api_key_env="SIMILARITY_TEST_KEY")
    assert vectors == [[1.0, 1.0], [2.0, 1.0]]
    assert requests[0]["input_type"] == "query"
    assert requests[0]["input"] == ["Alpha", "Beta"]
    assert metadata["usage"]["total_tokens"] == 9
    assert "test-boundary-key" not in json.dumps(metadata)


def test_report_ranks_all_groups_and_flags_screening_pairs_and_exact_duplicates():
    records = [record("a", "candidate", "Save the answer"),
               record("b", "candidate", "Cancel saving"),
               record("old", "legacy", " SAVE   THE ANSWER "),
               record("example", "prompt", "Store this reply")]
    report = build_report(records, [[1, 0], [0, 1], [1, 0], [0.8, 0.6]], threshold=0.75)
    nearest = report["per_case"][0]
    assert nearest["case_id"] == "a"
    assert nearest["nearest_within"][0]["id"] == "b"
    assert nearest["nearest_legacy"][0] == {"id": "old", "score": 1.0}
    assert nearest["nearest_prompt"][0] == {"id": "example", "score": 0.8}
    assert [(pair["left"], pair["right"]) for pair in report["exact_query_duplicates"]] == [("a", "old")]
    assert len(report["screened_pairs"]) == 2
    assert "screening" in report["interpretation"].lower()


def test_prompt_extraction_uses_ast_without_executing_module(tmp_path):
    path = tmp_path / "prompt_builder.py"
    path.write_text('raise RuntimeError("must never execute")\n'
                    'PLANNER_SYS = "Interpret the current user request carefully. '
                    'For example, \'Do not send this to Slack\' forbids delivery and preserves the explanation."\n',
                    encoding="utf-8")
    records = extract_prompt_records([path])
    assert any(row["query"] == "Do not send this to Slack" for row in records)
    assert all(row["group"] == "prompt" for row in records)
    assert any(row["kind"] == "quoted_example" for row in records)


def test_case_embedding_text_excludes_answer_facts_and_rationale(tmp_path):
    path = tmp_path / "cases.jsonl"
    row = {"case_id": "a", "query": "What timeout is configured?", "setup_turns": ["Use alpha.py"],
           "capability": "config_lookup", "oracle": {"required_facts": ["SECRETANSWER"]},
           "provenance": {"rationale": "SECRETREASON"}}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    records = load_case_records([path], group="candidate")
    assert "Use alpha.py" in records[0]["text"]
    assert "config_lookup" in records[0]["text"]
    assert "SECRET" not in records[0]["text"]


def test_invalid_vectors_cannot_produce_a_false_similarity_report():
    records = [record("a", "candidate", "Alpha")]
    with pytest.raises(ValueError, match="nonzero"):
        build_report(records, [[0, 0]])


def test_similarity_cli_writes_utf8_lf_reports_and_vectors(tmp_path, monkeypatch):
    from src.eval.dataset_similarity import main

    monkeypatch.setenv("SIMILARITY_TEST_KEY", "test-boundary-key")
    source = tmp_path / "candidates.jsonl"
    source.write_bytes('{"case_id":"one","query":"첨부를 요청해 주세요."}\n'.encode("utf-8"))
    report_path = tmp_path / "similarity.json"
    vector_path = tmp_path / "vectors.json"
    with embedding_endpoint() as (endpoint, requests):
        assert main(["--cases", str(source), "--legacy", "--prompts", "--out", str(report_path),
                     "--vectors", str(vector_path), "--endpoint", endpoint, "--model", "test-embed",
                     "--api-key-env", "SIMILARITY_TEST_KEY"]) == 0
    assert requests[0]["input"] == ["첨부를 요청해 주세요."]
    for path in (report_path, vector_path):
        content = path.read_bytes()
        assert b"\r" not in content
        assert not content.startswith(b"\xef\xbb\xbf")
        json.loads(content.decode("utf-8"))
