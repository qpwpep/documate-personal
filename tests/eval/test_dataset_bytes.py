from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.eval.io import dump_jsonl
from src.eval.nemo_generate import _write_jsonl
from src.eval.dataset_bytes import validate_text_bytes


@pytest.mark.parametrize("writer", [dump_jsonl, _write_jsonl], ids=["assembled", "generated"])
def test_candidate_writers_emit_utf8_lf_bytes(tmp_path, writer):
    target = tmp_path / "candidates.jsonl"
    records = [{"case_id": "case-1", "query": "업로드해 주세요."}, {"case_id": "case-2"}]

    writer(target, records)

    content = target.read_bytes()
    assert not content.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in content
    assert content.endswith(b"\n")
    assert [json.loads(line) for line in content.decode("utf-8").splitlines()] == records


@pytest.mark.parametrize("name", ["specs.jsonl", "review.json", "uploads/sample.py", "notebook.ipynb"])
@pytest.mark.parametrize("content, expected", [
    (b"\xef\xbb\xbf{}\n", "BOM"),
    (b"{}\r\n", "LF line endings"),
    (b"first\rsecond", "LF line endings"),
    (b"\xff\n", "valid UTF-8"),
])
def test_text_validation_rejects_noncanonical_bytes_without_changing_them(name, content, expected):
    before = hashlib.sha256(content).hexdigest()

    errors = validate_text_bytes(name, content)

    assert len(errors) == 1
    assert name in errors[0]
    assert expected in errors[0]
    assert hashlib.sha256(content).hexdigest() == before


def test_text_validation_allows_escaped_newlines_and_keeps_binaries_opaque():
    assert validate_text_bytes("review.json", '{"example":"안녕\\r\\n"}\n'.encode()) == []
    assert validate_text_bytes("report.pdf", b"%PDF\r\n\xff") == []


def test_source_generator_writes_canonical_text_in_an_isolated_destination(tmp_path, monkeypatch):
    for module in ("docx", "reportlab", "PIL"):
        pytest.importorskip(module)
    from script import build_release_sources as generator

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    for name in ("sample_data_analysis.py", "sample_pipeline.ipynb", "sample_visualization.ipynb"):
        (uploads / name).write_bytes((generator.UPLOADS / name).read_bytes())
    design = tmp_path / "design"
    monkeypatch.setattr(generator, "UPLOADS", uploads)
    monkeypatch.setattr(generator, "DESIGN", design)
    monkeypatch.setattr(generator, "SOURCES", {})
    monkeypatch.setattr(generator, "CASES", [])
    monkeypatch.setattr(sys, "argv", ["build_release_sources.py"])

    generator.main()

    rows = (design / "retrieval_specs.jsonl").read_bytes()
    assert len(rows.splitlines()) == 90
    assert "release/cleaning.py" in json.loads((design / "sources.json").read_bytes())
    for path in [*design.iterdir(), *(uploads / "release").iterdir()]:
        assert validate_text_bytes(path.name, path.read_bytes()) == []


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.mark.parametrize("autocrlf", ["true", "false", "input"])
def test_approved_bytes_survive_git_roundtrip_and_history_is_immutable(tmp_path, autocrlf):
    if shutil.which("git") is None:
        pytest.skip("Git is required to verify checkout behavior")
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "--quiet")
    # Keep every configuration change inside this temporary repository.
    git(repo, "config", "core.autocrlf", autocrlf)
    git(repo, "config", "core.safecrlf", "false")
    git(repo, "config", "core.eol", "native")
    attributes = Path(__file__).resolve().parents[2] / ".gitattributes"
    (repo / ".gitattributes").write_bytes(attributes.read_bytes())
    originals = {
        "data/benchmarks/fixtures/cases.generated.jsonl": '{"query":"질문"}\n'.encode(),
        "data/benchmarks/design/release_review.json": b'{"reviewed": true}\n',
        "data/benchmarks/design/run/source_specs.jsonl": b'{"case_id": "one"}\n',
        "data/benchmarks/fixtures/uploads/release/example.py": b"LIMIT = 17\n",
        "data/benchmarks/fixtures/uploads/sample_pipeline.ipynb": b'{"cells": []}\n',
        "data/benchmarks/fixtures/uploads/release/report.pdf": b"%PDF-1.4\r\nopaque\r\n",
        "data/benchmarks/fixtures/uploads/release/image.png": b"\x89PNG\r\n\x1a\nopaque",
        "data/benchmarks/fixtures/uploads/release/document.docx": b"PK\x03\x04\x00\r\nopaque",
        "data/benchmarks/history/old-review/design/release_review.json": b'{"reviewed": true}\r\n',
        "data/benchmarks/history/old-review/fixtures/cases.generated.jsonl": b'{"case_id":"old"}\r\n',
    }
    for name, content in originals.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    git(repo, "add", ".")
    for name in originals:
        (repo / name).unlink()

    git(repo, "checkout-index", "--all", "--force")

    changed = [name for name, content in originals.items()
               if hashlib.sha256((repo / name).read_bytes()).digest() != hashlib.sha256(content).digest()]
    assert changed == []
