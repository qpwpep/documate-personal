from pathlib import Path

import pytest

from src.eval.generate_cases import generate_cases_file


def test_template_generator_cannot_overwrite_curated_release(tmp_path, monkeypatch):
    release = tmp_path / "curated.jsonl"
    release.write_text('{"curated": true}\n', encoding="utf-8")
    monkeypatch.setattr("src.eval.generate_cases.get_generated_cases_fixture_path", lambda: release)
    before = release.read_bytes()
    with pytest.raises(ValueError, match="NeMo"):
        generate_cases_file(
            seed_path=Path("data/benchmarks/fixtures/cases.seed.jsonl"),
            out_path=release,
            target=120,
        )
    assert release.read_bytes() == before
