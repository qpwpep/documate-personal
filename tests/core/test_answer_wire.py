"""The public answer keeps original sources once and preserves checked references."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.app.transport_contracts import AgentResponse, agent_response_json_schema
from src.core.answer_schema import ContentUnit, ParagraphBlock, AnswerDocument, finalize_answer
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import build_evidence
from src.core.uploads import UploadManifest


def test_public_answer_serializes_one_source_and_restores_after_original_removal(tmp_path: Path) -> None:
    original = tmp_path / "source.py"
    source_text = "# original source line\n" * 8192
    original.write_text(source_text, encoding="utf-8")
    snapshot = build_snapshot(
        source_uri=str(original), title=original.name, media_type="text/x-python",
        source_type="upload", content=source_text, parser="python-ast", parser_version="2",
    )
    element = DocumentElement(element_id="python-source", kind="code", text=source_text, language="python")
    evidence = [build_evidence(snapshot=snapshot, element=element, start=index * 23, end=index * 23 + 20)
                for index in range(5)]
    answer = finalize_answer(AnswerDocument(blocks=[ParagraphBlock(content=[
        ContentUnit(text="서로 다른 다섯 범위를 확인했습니다.", basis="source", refs=[item.id for item in evidence]),
    ])]), evidence, retrieval_required=True)
    result = AgentResponse(
        response=answer, answer_id="answer-1", execution_id="execution-1", trace="",
        upload_manifest=UploadManifest(epoch="epoch", revision=0, files=[]),
    )

    payload = result.model_dump(mode="json")
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    original.unlink()

    assert len(payload["response"]["sources"]) == 1
    assert len(payload["response"]["citations"]) == 5
    assert all("evidence" not in citation for citation in payload["response"]["citations"])
    assert len(encoded) < len(source_text.encode("utf-8")) * 1.15
    restored = AgentResponse.model_validate_json(encoded)
    assert restored.response == answer
    assert [citation.evidence.excerpt for citation in restored.response.citations] == [item.excerpt for item in evidence]

    source_id = payload["response"]["citations"][0]["source_id"]
    del payload["response"]["sources"][source_id]
    with pytest.raises(ValidationError, match="source"):
        AgentResponse.model_validate(payload)


def test_public_schema_exposes_normalized_answer_sources() -> None:
    schemas = agent_response_json_schema()["components"]["schemas"]
    answer = schemas["SourcedAnswerPayload"]

    assert "sources" in answer["required"]
    assert schemas["SourceCitation"]["properties"]["source_id"]["type"] == "string"
    assert "evidence" not in schemas["SourceCitation"]["properties"]
