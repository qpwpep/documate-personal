"""A rejected source-grounded draft for transport and evaluation boundaries."""
from src.core.answer_schema import text_document
from src.core.answer_schema.models import document_hash
from src.core.contracts.validation_diagnostics import (
    DiagnosticPacket, DiagnosticPredicate, ValidationDiagnostics, ValidationObservation,
)
from src.core.decisions import prepared_packet_fingerprint
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import build_evidence
from src.core.request_contracts import RequestContract


def rejected_answer_diagnostics() -> ValidationDiagnostics:
    source_text = "A read-only service accepts read requests."
    snapshot = build_snapshot(
        source_uri="https://example.test/read-only", title="Read-only service",
        media_type="text/plain", source_type="official", content=source_text,
        parser="diagnostic-test", parser_version="1",
    )
    evidence = build_evidence(
        snapshot=snapshot,
        element=DocumentElement(element_id="body", kind="paragraph", text=source_text),
    )
    body = text_document("이 서비스는 쓰기 요청도 처리합니다.", basis="source", refs=[evidence.id])
    return ValidationDiagnostics(
        schema_version=1, evidence={evidence.id: evidence}, observations=[ValidationObservation(
            sequence=1, stage="post_synthesis_validation", attempt=1, revision="generated",
            model="fixture-validator", threshold=0.8, status="completed",
            user_input="읽기 전용 서비스가 처리하는 요청을 설명해줘.",
            request_contract=RequestContract(), tasks=[], body=body, body_hash=document_hash(body),
            packets=[DiagnosticPacket(
                mode="used", evidence_ids=[evidence.id], requirement_map={},
                fingerprint=prepared_packet_fingerprint([evidence], {}),
            )],
            predicates=[DiagnosticPredicate(
                name="grounded_0", criterion="grounded", unit_id="b0.content.0",
                evidence_id=evidence.id, status="evaluated", probability=0.12,
            )],
        )],
    )
