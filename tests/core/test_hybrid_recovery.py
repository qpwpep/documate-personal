import unittest

from src.core.answer_schema import AgentResponsePayloadModel
from src.core.contracts import RetrievalDiagnostic
from src.core.evidence import EvidenceItem
from src.runtime.nodes.validation.evidence_validator import assess_validation, build_validation_snapshot
from src.runtime.nodes.validation.policy import apply_validation_outcome
from src.runtime.nodes.validation.repair import repair_required_sections
from src.core.planner_schema import PlannerOutput, RetrievalTask


def _docs_evidence() -> dict:
    return {
        "kind": "official",
        "tool": "tavily_search",
        "source_id": "url:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
        "document_id": "url:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
        "url_or_path": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
        "title": "train_test_split",
        "snippet": "train_test_split splits arrays or matrices into random train and test subsets.",
        "score": 0.92,
    }


def _upload_evidence(
    snippet: str = "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
) -> dict:
    return {
        "kind": "local",
        "tool": "upload_search",
        "source_id": "path:uploads/demo/sample_pipeline.ipynb#cell=2;chunk=0;start=0;end=96",
        "document_id": "path:uploads/demo/sample_pipeline.ipynb",
        "url_or_path": "uploads/demo/sample_pipeline.ipynb",
        "snippet": snippet,
        "score": 0.81,
        "cell_id": 2,
        "chunk_id": 0,
        "start_offset": 0,
        "end_offset": 96,
    }


def _hybrid_snapshot(response_payload: AgentResponsePayloadModel):
    planner_output = PlannerOutput(
        use_retrieval=True,
        tasks=[
            RetrievalTask(route="docs", query="train_test_split official docs", k=3),
            RetrievalTask(route="upload", query="train_test_split uploaded notebook example", k=3),
        ],
    )
    parsed_evidence = [
        EvidenceItem.model_validate(_docs_evidence()),
        EvidenceItem.model_validate(
            _upload_evidence(
                "train_test_split(X, y, test_size=0.2, random_state=42)"
            )
        ),
    ]
    return build_validation_snapshot(
        user_input="Compare official train_test_split docs with the uploaded code.",
        planner_output=planner_output,
        parsed_evidence=parsed_evidence,
        current_attempt_retrieval_errors=[],
        current_attempt_retrieval_diagnostics=[
            RetrievalDiagnostic(
                tool="tavily_search",
                route="docs",
                status="success",
                message="",
                query="train_test_split official docs",
                attempt=1,
            ),
            RetrievalDiagnostic(
                tool="upload_search",
                route="upload",
                status="success",
                message="",
                query="train_test_split uploaded notebook example",
                attempt=1,
            ),
        ],
        response_payload=response_payload,
    )


class HybridRecoveryTest(unittest.TestCase):
    def test_hybrid_validation_rejects_repeated_section_bodies(self) -> None:
        repeated = "train_test_split splits arrays into train and test subsets."
        response_payload = AgentResponsePayloadModel.model_validate(
            {
                "answer": repeated,
                "claims": [
                    {
                        "text": repeated,
                        "evidence_ids": [
                            "url:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"
                        ],
                        "confidence": 0.9,
                    },
                    {
                        "text": "The uploaded code sets test_size=0.2 and random_state=42.",
                        "evidence_ids": [
                            "path:uploads/demo/sample_pipeline.ipynb#cell=2;chunk=0;start=0;end=96"
                        ],
                        "confidence": 0.8,
                    },
                ],
                "sections": [
                    {"kind": "official_docs", "heading": "Official", "body": repeated},
                    {"kind": "upload_code", "heading": "Upload", "body": repeated},
                    {"kind": "comparison", "heading": "Comparison", "body": repeated},
                ],
                "evidence": [],
                "confidence": 0.85,
            }
        )

        assessment = assess_validation(_hybrid_snapshot(response_payload))

        self.assertFalse(assessment.has_grounded_response_payload)
        self.assertEqual(assessment.retry_reason, "unsupported_claims")
        self.assertIn("HYBRID_SECTION_REPEATED", assessment.error_codes)
        self.assertNotIn("VALIDATION_UNSUPPORTED_CLAIMS", assessment.error_codes)

    def test_hybrid_validation_rejects_upload_section_without_actual_options(self) -> None:
        response_payload = AgentResponsePayloadModel.model_validate(
            {
                "answer": "The docs and upload both use train_test_split.",
                "claims": [
                    {
                        "text": "train_test_split splits arrays into train and test subsets.",
                        "evidence_ids": [
                            "url:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"
                        ],
                        "confidence": 0.9,
                    },
                    {
                        "text": "The uploaded code uses train_test_split.",
                        "evidence_ids": [
                            "path:uploads/demo/sample_pipeline.ipynb#cell=2;chunk=0;start=0;end=96"
                        ],
                        "confidence": 0.8,
                    },
                ],
                "sections": [
                    {
                        "kind": "official_docs",
                        "heading": "Official",
                        "body": "The official docs describe train/test splitting.",
                    },
                    {
                        "kind": "upload_code",
                        "heading": "Upload",
                        "body": "The uploaded code calls train_test_split.",
                    },
                    {
                        "kind": "comparison",
                        "heading": "Comparison",
                        "body": "The uploaded code follows the same train/test split pattern.",
                    },
                ],
                "evidence": [],
                "confidence": 0.85,
            }
        )

        assessment = assess_validation(_hybrid_snapshot(response_payload))

        self.assertFalse(assessment.has_grounded_response_payload)
        self.assertEqual(assessment.retry_reason, "unsupported_claims")
        self.assertIn("HYBRID_UPLOAD_SETTING_MISSING", assessment.error_codes)
        self.assertIn("HYBRID_COMPARISON_WEAK", assessment.error_codes)
        self.assertNotIn("VALIDATION_UNSUPPORTED_CLAIMS", assessment.error_codes)

    def test_hybrid_repair_labels_comparison_as_docs_vs_uploaded_settings(self) -> None:
        response_payload = AgentResponsePayloadModel.model_validate(
            {
                "answer": "The docs and upload both use train_test_split.",
                "claims": [
                    {
                        "text": "train_test_split splits arrays into train and test subsets.",
                        "evidence_ids": [
                            "url:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"
                        ],
                        "confidence": 0.9,
                    },
                    {
                        "text": "The uploaded code uses train_test_split.",
                        "evidence_ids": [
                            "path:uploads/demo/sample_pipeline.ipynb#cell=2;chunk=0;start=0;end=96"
                        ],
                        "confidence": 0.8,
                    },
                ],
                "sections": [
                    {
                        "kind": "official_docs",
                        "heading": "Official",
                        "body": "The official docs describe train/test splitting.",
                    },
                    {
                        "kind": "upload_code",
                        "heading": "Upload",
                        "body": "The uploaded code calls train_test_split.",
                    },
                    {
                        "kind": "comparison",
                        "heading": "Comparison",
                        "body": "The uploaded code follows the same train/test split pattern.",
                    },
                ],
                "evidence": [],
                "confidence": 0.85,
            }
        )
        snapshot = _hybrid_snapshot(response_payload)
        assessment = assess_validation(snapshot)

        updates = apply_validation_outcome(
            snapshot=snapshot,
            assessment=assessment,
            attempt=1,
            needs_retry=False,
        )

        sections = {
            section.kind: section.body
            for section in updates["response"].payload.sections
        }
        self.assertIn("test_size=0.2", sections["upload_code"])
        self.assertIn("random_state=42", sections["upload_code"])
        self.assertIn("공식 문서 옵션/기본값:", sections["comparison"])
        self.assertIn("업로드 코드 실제 설정:", sections["comparison"])
        self.assertIn("test_size=0.2", sections["comparison"])

    def test_hybrid_repair_preserves_uploaded_option_order_and_ignores_docs_options(self) -> None:
        docs = EvidenceItem.model_validate({
            **_docs_evidence(),
            "code_metadata": {"option_literals": ["docs_default=True"]},
        })
        upload = EvidenceItem.model_validate({
            **_upload_evidence("train_test_split(X, y, test_size=0.2, random_state=42)"),
            "code_metadata": {
                "option_literals": [" test_size = 0.2 ", "TEST_SIZE=0.2", "shuffle=True"],
            },
        })
        payload = AgentResponsePayloadModel.model_validate({
            "answer": "Official documentation and uploaded code are compared.",
            "claims": [
                {"text": "Official docs describe splitting arrays.", "evidence_ids": [docs.source_id]},
                {"text": "Uploaded code calls train_test_split.", "evidence_ids": [upload.source_id]},
            ],
            "evidence": [docs, upload],
        })
        snapshot = build_validation_snapshot(
            user_input="Compare official train_test_split docs with the uploaded code.",
            planner_output=PlannerOutput(use_retrieval=True, tasks=[
                RetrievalTask(route="docs", query="train_test_split official docs", k=3),
                RetrievalTask(route="upload", query="train_test_split uploaded code", k=3),
            ]),
            parsed_evidence=[docs, upload],
            current_attempt_retrieval_errors=[],
            current_attempt_retrieval_diagnostics=[],
            response_payload=payload,
        )

        repaired = repair_required_sections(payload=payload, snapshot=snapshot)

        sections = {section.kind: section.body for section in repaired.sections}
        expected_options = "test_size = 0.2, shuffle=True, random_state=42"
        self.assertEqual(
            sections["upload_code"].splitlines()[0],
            f"업로드 코드의 실제 설정: {expected_options}.",
        )
        self.assertIn(expected_options, sections["comparison"])
        self.assertNotIn("docs_default", repaired.answer)

    def test_hybrid_unsupported_claims_keeps_valid_claims_and_restates_briefly(self) -> None:
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[
                RetrievalTask(route="docs", query="train_test_split official docs", k=3),
                RetrievalTask(route="upload", query="train_test_split uploaded notebook example", k=3),
            ],
        )
        parsed_evidence = [
            EvidenceItem.model_validate(_docs_evidence()),
            EvidenceItem.model_validate(_upload_evidence()),
        ]
        response_payload = AgentResponsePayloadModel.model_validate(
            {
                "answer": "업로드 예시에서는 test_size=0.2와 random_state=42를 사용합니다. 공식 문서 설명은 근거가 잘못 연결되었습니다.",
                "claims": [
                    {
                        "text": "업로드 예시에서는 test_size=0.2와 random_state=42를 사용합니다.",
                        "evidence_ids": ["path:uploads/demo/sample_pipeline.ipynb#cell=2;chunk=0;start=0;end=96"],
                        "confidence": 0.81,
                    },
                    {
                        "text": "공식 문서 설명은 근거가 잘못 연결되었습니다.",
                        "evidence_ids": ["url:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.does-not-exist.html"],
                        "confidence": 0.12,
                    },
                ],
                "evidence": [],
                "confidence": 0.46,
            }
        )

        snapshot = build_validation_snapshot(
            user_input="train_test_split 공식 문법을 설명하고 업로드 노트북의 실제 사용 예를 찾아줘",
            planner_output=planner_output,
            parsed_evidence=parsed_evidence,
            current_attempt_retrieval_errors=[],
            current_attempt_retrieval_diagnostics=[
                RetrievalDiagnostic(
                    tool="tavily_search",
                    route="docs",
                    status="success",
                    message="",
                    query="train_test_split official docs",
                    attempt=1,
                ),
                RetrievalDiagnostic(
                    tool="upload_search",
                    route="upload",
                    status="success",
                    message="",
                    query="train_test_split uploaded notebook example",
                    attempt=1,
                ),
            ],
            response_payload=response_payload,
        )

        assessment = assess_validation(snapshot)
        self.assertEqual(assessment.retry_reason, "unsupported_claims")

        updates = apply_validation_outcome(
            snapshot=snapshot,
            assessment=assessment,
            attempt=1,
            needs_retry=False,
        )

        answer = updates["response"].final_answer
        self.assertIn("공식 문서 기준으로는", answer)
        self.assertIn("test_size=0.2", answer)
        self.assertIn("근거는 공식 문서 1건", answer)
        self.assertEqual(
            {claim.evidence_ids[0] for claim in updates["response"].payload.claims[:2]},
            {
                "url:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
                "path:uploads/demo/sample_pipeline.ipynb#cell=2;chunk=0;start=0;end=96",
            },
        )

    def test_hybrid_recovery_keeps_surviving_docs_claim_and_only_supplements_missing_upload_route(self) -> None:
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[
                RetrievalTask(route="docs", query="train_test_split official docs", k=3),
                RetrievalTask(route="upload", query="train_test_split uploaded notebook example", k=3),
            ],
        )
        parsed_evidence = [
            EvidenceItem.model_validate(_docs_evidence()),
            EvidenceItem.model_validate(_upload_evidence()),
        ]
        response_payload = AgentResponsePayloadModel.model_validate(
            {
                "answer": "공식 문서에서는 입력 배열을 훈련/테스트 세트로 나눈다고 설명합니다. 업로드 예시는 잘못 연결됐습니다.",
                "claims": [
                    {
                        "text": "공식 문서에서는 입력 배열을 훈련/테스트 세트로 나눈다고 설명합니다.",
                        "evidence_ids": [
                            "url:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"
                        ],
                        "confidence": 0.88,
                    },
                    {
                        "text": "업로드 예시는 잘못 연결됐습니다.",
                        "evidence_ids": ["path:uploads/demo/sample_pipeline.ipynb#cell=999;chunk=0;start=0;end=96"],
                        "confidence": 0.12,
                    },
                ],
                "evidence": [],
                "confidence": 0.5,
            }
        )

        snapshot = build_validation_snapshot(
            user_input="train_test_split 공식 문법을 설명하고 업로드 노트북의 실제 사용 예를 찾아줘",
            planner_output=planner_output,
            parsed_evidence=parsed_evidence,
            current_attempt_retrieval_errors=[],
            current_attempt_retrieval_diagnostics=[
                RetrievalDiagnostic(
                    tool="tavily_search",
                    route="docs",
                    status="success",
                    message="",
                    query="train_test_split official docs",
                    attempt=1,
                ),
                RetrievalDiagnostic(
                    tool="upload_search",
                    route="upload",
                    status="success",
                    message="",
                    query="train_test_split uploaded notebook example",
                    attempt=1,
                ),
            ],
            response_payload=response_payload,
        )

        assessment = assess_validation(snapshot)
        self.assertEqual(assessment.retry_reason, "unsupported_claims")

        updates = apply_validation_outcome(
            snapshot=snapshot,
            assessment=assessment,
            attempt=1,
            needs_retry=False,
        )

        answer = updates["response"].final_answer
        self.assertIn("훈련/테스트 세트로 나눈다고 설명합니다", answer)
        self.assertIn("test_size=0.2", answer)
        self.assertNotIn("train_test_split splits arrays or matrices", answer)
        self.assertIn("근거는 공식 문서 1건과 업로드 파일 1건만 반영했습니다.", answer)


if __name__ == "__main__":
    unittest.main()
