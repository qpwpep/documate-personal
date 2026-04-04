import unittest

from src.answer_schema import AgentResponsePayloadModel
from src.contracts import RetrievalDiagnostic
from src.evidence import EvidenceItem
from src.nodes.validation.evidence_validator import assess_validation, build_validation_snapshot
from src.nodes.validation.recovery import apply_validation_outcome
from src.planner_schema import PlannerOutput, RetrievalTask


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


def _upload_evidence() -> dict:
    return {
        "kind": "local",
        "tool": "upload_search",
        "source_id": "path:uploads/demo/sample_pipeline.ipynb#cell=2;chunk=0;start=0;end=96",
        "document_id": "path:uploads/demo/sample_pipeline.ipynb",
        "url_or_path": "uploads/demo/sample_pipeline.ipynb",
        "snippet": "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
        "score": 0.81,
        "cell_id": 2,
        "chunk_id": 0,
        "start_offset": 0,
        "end_offset": 96,
    }


class HybridRecoveryTest(unittest.TestCase):
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

    def test_hybrid_recovery_keeps_surviving_docs_claim_and_only_supplements_missing_local_route(self) -> None:
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
