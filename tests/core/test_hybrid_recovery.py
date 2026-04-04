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
    def test_hybrid_unsupported_claims_recovery_preserves_required_route_coverage(self) -> None:
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
                "answer": "업로드 예시는 test_size=0.2와 random_state=42를 사용합니다. [1] 공식 문법 설명은 생략합니다.",
                "claims": [
                    {
                        "text": "업로드 예시는 test_size=0.2와 random_state=42를 사용합니다.",
                        "evidence_ids": ["path:uploads/demo/sample_pipeline.ipynb#cell=2;chunk=0;start=0;end=96"],
                        "confidence": 0.81,
                    },
                    {
                        "text": "공식 문법 설명은 생략합니다.",
                        "evidence_ids": ["url:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.does-not-exist.html"],
                        "confidence": 0.12,
                    },
                ],
                "evidence": [],
                "confidence": 0.46,
            }
        )

        snapshot = build_validation_snapshot(
            user_input="train_test_split 공식 문법을 설명하고 업로드 노트북의 실제 사용 예를 찾아줘.",
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
        self.assertIn("공식 문서 기준", answer)
        self.assertIn("반면", answer)
        self.assertEqual(
            [claim.evidence_ids[0] for claim in updates["response"].payload.claims],
            [
                "url:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
                "path:uploads/demo/sample_pipeline.ipynb#cell=2;chunk=0;start=0;end=96",
            ],
        )


if __name__ == "__main__":
    unittest.main()
