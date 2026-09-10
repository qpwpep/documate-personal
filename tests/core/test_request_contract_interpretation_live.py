"""Opt-in route to the complete, artifact-preserving v2 interpretation batch.

The original v1 inputs, expectations, and observed 12/21 result are preserved in
data/benchmarks/request_contracts/history.v1.json. This test never rescores them.
All 126 v2 samples finish and record every dimension before this test asserts.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from uuid import uuid4

import pytest

from src.eval.request_contract_eval import main


@pytest.mark.skipif(os.getenv("LIVE_TEST") != "true", reason="The complete v2 interpretation batch requires LIVE_TEST=true")
def test_live_request_contract_interpretation_batch_v2():
    run_id = os.getenv("REQUEST_CONTRACT_EVAL_RUN_ID") or (
        f"pytest-v2-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    )
    outcome = main(["--live", "--run-id", run_id, "--repeats", "3", "--max-workers", "4"])
    assert outcome == 0, f"The complete result is preserved in output/request_contract_evals/{run_id}/summary.json"
