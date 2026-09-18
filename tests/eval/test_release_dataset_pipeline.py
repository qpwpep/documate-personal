"""Run the entire release dataset through the client with an HTTP test peer.

This is a transport/schema/staging smoke, not a model-quality evaluation. The
peer deliberately emits a neutral response and no tool receipts, so it cannot
be used as evidence that the product meets the new cases' semantic oracles.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from uuid import uuid4

import requests
import pytest

from src.core.contracts.debug import DebugPayload, TokenUsage
from src.eval.config_models import BenchmarkConfig
from src.eval.io import load_cases_jsonl
from src.eval.online_runner import run_online_benchmark
from src.eval.release_dataset import promote
from tests.eval.response_fixtures import answer_provenance, plain_response, sse_http_response


FIXTURES = Path(__file__).resolve().parents[2] / "data/benchmarks/fixtures/cases.generated.jsonl"
SCENARIO_PLAN = {
    "docs_only": {"standard": 20, "boundary": 8, "regression": 2},
    "rag_only": {"standard": 14, "boundary": 6, "injection": 8, "regression": 2},
    "hybrid": {"standard": 14, "boundary": 6, "injection": 8, "regression": 2},
    "tool_action": {"standard": 12, "boundary": 4, "correction": 8, "failure": 4, "regression": 2},
}


def test_release_distribution_matches_the_authored_plan():
    cases = load_cases_jsonl(FIXTURES)
    assert len(cases) == 120
    assert Counter(case.category for case in cases) == {category: 30 for category in SCENARIO_PLAN}
    assert Counter(case.evaluation_role for case in cases) == {"new_evaluation": 112, "public_regression": 8}
    for category, scenarios in SCENARIO_PLAN.items():
        category_cases = [case for case in cases if case.category == category]
        assert Counter(case.scenario for case in category_cases) == scenarios
        assert Counter(case.difficulty for case in category_cases) == {"easy": 8, "medium": 14, "hard": 8}


