# Benchmark Report (20260307_101108)

- Mode: `online`
- Endpoint: `http://localhost:8000`
- Fixtures: `data\benchmarks\fixtures\cases.generated.jsonl`
- Overall: `FAIL`

## Metrics

| Metric | Value |
|---|---:|
| total_cases | 120 |
| scored_cases | 120 |
| passed_cases | 44 |
| pass_rate | 0.3667 |
| tool_precision | 0.9091 |
| tool_recall | 1.0000 |
| citation_compliance | 0.8167 |
| p50_latency_ms | 24374.5 |
| p95_latency_ms | 46977.55 |
| avg_cost_per_case_usd | 0.00081372 |

## Hard Gates

| Gate | Threshold | Actual | Passed |
|---|---:|---:|:---:|
| pass_rate | 0.82 | 0.3667 | N |
| tool_precision | 0.9 | 0.9091 | Y |
| tool_recall | 0.85 | 1.0 | Y |
| citation_compliance | 0.88 | 0.8167 | N |
| p95_latency_ms | 20000 | 46977.55 | N |
| avg_cost_per_case_usd | 0.01 | 0.00081372 | Y |

## Failures (Top 20)

| case_id | category | reason |
|---|---|---|
| docs_only_seed_mutation_001 | docs_only | score below threshold |
| docs_only_seed_mutation_002 | docs_only | score below threshold |
| docs_only_seed_mutation_003 | docs_only | score below threshold |
| docs_only_seed_mutation_004 | docs_only | score below threshold |
| docs_only_seed_mutation_005 | docs_only | score below threshold |
| docs_only_seed_mutation_007 | docs_only | score below threshold |
| docs_only_seed_mutation_008 | docs_only | score below threshold |
| docs_only_adversarial_001 | docs_only | score below threshold |
| docs_only_adversarial_002 | docs_only | score below threshold |
| docs_only_adversarial_003 | docs_only | score below threshold |
| docs_only_adversarial_004 | docs_only | score below threshold |
| docs_only_adversarial_005 | docs_only | score below threshold |
| docs_only_adversarial_006 | docs_only | score below threshold |
| docs_only_adversarial_007 | docs_only | score below threshold |
| docs_only_adversarial_008 | docs_only | score below threshold |
| docs_only_regression_001 | docs_only | score below threshold |
| docs_only_regression_002 | docs_only | score below threshold |
| docs_only_regression_003 | docs_only | score below threshold |
| docs_only_regression_004 | docs_only | score below threshold |
| docs_only_regression_005 | docs_only | score below threshold |
