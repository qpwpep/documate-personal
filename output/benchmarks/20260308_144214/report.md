# Benchmark Report (20260308_144214)

- Mode: `online`
- Endpoint: `http://localhost:8000`
- Fixtures: `data\benchmarks\fixtures\cases.generated.jsonl`
- Overall: `FAIL`

## Metrics

| Metric | Value |
|---|---:|
| total_cases | 120 |
| scored_cases | 120 |
| passed_cases | 43 |
| pass_rate | 0.3583 |
| tool_precision | 0.9085 |
| tool_recall | 0.9933 |
| citation_compliance | 0.7778 |
| p50_latency_ms | 23178.5 |
| p95_latency_ms | 50381.65 |
| avg_cost_per_case_usd | 0.00086081 |

## Hard Gates

| Gate | Threshold | Actual | Passed |
|---|---:|---:|:---:|
| pass_rate | 0.82 | 0.3583 | N |
| tool_precision | 0.9 | 0.9085 | Y |
| tool_recall | 0.85 | 0.9933 | Y |
| citation_compliance | 0.88 | 0.7778 | N |
| p95_latency_ms | 20000 | 50381.65 | N |
| avg_cost_per_case_usd | 0.035 | 0.00086081 | Y |

## Failures (Top 20)

| case_id | category | reason |
|---|---|---|
| docs_only_seed_mutation_001 | docs_only | score below threshold |
| docs_only_seed_mutation_002 | docs_only | score below threshold |
| docs_only_seed_mutation_003 | docs_only | score below threshold |
| docs_only_seed_mutation_004 | docs_only | request timeout |
| docs_only_seed_mutation_005 | docs_only | score below threshold |
| docs_only_seed_mutation_006 | docs_only | score below threshold |
| docs_only_seed_mutation_007 | docs_only | score below threshold |
| docs_only_seed_mutation_008 | docs_only | score below threshold |
| docs_only_adversarial_002 | docs_only | score below threshold |
| docs_only_adversarial_003 | docs_only | score below threshold |
| docs_only_adversarial_004 | docs_only | score below threshold |
| docs_only_adversarial_005 | docs_only | score below threshold |
| docs_only_adversarial_006 | docs_only | score below threshold |
| docs_only_adversarial_008 | docs_only | score below threshold |
| docs_only_regression_001 | docs_only | score below threshold |
| docs_only_regression_002 | docs_only | score below threshold |
| docs_only_regression_003 | docs_only | score below threshold |
| docs_only_regression_004 | docs_only | score below threshold |
| docs_only_regression_005 | docs_only | score below threshold |
| docs_only_regression_006 | docs_only | score below threshold |
