# Benchmark Report (20260306_123220)

- Mode: `online`
- Endpoint: `http://localhost:8000`
- Fixtures: `data\benchmarks\fixtures\cases.generated.jsonl`
- Overall: `FAIL`

## Metrics

| Metric | Value |
|---|---:|
| total_cases | 120 |
| scored_cases | 120 |
| passed_cases | 30 |
| pass_rate | 0.2500 |
| tool_precision | 0.8824 |
| tool_recall | 0.2000 |
| citation_compliance | 0.0000 |
| p50_latency_ms | 15905.5 |
| p95_latency_ms | 41446.1 |
| avg_cost_per_case_usd | 0.00058941 |

## Hard Gates

| Gate | Threshold | Actual | Passed |
|---|---:|---:|:---:|
| pass_rate | 0.82 | 0.25 | N |
| tool_precision | 0.9 | 0.8824 | N |
| tool_recall | 0.85 | 0.2 | N |
| citation_compliance | 0.88 | 0.0 | N |
| p95_latency_ms | 20000 | 41446.1 | N |
| avg_cost_per_case_usd | 0.01 | 0.00058941 | Y |

## Failures (Top 20)

| case_id | category | reason |
|---|---|---|
| docs_only_seed_mutation_001 | docs_only | score below threshold |
| docs_only_seed_mutation_002 | docs_only | score below threshold |
| docs_only_seed_mutation_003 | docs_only | score below threshold |
| docs_only_seed_mutation_004 | docs_only | score below threshold |
| docs_only_seed_mutation_005 | docs_only | score below threshold |
| docs_only_seed_mutation_006 | docs_only | score below threshold |
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
