# Benchmark Report (20260306_163931)

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
| tool_precision | 0.9062 |
| tool_recall | 0.9667 |
| citation_compliance | 0.8500 |
| p50_latency_ms | 29327.0 |
| p95_latency_ms | 59439.2 |
| avg_cost_per_case_usd | 0.00085588 |

## Hard Gates

| Gate | Threshold | Actual | Passed |
|---|---:|---:|:---:|
| pass_rate | 0.82 | 0.3583 | N |
| tool_precision | 0.9 | 0.9062 | Y |
| tool_recall | 0.85 | 0.9667 | Y |
| citation_compliance | 0.88 | 0.85 | N |
| p95_latency_ms | 20000 | 59439.2 | N |
| avg_cost_per_case_usd | 0.035 | 0.00085588 | Y |

## Failures (Top 20)

| case_id | category | reason |
|---|---|---|
| docs_only_seed_mutation_001 | docs_only | score below threshold |
| docs_only_seed_mutation_002 | docs_only | score below threshold |
| docs_only_seed_mutation_004 | docs_only | score below threshold |
| docs_only_seed_mutation_005 | docs_only | score below threshold |
| docs_only_seed_mutation_006 | docs_only | request timeout |
| docs_only_seed_mutation_007 | docs_only | score below threshold |
| docs_only_seed_mutation_008 | docs_only | score below threshold |
| docs_only_adversarial_001 | docs_only | score below threshold |
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
