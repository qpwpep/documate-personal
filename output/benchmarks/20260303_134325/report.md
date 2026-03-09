# Benchmark Report (20260303_134325)

- Schema version: `legacy`
- Mode: `online`
- Endpoint: `http://localhost:8000`
- Fixtures: `data\benchmarks\fixtures\cases.generated.jsonl`
- Overall: `FAIL`
- Git SHA: `unknown`

## Metrics

| Metric | Value |
|---|---:|
| total_cases | 120 |
| scored_cases | 120 |
| passed_cases | 46 |
| pass_rate | 0.3833 |
| tool_precision | 0.3780 |
| tool_recall | 0.2067 |
| citation_compliance | 0.3056 |
| p50_latency_ms | 49835.5 |
| p95_latency_ms | 62063.0 |
| avg_cost_per_case_usd | 0.00219042 |

## Hard Gates

| Gate | Threshold | Actual | Passed |
|---|---:|---:|:---:|
| pass_rate | 0.82 | 0.3833 | N |
| tool_precision | 0.9 | 0.378 | N |
| tool_recall | 0.85 | 0.2067 | N |
| citation_compliance | 0.88 | 0.3056 | N |
| p95_latency_ms | 20000 | 62063.0 | N |
| avg_cost_per_case_usd | 0.01 | 0.00219042 | Y |

## Category Breakdown

| Category | Cases | Pass Rate | Avg Final | Avg Tool | Avg Citation | Timeouts | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| docs_only | 30 | 0.8000 | 0.8394 | 0.8 | 0.8333 | 5 | 62055.25 |
| hybrid | 30 | 0.1667 | 0.2802 | 0.1667 | 0.0833 | 25 | 62078.0 |
| rag_only | 30 | 0.5333 | 0.5967 | 0.5667 | 0.0 | 0 | 54720.7 |
| tool_action | 30 | 0.0333 | 0.5623 | 0.0333 | 1.0 | 12 | 62062.55 |

## Scenario Breakdown

| Scenario | Cases | Pass Rate | Avg Final | Avg Tool | Avg Citation | Timeouts | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 30 | 0.4333 | 0.5888 | 0.4333 | 0.5 | 11 | 62071.25 |
| ambiguity | 30 | 0.4000 | 0.5967 | 0.4 | 0.5167 | 10 | 62070.7 |
| regression | 30 | 0.5333 | 0.6472 | 0.5667 | 0.4667 | 6 | 62054.7 |
| seed_mutation | 30 | 0.1667 | 0.4458 | 0.1667 | 0.4333 | 15 | 62063.0 |

## Category x Scenario

| Category | Scenario | Cases | Pass Rate | Avg Final | Timeouts |
|---|---|---:|---:|---:|---:|
| docs_only | adversarial | 8 | 1.0000 | 0.9925 | 0 |
| docs_only | ambiguity | 7 | 1.0000 | 0.9971 | 0 |
| docs_only | regression | 7 | 0.7143 | 0.7589 | 2 |
| docs_only | seed_mutation | 8 | 0.5000 | 0.6186 | 3 |
| hybrid | adversarial | 7 | 0.0000 | 0.1562 | 7 |
| hybrid | ambiguity | 8 | 0.1250 | 0.2492 | 7 |
| hybrid | regression | 8 | 0.5000 | 0.5281 | 4 |
| hybrid | seed_mutation | 7 | 0.0000 | 0.1562 | 7 |
| rag_only | adversarial | 8 | 0.6250 | 0.61 | 0 |
| rag_only | ambiguity | 7 | 0.4286 | 0.54 | 0 |
| rag_only | regression | 8 | 0.8750 | 0.79 | 0 |
| rag_only | seed_mutation | 7 | 0.1429 | 0.4171 | 0 |
| tool_action | adversarial | 7 | 0.0000 | 0.5357 | 4 |
| tool_action | ambiguity | 8 | 0.1250 | 0.6434 | 3 |
| tool_action | regression | 7 | 0.0000 | 0.5086 | 0 |
| tool_action | seed_mutation | 8 | 0.0000 | 0.5516 | 5 |

## Timeout Split

| Bucket | Cases | Pass Rate | Avg Final | Timeouts | P95 |
|---|---:|---:|---:|---:|---:|
| timeout | 42 | 0.0000 | 0.2723 | 42 | 62077.95 |
| non_timeout | 78 | 0.5897 | 0.7297 | 0 | 55400.8 |

## Blocking Failures

| Failure | Count |
|---|---:|
| none | 0 |

## Rule Failure Top-N

| Rule | Count |
|---|---:|
| tool_match | 73 |
| citation_compliance | 65 |
| safety_format | 42 |
| content_constraints | 30 |

## Tool Confusion Matrix

| Expected | Observed | Count |
|---|---|---:|
| tavily_search, upload_search | - | 25 |
| tavily_search | tavily_search | 24 |
| slack_notify | - | 17 |
| upload_search | rag_search | 17 |
| upload_search | rag_search, tavily_search | 13 |
| save_text | - | 10 |
| tavily_search | - | 5 |
| tavily_search, upload_search | rag_search, tavily_search | 5 |
| slack_notify | rag_search | 2 |
| slack_notify | slack_notify | 1 |
| tavily_search | rag_search, tavily_search | 1 |

## Route Confusion Matrix

| Expected | Observed | Count |
|---|---|---:|
| - | - | 28 |
| docs, upload | - | 25 |
| docs | docs | 24 |
| upload | local | 17 |
| upload | docs, local | 13 |
| docs | - | 5 |
| docs, upload | docs, local | 5 |
| - | local | 2 |
| docs | docs, local | 1 |

## Judge Availability

- judge_enabled: `True`
- judge_model: `gpt-5-mini`
- judge_unavailable_rate: `0.0000`
- judge_unavailable_count: `0` / `78`

## Failures (Top 20)

| case_id | category | reason |
|---|---|---|
| docs_only_seed_mutation_001 | docs_only | request timeout |
| docs_only_seed_mutation_004 | docs_only | request timeout |
| docs_only_seed_mutation_005 | docs_only | request timeout |
| docs_only_seed_mutation_008 | docs_only | score below threshold |
| docs_only_regression_004 | docs_only | request timeout |
| docs_only_regression_006 | docs_only | request timeout |
| rag_only_seed_mutation_001 | rag_only | score below threshold |
| rag_only_seed_mutation_002 | rag_only | score below threshold |
| rag_only_seed_mutation_004 | rag_only | score below threshold |
| rag_only_seed_mutation_005 | rag_only | score below threshold |
| rag_only_seed_mutation_006 | rag_only | score below threshold |
| rag_only_seed_mutation_007 | rag_only | score below threshold |
| rag_only_adversarial_002 | rag_only | score below threshold |
| rag_only_adversarial_003 | rag_only | score below threshold |
| rag_only_adversarial_008 | rag_only | score below threshold |
| rag_only_regression_007 | rag_only | score below threshold |
| rag_only_ambiguity_001 | rag_only | score below threshold |
| rag_only_ambiguity_004 | rag_only | score below threshold |
| rag_only_ambiguity_006 | rag_only | score below threshold |
| rag_only_ambiguity_007 | rag_only | score below threshold |
