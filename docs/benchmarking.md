# 벤치마크 가이드

DocuMate는 `FastAPI /agent` 엔드포인트를 대상으로 온라인 벤치마크를 실행합니다. 이 문서는 실행 방법, 산출물 위치, Hard Gate 기준만 설명합니다. 최신 수치는 문서에 고정하지 않고, 실제 run 산출물과 README의 운영 상태 요약을 기준으로 확인합니다.

## 1. 실행 순서

### 1.1 케이스 생성

```bash
uv run python -m src.eval.main generate \
  --seed data/benchmarks/fixtures/cases.seed.jsonl \
  --regression-seed data/benchmarks/fixtures/cases.regression.seed.jsonl \
  --out data/benchmarks/fixtures/cases.generated.jsonl \
  --target 120
```

### 1.2 온라인 실행

```bash
uv run python -m src.eval.main run \
  --mode online \
  --fixtures data/benchmarks/fixtures/cases.generated.jsonl \
  --endpoint http://localhost:8000
```

### 1.3 리포트 재생성

```bash
uv run python -m src.eval.main report --run output/benchmarks/<run_id>
```

### 1.4 이력 갱신

```bash
uv run python -m src.eval.main history
```

## 2. 산출물

- `output/benchmarks/<run_id>/raw_results.jsonl`
- `output/benchmarks/<run_id>/summary.json`
- `output/benchmarks/<run_id>/report.md`
- `output/benchmarks/latest_run.txt`

최신 run id는 `output/benchmarks/latest_run.txt`를 기준으로 봅니다. `summary.json`은 기계 판독용 지표와 gate 결과, `report.md`는 사람이 읽기 쉬운 해석과 failure breakdown을 제공합니다. README는 최신 수치의 복사본이 아니라 현재 운영 상태와 참조 경로만 유지합니다.

## 3. Hard Gate

현재 기본 설정은 `data/benchmarks/config.toml`을 source of truth로 사용합니다.

| Gate | Threshold |
|---|---:|
| `pass_rate` | 0.90 |
| `tool_precision` | 0.90 |
| `tool_recall` | 0.85 |
| `citation_compliance` | 0.95 |
| `p95_latency_ms` | 20000 |
| `avg_cost_per_case_usd` | 0.01 |

## 4. 운영 원칙

- 이 문서에 특정 run id나 최신 수치를 고정하지 않습니다.
- 최신 결과 설명이 필요하면 `output/benchmarks/latest_run.txt`가 가리키는 run 디렉터리의 `summary.json`과 `report.md`를 확인합니다.
- README에는 최신 저장 런의 상태 요약과 참조 경로만 남기고, 상세 지표와 failure breakdown은 run 산출물에서 확인합니다.
- 비교 추세가 필요하면 `uv run python -m src.eval.main history`를 실행해 README와 SVG를 갱신합니다.
