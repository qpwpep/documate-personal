# 벤치마크 가이드

DocuMate 벤치마크는 FastAPI의 `POST /agent` 엔드포인트를 대상으로 하는 온라인 평가만 지원합니다. 실행 진입점은 `src/eval/main.py`이며, 설정 기준은 `data/benchmarks/config.toml`입니다.

## 1. 사전 준비

- FastAPI 서버가 실행 중이어야 합니다.
- `OPENAI_API_KEY`가 설정되어 있어야 합니다.
- judge를 사용할 경우 `JUDGE_MODEL` 또는 config의 기본값이 유효해야 합니다.
- 기본 endpoint는 `http://127.0.0.1:8000`입니다.

권장 실행 순서:

1. `uv sync`
2. `.env` 준비
3. `uv run python -m src.service_manager startweb`
4. benchmark fixture 생성 또는 기존 fixture 확인
5. benchmark 실행

## 2. 주요 명령

### 2.1 fixture 생성

```bash
uv run python -m src.eval.main generate \
  --seed data/benchmarks/fixtures/cases.seed.jsonl \
  --regression-seed data/benchmarks/fixtures/cases.regression.seed.jsonl \
  --out data/benchmarks/fixtures/cases.generated.jsonl \
  --target 120
```

### 2.2 온라인 벤치마크 실행

```bash
uv run python -m src.eval.main run \
  --mode online \
  --track release \
  --fixtures data/benchmarks/fixtures/cases.generated.jsonl \
  --endpoint http://127.0.0.1:8000
```

짧은 smoke run이 필요하면 `--limit`을 사용할 수 있습니다. `--track`를 생략하면 `--limit` 런은 기본적으로 `smoke`로 분류됩니다.

```bash
uv run python -m src.eval.main run \
  --mode online \
  --fixtures data/benchmarks/fixtures/cases.generated.jsonl \
  --limit 10
```

### 2.3 기존 run에서 보고서 재생성

```bash
uv run python -m src.eval.main report --run output/benchmarks/<run_id>
```

### 2.4 README/SVG 이력 갱신

```bash
uv run python -m src.eval.main history
```

release 기준 정본은 이 명령으로 아래를 함께 갱신합니다.

- `README.md`의 9, 10번 섹션
- `docs/assets/benchmark_history.svg`

smoke 히스토리는 release README/SVG를 덮어쓰지 않도록 별도 경로를 명시해야 합니다.

```bash
uv run python -m src.eval.main history \
  --track smoke \
  --readme docs/benchmarking_smoke.md \
  --svg docs/assets/benchmark_history_smoke.svg
```

## 3. 출력 산출물

각 run은 `output/benchmarks/<run_id>/` 아래에 저장됩니다.

| 파일 | 설명 |
|---|---|
| `raw_results.jsonl` | 케이스별 원시 실행 결과 |
| `summary.json` | 집계 지표, gate 판정, 비용/모델 정보, `track`, `requested_limit` |
| `report.md` | 사람이 읽기 쉬운 분석 보고서 |
| `latest_release_run.txt` | 최신 release run id |
| `latest_smoke_run.txt` | 최신 smoke run id |

실무 기준 source of truth:

- 최신 release run 확인: `output/benchmarks/latest_release_run.txt`
- 최신 smoke run 확인: `output/benchmarks/latest_smoke_run.txt`
- 자동 판정 확인: `output/benchmarks/<run_id>/summary.json`
- 상세 해석 확인: `output/benchmarks/<run_id>/report.md`

## 4. Hard Gate 기준

기준 파일은 `data/benchmarks/config.toml`입니다.

| Gate | Threshold |
|---|---:|
| `pass_rate` | `0.90` |
| `tool_precision` | `0.90` |
| `tool_recall` | `0.85` |
| `citation_compliance` | `0.95` |
| `p95_latency_ms` | `20000` |
| `avg_cost_per_case_usd` | `0.01` |

judge minimum score와 pricing도 같은 파일에서 관리합니다.

## 5. 환경 변수 override

`src/eval/main.py`는 아래 환경 변수로 일부 설정을 덮어쓸 수 있습니다. 기본값은 `data/benchmarks/config.toml`, override 정의와 `.env.example` 생성 기준은 `src/settings.py`입니다.

| 이름 | 기본값 | 설명 |
|---|---|---|
| `BENCHMARK_ENDPOINT` | `http://127.0.0.1:8000` | run 명령 기본 endpoint |
| `JUDGE_MODEL` | config 값 사용 | judge 모델 override |
| `BENCHMARK_JUDGE_ENABLED` | config 값 사용 | judge 사용 여부 override |

## 6. 비교 이력 규칙

`src/eval/history.py`는 모든 run을 같은 기준으로 비교하지 않습니다. 먼저 `track`을 분리하고, 그 안에서 아래 두 조건이 같은 run만 comparable run으로 묶습니다.

- `track`
- `fixtures_path`
- `total_cases`

즉, fixture 파일이나 케이스 수가 다르면 README 추세표와 SVG에는 함께 들어가지 않을 수 있습니다.

## 7. 운영 메모

- benchmark는 현재 `online` 모드만 지원합니다.
- `report` 명령은 기존 `summary.json`과 `raw_results.jsonl`이 있어야 합니다.
- `history` 명령은 README 안의 자동 갱신 마커를 기준으로 동작하므로, `README.md`의 `## 9. 최신 벤치마크 결과`와 `## 11. 테스트 및 검증` 제목은 유지해야 합니다.
