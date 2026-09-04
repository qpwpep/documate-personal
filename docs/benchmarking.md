# 벤치마크 가이드

DocuMate 벤치마크는 FastAPI의 `POST /agent` 엔드포인트를 대상으로 하는 온라인 평가만 지원합니다. 실행 진입점은 `src/eval/main.py`이며, 설정 기준은 `data/benchmarks/config.toml`입니다.

평가 category는 `docs_only`, `rag_only`, `hybrid`, `tool_action`입니다. `rag_only`는 기존 fixture와 결과의 호환성을 위한 분류명이며, 현재 fixture의 `rag_only`와 `hybrid` 파일 검색은 `upload_search`를 기대합니다. 런타임 검색 route는 `docs`, `upload`이고, 업로드 evidence의 `kind="local"`과 fixture의 `require_local_citation`은 파일 근거 인용을 나타냅니다. 과거 결과에 남은 `local` route와 `rag_search` 호출은 당시 실행 기록으로 해석합니다.

## 1. 사전 준비

- FastAPI 서버가 실행 중이어야 합니다.
- `OPENAI_API_KEY`가 설정되어 있어야 합니다.
- judge를 사용할 경우 `JUDGE_MODEL` 또는 config의 기본값이 유효해야 합니다.
- 기본 endpoint는 `http://127.0.0.1:8000`입니다.

권장 실행 순서:

1. `uv sync`
2. `.env` 준비
3. `uv run python -m src.app.service_manager startweb`
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

### 2.3 live Slack 전송을 켠 benchmark 실행

실제 Slack 전송은 기본적으로 꺼져 있으며, `--live-slack` 또는 `BENCHMARK_SLACK_ENABLED=true`일 때만 동작합니다.

- benchmark CLI의 override 우선순위는 `CLI > .env > OS env > config.toml`입니다.
- 즉 `.env`에 `BENCHMARK_SLACK_*`, `BENCHMARK_ENDPOINT`, `JUDGE_MODEL`, `BENCHMARK_JUDGE_ENABLED`를 넣으면 별도 export 없이도 benchmark CLI가 그대로 읽습니다.

- fixture의 `C123BENCH`, `U123BENCH`는 live 모드에서 실제 목적지가 아니라 케이스 분류 힌트로만 사용됩니다.
- channel 케이스는 `--live-slack-channel-id` 또는 `BENCHMARK_SLACK_CHANNEL_ID`가 필요합니다.
- DM 케이스는 `--live-slack-user-id`, `--live-slack-email`, `BENCHMARK_SLACK_USER_ID`, `BENCHMARK_SLACK_EMAIL`, 또는 app 기본 DM 설정을 사용합니다.

```bash
uv run python -m src.eval.main run \
  --mode online \
  --track release \
  --fixtures data/benchmarks/fixtures/cases.generated.jsonl \
  --endpoint http://127.0.0.1:8000 \
  --live-slack \
  --live-slack-channel-id C0123456789 \
  --live-slack-user-id U0123456789
```

live Slack 실행에서는 `summary.json`과 `report.md`에 Slack delivery audit 지표가 추가됩니다. 이 지표는 audit-only이며 release gate를 직접 차단하지는 않습니다.

### 2.4 기존 run에서 보고서 재생성

```bash
run_id="$(<output/benchmarks/latest_release_run.txt)"
uv run python -m src.eval.main report --run "output/benchmarks/$run_id"
```

`run` 명령이 갱신한 release 포인터를 사용하는 예시입니다. smoke 보고서를 재생성하려면 `latest_smoke_run.txt`를 읽습니다. `report`는 대상 run에 `summary.json`과 `raw_results.jsonl`이 모두 있는지 검증한 뒤 같은 디렉터리의 `report.md`만 다시 씁니다.

### 2.5 release 요약과 benchmark history SVG 갱신

```bash
uv run python -m src.eval.main history --track release
```

`history`는 로컬 `output/benchmarks/*/summary.json`을 읽고 release run을 선택해 저장소에서 유지하는 두 공개 산출물을 함께 갱신합니다.

- `README.md`의 `## 검증 결과` 섹션
- `docs/assets/benchmark_history.svg`

최신 release 선택에는 `output/benchmarks/latest_release_run.txt`를 우선 사용합니다. 포인터가 없거나 가리키는 release run을 찾지 못하면 release track에서 전체 케이스 수가 가장 큰 run들 중 가장 최근 `summary.json`으로 fallback합니다. README 요약과 SVG의 지표는 `summary.json`에서 읽으며, `report.md`는 history 입력이 아니라 사람이 확인하거나 `report` 명령으로 재생성하는 로컬 상세 보고서입니다.

이미 존재하는 release run을 공개 요약에 반영할 때는 full benchmark를 다시 실행할 필요가 없습니다. 새 release 수치가 필요할 때만 2.2의 `run` 명령을 먼저 실행합니다. `history`는 pytest를 실행하지 않고 README 표에 기록된 기존 테스트 결과를 보존하므로, 테스트 수치를 바꾸려면 `uv run pytest -q`로 별도 검증한 뒤 README의 테스트 행을 갱신해야 합니다.

SVG는 현재 로컬에 남아 있는 comparable release summary만으로 다시 생성됩니다. 의도한 과거 release run의 `summary.json`이 모두 있는지 확인한 뒤 실행해야 기존 추세 지점이 빠지지 않습니다.

스크린샷과 데모 GIF는 `history` 명령이 갱신하지 않습니다. 실제 앱 캡처를 갱신한 뒤 공개용 자산만 `docs/assets/demo-final.png`, `docs/assets/demo-flow.gif`로 별도 저장하고 README에서 이 경로를 참조합니다.

저장소는 별도 smoke history 문서나 SVG를 유지하지 않습니다. smoke 결과는 해당 run의 `summary.json`과 `report.md`에서 확인하며, 일반적인 smoke 실행 뒤에는 `history`를 실행하지 않습니다. CLI도 smoke track이 기본 release README 또는 SVG를 덮어쓰지 못하게 차단합니다.

## 3. 출력 산출물

각 run은 `output/benchmarks/<run_id>/` 아래에 저장됩니다. 최신 run 포인터는 run 디렉터리 안이 아니라 `output/benchmarks/` 루트에 저장됩니다. `output/` 전체는 Git 추적 대상이 아닌 로컬 실행 산출물입니다.

| 파일 | 설명 |
|---|---|
| `raw_results.jsonl` | 케이스별 원시 실행 결과 |
| `summary.json` | 집계 지표, gate 판정, 비용/모델 정보, `track`, `requested_limit` |
| `report.md` | 사람이 읽기 쉬운 분석 보고서 |
| `request_map.jsonl` | 케이스별 `session_id`, `request_id`, query hash, trace 매핑 |
| `output/benchmarks/latest_release_run.txt` | 최신 release run id를 가리키는 루트 포인터 |
| `output/benchmarks/latest_smoke_run.txt` | 최신 smoke run id를 가리키는 루트 포인터 |

`summary.json`의 `judge_model`은 config와 환경 변수 override를 모두 반영해 실제 실행에 적용된 effective judge model입니다.

산출물 역할:

- 공개 release 요약: `README.md`의 `## 검증 결과`
- 공개 release 추세: `docs/assets/benchmark_history.svg`
- 로컬 최신 run 선택: `output/benchmarks/latest_release_run.txt`, `output/benchmarks/latest_smoke_run.txt`
- 로컬 기계 판정과 집계 정본: `output/benchmarks/<run_id>/summary.json`
- 로컬 상세 분석: `output/benchmarks/<run_id>/report.md`

## 4. Hard Gate 기준

기준 파일은 `data/benchmarks/config.toml`입니다.

| Gate | Threshold |
|---|---:|
| `pass_rate` | `0.90` |
| `tool_precision` | `0.90` |
| `tool_recall` | `0.85` |
| `citation_compliance` | `0.95` |
| `p95_latency_ms` | `10000` |
| `avg_cost_per_case_usd` | `0.01` |
| `cost_gate_min_llm_call_coverage` | `0.80` |

judge minimum score와 pricing도 같은 파일에서 관리합니다. `cost_gate_min_llm_call_coverage`는 `src/eval/config_models.py::HardGates`의 기본값이며, config에 명시하지 않으면 `0.80`이 적용됩니다. 비용 지표는 app 응답 생성 LLM 호출 비용 기준이며, 현재 judge 호출 비용은 benchmark cost gate에 포함하지 않습니다.

## 5. 환경 변수 override

`src/eval/main.py`는 아래 환경 변수로 일부 설정을 덮어쓸 수 있습니다. 기본값은 `data/benchmarks/config.toml`, override 정의와 `.env.example` 생성 기준은 `src/infra/settings.py`입니다.

우선순위는 `CLI > .env > OS env > config.toml`입니다.

| 이름 | 기본값 | 설명 |
|---|---|---|
| `BENCHMARK_ENDPOINT` | `http://127.0.0.1:8000` | run 명령 기본 endpoint |
| `JUDGE_MODEL` | config 값 사용 | judge 모델 override |
| `BENCHMARK_JUDGE_ENABLED` | config 값 사용 | judge 사용 여부 override |
| `BENCHMARK_SLACK_ENABLED` | `false` | benchmark live Slack 전송 opt-in |
| `BENCHMARK_SLACK_CHANNEL_ID` | 없음 | live channel 케이스 전송용 Slack channel id |
| `BENCHMARK_SLACK_USER_ID` | 없음 | live DM 케이스 전송용 Slack user id |
| `BENCHMARK_SLACK_EMAIL` | 없음 | live DM 케이스 전송용 Slack email |

## 6. 비교 이력 규칙

history 리포터는 모든 run을 같은 기준으로 비교하지 않습니다. 아래 세 조건이 모두 같은 run만 comparable run으로 묶습니다.

- `track`
- `fixtures_path`
- `total_cases`

즉, fixture 파일이나 케이스 수가 다르면 README 요약과 SVG에는 함께 들어가지 않을 수 있습니다.

## 7. 운영 메모

- benchmark는 현재 `online` 모드만 지원합니다.
- `history` 명령은 README 안의 자동 갱신 마커를 기준으로 동작하므로, `README.md`의 `## 검증 결과`와 `## 문서` 제목은 유지해야 합니다.
- 공개 release 결과의 정본은 README 요약이고, 비교 추세는 기존 benchmark history SVG에 유지합니다. 별도 결과 문서나 smoke history 파일은 만들지 않습니다.
