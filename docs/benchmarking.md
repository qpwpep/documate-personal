# 벤치마크 가이드

DocuMate release 벤치마크는 Streamlit과 같은 `src/app/client.py`의 `AgentSessionClient`와 `src/app/uploads.py`를 사용합니다. 요청 구성, 첨부 준비·동기화, SSE 처리, 최종 `AnswerResponse`·`UploadManifest` 검증을 공유하며, 실행은 일반 FastAPI 첨부 API와 `POST /agent/stream`을 거칩니다. `src/eval/online_runner`는 시나리오 순서·세션 격리·결과 수집을 담당하고 HTTP 요청이나 에이전트 실행을 별도로 구현하지 않습니다. CLI 진입점은 `src/eval/main.py`, 설정 기준은 `data/benchmarks/config.toml`입니다. planner 단독 요청 계약 평가는 별도 회귀 진단으로 유지하며 2.6에 정리했습니다.

평가 category는 `docs_only`, `rag_only`, `hybrid`, `tool_action`입니다. `rag_only`와 fixture의 `require_local_citation`은 파일 검색·인용을 평가하는 분류명입니다. 현재 파일 검색 도구는 `upload_search`, 검색 route와 snapshot의 source type은 `upload`입니다. 과거 결과에 남은 `local` route와 `rag_search` 호출은 당시 실행 기록이며 새 실행의 업로드 검색 충족으로 인정하지 않습니다.

온라인 평가는 SSE `final_response.data`의 `response`, `trace`, `debug` 전체를 읽습니다. 답변 평가 입력인 `response`는 `AnswerResponse`이며, 실제 표시한 `content.blocks`, 사용한 `citations`, 내용별 `checks`, `issues`, `actions`를 평가합니다. 출처 연결은 현재 검색의 `observed_hits`, 서버가 선택한 선행 답변, 최종 결과의 `answer_provenance.evidence_packet`을 구분해 검증합니다. 별도 답변 문자열이나 주장 목록을 추출해 대신 평가하지 않습니다.

벤치마크는 공용 클라이언트에 `include_debug=true`를 지정합니다. HTTP `200`, 첫 진행 이벤트, `done` 수신만으로 성공을 판정하지 않습니다. HTTP 오류, SSE `error`, 연결 단절, timeout, 최종 응답 누락, 응답 계약 오류를 구분하고, `error` 뒤의 최종 응답도 앞선 오류와 함께 보존합니다. 잘못된 manifest는 UI와 마찬가지로 사용 가능한 답변으로 승인하지 않으며 원본 envelope는 진단에 남깁니다. 유효한 최종 응답을 수신해도 benchmark 통과를 의미하지는 않습니다. 질문은 자동 재전송하지 않으며 redirect나 JSON endpoint fallback도 사용하지 않습니다.

## 1. 사전 준비

- FastAPI 서버가 실행 중이어야 합니다.
- 업로드 사례는 서버와 같은 공유 파일시스템을 사용해야 합니다. `/uploads/sync`는 파일 bytes 업로드가 아니라 준비된 파일의 경로를 받는 JSON API입니다.
- `OPENAI_API_KEY`가 설정되어 있어야 합니다.
- judge를 사용할 경우 `JUDGE_MODEL` 또는 config의 기본값이 유효해야 합니다.
- 기본 endpoint는 `http://127.0.0.1:8000`입니다. `--endpoint`와 `BENCHMARK_ENDPOINT`에는 공용 클라이언트가 호출할 FastAPI 기본 주소를 지정합니다.

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

각 사례는 새 세션에서 시작합니다. `upload_fixtures`에 나열한 파일들을 UI와 같은 staging·동기화 절차로 등록한 뒤, `setup_turns`를 순서대로 보내고 마지막 `query`만 사례의 기대값으로 채점합니다. 각 턴의 최종 manifest를 다음 질문에 사용하며 내부 대화 상태를 직접 주입하지 않습니다. 준비 턴의 실행·응답·필수 진단이 실패하면 후속 요청을 보내지 않고 해당 사례를 실패로 기록합니다. judge에는 실제 준비 질문·답변·검색 근거도 전달합니다.

fixture의 최소 예시는 다음과 같습니다. 기존 단일 `upload_fixture`도 읽을 수 있지만 `upload_fixtures`와 동시에 지정할 수 없습니다.

```json
{
  "case_id": "save_previous",
  "category": "tool_action",
  "setup_turns": ["다음 메모를 두 문장으로 정리해줘: CSV를 읽고 결측 행을 제거한 뒤 날짜별로 집계한다."],
  "query": "방금 답변을 txt로 저장해줘.",
  "upload_fixtures": [],
  "expected_tools": ["save_text"]
}
```

현재 120개 fixture에는 준비 턴을 포함한 액션 사례 30개와 복수 파일 사례가 있습니다. 따라서 전체 정상 실행은 최종 평가 질문 120회와 준비 질문 30회, 첨부 API 요청 및 별도 judge 호출로 구성됩니다.

```bash
uv run python -m src.eval.main run \
  --mode online \
  --track release \
  --fixtures data/benchmarks/fixtures/cases.generated.jsonl \
  --endpoint http://127.0.0.1:8000
```

release track은 judge 평가가 필수입니다. `judge_enabled=false`인 release 실행은 시작 전 설정 검증에서 종료 코드 2로 거부하며, 실행 결과가 release 기준에 미달하면 산출물을 저장한 뒤 종료 코드 1을 반환합니다. judge가 정상 완료되고 모든 gate가 통과한 release만 종료 코드 0입니다.

짧은 smoke run이 필요하면 `--limit`을 사용할 수 있습니다. `--track`를 생략하면 `--limit` 런은 기본적으로 `smoke`로 분류됩니다. `judge_enabled=false`로 실행한 smoke는 `judge_status=disabled`의 규칙 진단 결과만 기록하며 release 통과로 표시하지 않고, 정상 완료 시 종료 코드 0을 반환합니다.

```bash
uv run python -m src.eval.main run \
  --mode online \
  --fixtures data/benchmarks/fixtures/cases.generated.jsonl \
  --limit 10
```

### 2.3 live Slack 전송을 켠 benchmark 실행

`--live-slack` 또는 `BENCHMARK_SLACK_ENABLED=true`는 fixture의 목적지를 지정한 실제 목적지로 바꾸고 전송 성공 감사를 활성화합니다. 서버의 Slack 실행을 차단하는 스위치는 아닙니다. 비활성 상태에서도 fixture 목적지는 일반 요청에 전달되며, 서버에 토큰과 실행 가능한 본문이 있으면 Slack API 호출을 시도할 수 있습니다. 외부 전송을 하지 않는 검증은 Slack 토큰·기본 목적지가 없는 별도 서버나 HTTP 대체 경계를 사용해야 합니다. 파일 저장도 일반 저장 도구를 실제 실행합니다.

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

### 2.6 요청 계약 실제 모델 평가

제품 정책과 기대값은 [`policy.v2.json`](../data/benchmarks/request_contracts/policy.v2.json), [`cases.v2.jsonl`](../data/benchmarks/request_contracts/cases.v2.jsonl)에 실행 전에 고정합니다. 기존 21개와 추가 21개를 각 3회, 총 126개 표본으로 평가합니다. 전체 성공률 95% 이상, 계약 수용률 99% 이상, 기존 21개 모두 3/3 성공, 심각 실패 0건을 동시에 요구합니다. 추가 21개도 직접 작성한 회귀 사례이므로 독립적인 일반화 성능 표본이라고 주장하지 않습니다.

```bash
uv run python -m src.eval.request_contract_eval \
  --live --run-id v2_run_01 --repeats 3
```

`--run-id`는 아직 존재하지 않는 이름이어야 합니다. `--plan-only`는 모델 호출 없이 실행 manifest만 만들며 `--live`와 함께 사용할 수 없습니다. `--cases`로 일부 사례를 지정하거나 반복 횟수를 줄인 실행은 진단용이며 전체 통과로 인정하지 않습니다. 수정 후 실행은 `--comparison-run`과 `--change-note`로 비교 대상과 변경 근거를 실행 전에 기록할 수 있습니다. `--model`, `--max-tokens`는 명시적인 비교 조건으로 기록하고 `.env`를 변경하지 않습니다.

이 평가는 현재 설정된 planner 모델과 실제 계약 확정 코드를 사용합니다. 계약 유효성, 행동 의사·목적지, 본문 작업·정확한 참조, 내용·형식·선호, 질문·부족한 정보, 요청 ID·revision과 보류 상태를 분리해 채점합니다. 모델이 요청되지 않은 행동을 허용한 오류는 서버가 차단했더라도 원시 출력의 심각 실패로 기록합니다. `overall`은 이 해석 과제의 모든 차원이 성공한 표본이며 실제 검색·합성·파일·Slack까지 실행한 성공률이 아닙니다. 전체 전달 상태 전이는 pytest의 실제 임시 파일·localhost Slack 대체 서버 검사로 검증합니다.

각 실행은 `output/request_contract_evals/<run_id>/`에 덮어쓰기 없이 보존합니다.

| 파일 | 내용 |
| --- | --- |
| `manifest.json` | 실행 전 표본 순서·반복 횟수·정책·모델 설정, 코드·스키마·프롬프트·fixture hash |
| `results.jsonl` | 실패를 포함한 모든 표본의 모델 원문·사용량·시간·계약·오류·차원별 판정 |
| `summary.json` | 차원별 실패, 각 사례의 3회 변동성, 통과 기준 판정 |

실행 중 코드가 바뀌면 해당 배치는 통과로 인정하지 않습니다. 결함 수정 뒤에는 새 실행 ID로 결과를 남기며 기존 실패를 삭제하거나 성공한 재실행만 합산하지 않습니다. 최초 v1의 **12/21 통과·9/21 실패**와 별도 진단 재실행은 [`history.v1.json`](../data/benchmarks/request_contracts/history.v1.json)에 보존합니다. v1 원시 응답 전체는 당시 파일로 저장되지 않았으며 이 이력은 남아 있는 실행 기록을 근거로 작성했습니다. v2 정책으로 v1 결과를 다시 채점하거나 두 수치로 개선율을 계산하지 않습니다.

### 2.6.1 현재 검증 기록

2026-09-10 KST 기준 전체 회귀 검사는 `838 passed, 109 skipped, 67 subtests passed`입니다. 실제 모델 평가는 `gpt-5.6-luna`, `planner_max_tokens=1920`, `temperature=0`, 전체 배치 동시 요청 수 4로 실행했습니다. 모든 결과는 아래 실행 ID의 로컬 `output/request_contract_evals/` 디렉터리에 보존합니다.

| 실행 ID | 구분 | 전체 해석 성공 | 계약 수용 | 판정 |
| --- | --- | ---: | ---: | --- |
| `v2-20260910-schema-01` | 1건 스키마 호환성 진단 | 0/1 | 1/1 | 스키마는 수용했으나 불필요한 추가 질문 발생. 전체 배치에서 제외 |
| `v2-20260910-run-01` | 첫 42개 × 3회 실제 모델 배치 | 88/126 (69.84%) | 114/126 (90.48%) | 품질 기준 미달 |
| `v2-20260910-run-02` | 계약·상태 처리 수정 후 같은 정책·사례·모델 설정으로 실행 | 103/126 (81.75%) | 122/126 (96.83%) | 품질 기준 미달 |
| `v2-20260910-replay-02` | 목적지 보존 수정 후 두 번째 배치의 원시 응답 재검증 | 105/126 (83.33%) | 122/126 (96.83%) | 새 모델 호출 0건. 독립적인 실모델 표본이나 품질 통과 배치로 사용하지 않음 |
| `v2-20260910-astra-schema-01` | `gpt-6-astra` 호환성 진단 1건 | 생성 응답 없음 | 생성 응답 없음 | `temperature=0` 미지원으로 HTTP 400. 추가 Astra 호출 없음 |

두 전체 실모델 배치 모두 실행 중 코드 변경 없이 완료했으며 심각 실패는 0건이었습니다. 두 번째 배치의 기존 사례는 59/63 표본 성공, 21개 중 17개 사례가 3/3 성공했습니다. 따라서 기존 사례 전부 3/3, 전체 95%, 계약 수용 99% 기준을 충족하지 못합니다. 사례별 변동성은 각 `summary.json`의 `cases`, 반복별 원시 결과는 `results.jsonl`에 남깁니다.

두 번째 배치의 차원별 실패는 계약 유효성 4건, 행동·목적지 6건, 본문 작업·참조 3건, 내용·형식 요구 7건, 질문 필요성 3건, 부족한 정보 6건, 상태 전이 10건입니다. 한 표본이 여러 차원에서 실패할 수 있으므로 합산하지 않습니다. 정확한 인용 범위, 정정·보충·취소의 분류, 형식 제약과 부족한 정보의 해석에서 실패가 남았습니다. 모든 모델 응답은 정상 종료했고 최대 출력량은 첫 배치 877, 두 번째 배치 690 토큰이어서 출력 한도 부족을 원인으로 판단하지 않았습니다.

첫 배치 이후에는 근거 ID를 서버의 요청·revision 범위에 결합하고, 확인된 과거 사실을 모델이 다시 작성하지 않도록 전달했습니다. 순수 금지의 확인 응답, 줄 수 형식의 타입 제약, 미완료 본문·행동 의사 보존도 수정했습니다. 두 번째 배치에서 발견한 별도 결함은 행동 금지 시 명시된 목적지까지 제거하던 처리였으며, 목적지 사실을 유지하도록 고친 뒤 실패를 재현한 회귀 테스트와 원시 응답 재검증으로 확인했습니다. 새 전체 실모델 배치는 이 마지막 수정 이후 실행하지 않았습니다.

평가 정책·사례·채점 기준은 첫 모델 호출 전에 고정한 버전을 유지했습니다. 프로젝트 기본 모델과 `.env`를 변경하지 않았으며, Astra 진단은 현재 모델에서 남은 실패를 비교하려던 별도 시도입니다. 실제 모델 평가에는 planner만 연결했고, 실제 저장·Slack 소비 동작은 pytest의 임시 파일·localhost HTTP 대체 서버로 확인했습니다. 실서비스 Slack 전송, 실제 검색·합성을 포함한 전체 120-case release benchmark는 이번 검증에 포함하지 않았습니다.

## 3. 출력 산출물

각 run은 `output/benchmarks/<run_id>/` 아래에 저장됩니다. 최신 run 포인터는 run 디렉터리 안이 아니라 `output/benchmarks/` 루트에 저장됩니다. `output/` 전체는 Git 추적 대상이 아닌 로컬 실행 산출물입니다.

| 파일 | 설명 |
|---|---|
| `raw_results.jsonl` | 최종 평가 결과, 모든 `scenario_turns`의 요청·응답·manifest·debug·오류, 소비한 첨부 hash와 정리 오류 |
| `summary.json` | 집계 지표, gate 판정, 비용/모델 정보, track, 제한 수, 실행·측정 계약, 채점 버전과 비교 fingerprint |
| `report.md` | 사람이 읽기 쉬운 분석 보고서 |
| `request_map.jsonl` | 사례의 최종 평가 요청을 기준으로 한 session/request ID, query hash, trace 매핑. 준비 턴은 `raw_results.jsonl`에서 확인 |
| `output/benchmarks/latest_release_run.txt` | 최신 release run id를 가리키는 루트 포인터 |
| `output/benchmarks/latest_smoke_run.txt` | 최신 smoke run id를 가리키는 루트 포인터 |

`summary.json`의 `judge_model`은 config와 환경 변수 override를 모두 반영해 실제 실행에 적용된 effective judge model입니다.

시간과 비용의 범위는 다음과 같습니다. 시간 값이 `null`이면 미측정이며 0으로 대체하지 않습니다.

| 필드 | 측정 범위 |
|---|---|
| `attachment_setup_ms` | 사례 시작부터 최초 첨부 목록 조회, 로컬 파일 읽기·staging, 동기화와 인덱스 준비까지. 준비 실패 시 실패 확인까지 |
| `question_response_ms` | 최종 평가 질문 POST부터 final 수신까지. 공용 응답 검증과 `done` 대기는 제외. 실패 시 오류 확인까지, 질문을 보내지 않았으면 `null` |
| `latency_ms_e2e` | 기존 소비자를 위한 `question_response_ms` 별칭. 브라우저 렌더링 시간은 아님 |
| `scenario_total_ms` | 최초 준비와 모든 준비 질문·최종 질문, 응답 검증·평가 입력 해석까지. judge·결과 파일 저장·정리 요청은 제외 |
| `cost_usd` | 실행한 모든 턴에서 관측한 앱 LLM 비용 합계. judge·검색 provider·임베딩 비용은 포함하지 않음 |

최상위 `token_usage`, `llm_calls`, `models_used`, `debug`는 최종 평가 질문의 정보입니다. 모든 턴의 정보는 `scenario_turns`에 남습니다. summary와 보고서에는 세 시간 구간의 p50/p95를 표시하며 `p95_latency_ms` gate는 최종 질문 시간을 평가합니다. 이전 방식에서 질문 시간에 포함되던 초기화·인덱스 준비가 첨부 구간으로 이동했으므로 과거 수치와 직접 비교하지 않습니다.

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

judge minimum score와 pricing도 같은 파일에서 관리합니다. `cost_gate_min_llm_call_coverage`는 `src/eval/config_models.py::HardGates`의 기본값이며, config에 명시하지 않으면 `0.80`이 적용됩니다. 다중 턴 비용 관측률은 모든 실행 턴을 검사합니다. 준비 턴에 LLM 사용량이 있고 마지막 저장이 deterministic이면 인정하지만, 어느 턴의 사용량이 누락되면 최종 질문의 정상 진단만으로 비용 gate를 활성화하지 않습니다.

### 4.0.1 평가 상태와 release 판정

사례별 결과는 평가 실행 상태와 답변 품질 합격을 분리해 기록합니다.

- `judge_status`: `disabled`(설정으로 비활성화), `not_run`(제품 응답 부재·입력 불완전 등으로 미실행, 사유는 `judge_status_reason`), `failed`(호출 실패 또는 출력 계약 오류), `succeeded`(입력 완비·호출·출력 계약 통과. 품질 합격이 아님), `legacy_unknown`(상태 필드가 없는 과거 결과).
- `eval_validity`: `valid`(현재 정책의 평가가 완료되어 판정 가능, 품질 불합격도 유효한 평가), `incomplete`(필수 입력·단계 부족), `invalid`(judge 호출 실패·출력 계약 오류로 결과 신뢰 불가), `legacy_unknown`.
- `llm_judge_score`: 유효한 judge 0점은 `0`으로, 평가하지 못한 점수는 `null`로 보존합니다. 평가 오류를 점수로 대체하거나 범위 밖 값을 잘라내지 않습니다.
- `judge_pass`: `succeeded` 평가가 전체 점수와 필수 세부 점수 기준을 모두 충족할 때만 `true`. `disabled`·`not_run`·`failed`는 `null`입니다.
- `product_pass`: 완전한 composite 점수가 있을 때 composite 기준 판정. 제품 실행 실패는 `false`, judge 장애로 composite를 계산할 수 없으면 `null`입니다.
- `release_pass`: 평가 유효성(`valid`), 제품 판정(`product_pass`), judge 품질 판정(`judge_pass`), 응답 계약을 모두 충족할 때만 `true`입니다. `incomplete`·`invalid` 평가는 항상 `false`입니다.

`composite_quality_score`는 judge 점수가 있을 때만 가중합으로 계산합니다. judge 점수가 없으면 남은 규칙 점수를 재정규화해 composite를 만들지 않고 `null`로 남기며, 규칙 원점수는 `rule_scores`·`rule_score_total`에 진단용으로 보존합니다. judge 가용성은 제품 규칙 점수를 바꾸지 않습니다.

모든 카테고리에 `judge_min_score`가 명시적으로 존재합니다(기본 `0.70`, case의 `judge_min_score` 필드가 우선). 세부 점수 기준은 `[judge_min_subscores]`의 `answer_quality`(전 카테고리)와 `groundedness`(근거 기반 답변)이며, 인용을 요구하지 않는 순수 `tool_action` 사례에는 groundedness를 적용하지 않습니다. 이 값들은 인간 평가로 보정된 최적값이 아니라 정책 출발점입니다.

run 집계의 `release_pass_rate` 분모는 실행 대상으로 확정한 전체 사례(`planned_cases`)입니다. composite가 `null`이거나 judge 오류·제품 실패인 사례를 빼지 않으며, 계획 대비 결과 누락·중복·예상외 case_id는 `evaluation_completeness` release gate가 차단합니다. judge 상태별 개수(`judge_succeeded/failed/not_run/disabled_cases`)와 `judge_execution_rate`(judge 필수 대상 중 succeeded 비율)를 별도로 기록하며, 대상이 0개인 비율은 `-`(N/A)로 표시합니다.

### 4.1 참조 연결과 의미적 지지의 구분

| 지표·입력 | 측정 범위 |
|---|---|
| `reference_coverage` | `interaction`을 제외한 실제 내용 단위 중 refs가 있고, 모든 참조가 현재 검색 또는 선택한 선행 답변의 검증된 인용에서 최종 packet으로 연결되는 비율. rule 가중치는 `0.20` |
| `citation_traceability` | 요청한 docs/upload 출처 범위와 최종 packet의 채택 참조를 확인. 현재 검색의 근거는 해당 턴의 도구 실행을 요구하며, 검증된 선행 인용은 재검색 없이 계승 가능 |
| `checks.reference_status` | 런타임의 참조 연결 결과. `resolved`는 의미적 정확성 판정이 아님 |
| `checks.support_status` | `not_evaluated`, `exact_match`, `unsupported` 개수를 별도 집계. `not_evaluated`를 자동으로 0점 처리하지 않음 |
| LLM judge의 groundedness | 실제 표시 본문과 연결된 근거가 설명·해석을 의미적으로 뒷받침하는지 평가 |
| `actions`와 도구 실행 기록 | 저장·전송 성공 여부와 목표 도구 동작을 확인. 본문에 성공 문구가 있다는 이유로 액션 성공으로 판단하지 않음 |

현재 검색에서 packet으로 이어지는 근거는 snapshot과 원문 element가 같고, packet의 문자 범위 또는 표 cell ID가 `observed_hits`의 선택 범위에 포함되는지 확인합니다. synthesis 예산 때문에 범위를 줄이면 새 근거 ID가 생깁니다. 최종 citation은 이렇게 검증한 실제 packet의 ID와 일치해야 하므로, 검색 원문에 있었지만 모델에 제공하지 않은 범위는 인정하지 않습니다. 페이지 bbox가 없더라도 snapshot·요소·선택 범위가 유효하면 원문 연결을 인정합니다.

기존 답변 복사·변환은 debug `answer_provenance.source`의 `ref`, `response_hash`, `citation_ids`로 서버가 선택한 본문과 실제 채택 출처를 확인합니다. 같은 사례·세션의 앞선 턴에서 hash와 전체 citation ID 목록이 모두 일치하는 가장 최근 답변을 찾고, 그 답변의 응답 구조와 출처 연결이 검증되어야 상속을 허용합니다. 상속 가능한 근거도 그 답변의 실제 인용 범위 안에 있는 현재 최종 packet으로 제한합니다. 최종 citation은 이 packet의 정확한 ID와 일치해야 합니다. 단순히 conversation에 등장한 답변이나 이전 턴에서 검색만 한 근거는 합치지 않습니다. 현재 턴의 `observed_hits`, 도구 정밀도·재현율, 원문 복사 감점 입력은 유지하므로 과거 검색 호출이 현재 도구 실행으로 집계되지 않습니다.

`content_hash`는 본문 구조·basis·refs를 포함하지만 citation의 존재 여부나 전달 receipt 전체를 hash하지 않습니다. 따라서 선행 답변의 실제 citation 목록을 함께 비교합니다. 이 연결은 동일한 본문·채택 출처를 확인하며, 같은 내용이 반복된 대화의 정확한 발생 시점이나 설명의 의미적 지지를 증명하지 않습니다. 현재 online 실행에서 provenance 누락·불일치·연결되지 않은 선행 답변은 응답 계약 또는 필수 진단 오류이며, 이전 대화나 검색 원문을 합쳐 성공으로 승격하지 않습니다.

runtime의 exact excerpt 검사는 발췌와 원문의 일치만 보장합니다. 인용된 원문 자체의 진실성이나 답변 전체의 충분함까지 보장하지 않으므로, rule 지표와 judge 결과를 함께 해석해야 합니다.

## 5. 환경 변수 override

`src/eval/main.py`는 아래 환경 변수로 일부 설정을 덮어쓸 수 있습니다. 기본값은 `data/benchmarks/config.toml`, override 정의와 `.env.example` 생성 기준은 `src/infra/settings.py`입니다.

우선순위는 `CLI > .env > OS env > config.toml`입니다.

| 이름 | 기본값 | 설명 |
|---|---|---|
| `BENCHMARK_ENDPOINT` | `http://127.0.0.1:8000` | `/agent/stream`을 붙여 호출할 FastAPI 기본 주소 |
| `JUDGE_MODEL` | config 값 사용 | judge 모델 override |
| `BENCHMARK_JUDGE_ENABLED` | config 값 사용 | judge 사용 여부 override |
| `BENCHMARK_SLACK_ENABLED` | `false` | 실제 목적지 치환·전송 성공 감사 활성화. 서버 전송 차단 스위치는 아님 |
| `BENCHMARK_SLACK_CHANNEL_ID` | 없음 | live channel 케이스 전송용 Slack channel id |
| `BENCHMARK_SLACK_USER_ID` | 없음 | live DM 케이스 전송용 Slack user id |
| `BENCHMARK_SLACK_EMAIL` | 없음 | live DM 케이스 전송용 Slack email |

## 6. 비교 이력 규칙

history 리포터는 다음 조건이 모두 같은 run만 comparable run으로 묶습니다.

- `track`
- `fixtures_path`
- `total_cases`
- `execution_contract_version`: 현재 `shared-client-scenario-v1`
- `measurement_contract_version`: 현재 `attachment-question-scenario-v1`
- `suite_fingerprint`: 준비 턴·첨부 목록을 포함한 사례 내용과 staging에서 실제 읽은 첨부 bytes의 SHA256
- `evaluation_fingerprint`: 채점 계약 버전과 가중치·gate·pricing·judge·timeout 설정, Slack 실행 옵션

같은 경로의 fixture를 덮어써도 내용이나 첨부 bytes가 달라지면 자동 비교되지 않습니다. 새 계약 정보가 일부만 있는 run은 자기 자신만 표시합니다. 계약 정보가 없는 legacy끼리는 이전 경로·사례 수 비교를 유지하지만, 새 실행과 섞지 않습니다. 과거 JSON을 읽을 때 새 계약으로 자동 승격하지 않습니다.

현재 채점 계약은 `judge-state-contract-v2`이며 `summary.json`의 `audit_metrics.scoring_contract_version`에 기록합니다. 이 버전도 `evaluation_fingerprint`에 포함하므로 같은 fixture·설정이라도 이전 채점 결과와 자동 비교하지 않습니다. 실행·측정·채점 계약의 의미를 바꾸면 해당 버전도 갱신해야 합니다. 과거 summary는 당시 값 그대로 읽고 새 채점 버전을 채워 넣지 않습니다. 원격 서버의 모든 설정이나 모델 provider의 변동을 fingerprint가 자동 고정하지는 않습니다. 비교할 변경은 동일한 평가 계약·fixture와 확인된 서버 설정에서 다시 실행합니다. 현재 rule의 `reference_coverage`는 의미적 groundedness를 측정하지 않으며, 이전 스키마·rule의 기록을 새 계약의 품질 상승·하락 근거로 사용하지 않습니다.

## 7. 운영 메모

- benchmark는 현재 `online` 모드만 지원합니다.
- 확정된 첨부는 사례 종료 후 일반 sync API의 clear로 해제하고 staging 파일은 공용 정리 함수를 사용합니다. 질문 결과나 sync 처리 여부가 불확실하면 재전송·세션 폴더 직접 삭제를 하지 않습니다. 남은 세션·파일은 일반 TTL/LRU와 서버 파일 정리 주기에 따르며, 후속 요청이나 서버 재시작의 정리 실행까지 남을 수 있습니다. 정리 요청 실패는 `cleanup_errors`로 기록합니다.
- 공용 클라이언트 평가는 Streamlit 화면 렌더링·브라우저 업로드 시간·다운로드 클릭을 측정하지 않습니다. UI 회귀는 Streamlit AppTest와 첨부 통합 테스트로 별도 확인합니다.
- `history` 명령은 README 안의 자동 갱신 마커를 기준으로 동작하므로, `README.md`의 `## 검증 결과`와 `## 문서` 제목은 유지해야 합니다.
- 공개 release 결과의 정본은 README 요약이고, 비교 추세는 기존 benchmark history SVG에 유지합니다. 별도 결과 문서나 smoke history 파일은 만들지 않습니다.
