# Runtime Reference

DocuMate 실행, 환경 변수, API 계약, 파일 제약, 운영 메모를 모아 둔 참고 문서입니다. 포트폴리오 관점의 요약은 [README](../README.md), 설계 판단은 [design_rationale.md](design_rationale.md)를 참고하세요.

## 1. 빠른 시작

### 1.1 요구 사항

- Python 3.12 이상
- `uv`
- OpenAI API 키
- Tavily API 키

### 1.2 설치

개발 환경:

```bash
uv sync
```

런타임 의존성만 설치:

```bash
uv sync --no-dev
```

### 1.3 환경 변수 준비

```bash
cp .env.example .env
```

필수 값:

- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

선택 값:

- `SLACK_BOT_TOKEN`
- `SLACK_DEFAULT_USER_ID`
- `SLACK_DEFAULT_DM_EMAIL`

`src.infra.settings.validate_required_keys()` 때문에 FastAPI와 `startweb` 실행 시 현재는 `OPENAI_API_KEY`와 `TAVILY_API_KEY`가 모두 필요합니다.

### 1.4 웹 서비스 실행

권장 방식:

```bash
uv run python -m src.app.service_manager startweb
uv run python -m src.app.service_manager stopweb
```

- FastAPI: `http://127.0.0.1:8000`
- Streamlit: `http://127.0.0.1:8501`
- 런타임 로그: `output/runtime/fastapi.log`, `output/runtime/streamlit.log`
- 상태 파일: `output/runtime/web_services_state.json`

직접 실행도 가능합니다.

```bash
uv run python -X utf8 -m uvicorn src.app.web.app:app --host 0.0.0.0 --port 8000
uv run python -X utf8 -m streamlit run src/app/web/streamlit_app.py --server.port 8501
```

Windows 환경에서는 `-X utf8` 또는 `PYTHONUTF8=1` 사용을 권장합니다. `src.app.service_manager`와 `src.app.web.app`는 UTF-8 실행을 우선하도록 구성돼 있습니다.

## 2. 환경 변수

### 2.1 애플리케이션 설정

기준 파일:

- 기본값 source of truth: `src/infra/settings.py`
- 예시 환경 파일: `.env.example` (생성 산출물)
- 동기화 명령: `uv run python script/sync_env_example.py`

| 이름 | 기본값 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | 없음 | OpenAI 호출과 임베딩 생성에 필요 |
| `TAVILY_API_KEY` | 없음 | 공식 문서 검색에 필요 |
| `CHAT_MODEL` | `gpt-5.4-nano` | synthesis 모델 기본값 |
| `PLANNER_MODEL` | `gpt-5.4-nano` | planner 모델 기본값 |
| `SUMMARY_MODEL` | `gpt-5.4-nano` | session summary 모델 기본값 |
| `PLANNER_MAX_TOKENS` | `1920` | planner structured output 최대 토큰 |
| `DOCS_SEARCH_TIMEOUT_SECONDS` | `5` | Tavily 요청별 timeout |
| `SYNTHESIS_TIMEOUT_SECONDS` | `20` | synthesis provider 요청 timeout |
| `SYNTHESIS_USE_RESPONSES_API` | `false` | synthesis Responses API 사용 여부 |
| `SYNTHESIS_MAX_RETRIES` | `0` | synthesis provider SDK 재시도 횟수 |
| `SYNTHESIS_MAX_TOKENS` | `1920` | synthesis max tokens |
| `SYNTHESIS_PROMPT_SNIPPET_CHARS` | `960` | evidence snippet 길이 제한 |
| `SYNTHESIS_REASONING_EFFORT` | 없음 | synthesis reasoning effort override (none/minimal/low/medium/high/xhigh, 빈 값이면 모델 기본값, none은 명시 override) |
| `VERBOSE` | `true` | 에이전트 런타임 상세 로그 출력 |
| `FASTAPI_URL` | `http://127.0.0.1:8000` | Streamlit이 호출하는 API 주소 |
| `SESSION_TTL_SECONDS` | `1800` | 세션 TTL |
| `MAX_ACTIVE_SESSIONS` | `200` | 최대 활성 세션 수 |
| `SESSION_CLEANUP_INTERVAL_SECONDS` | `60` | 세션 정리 주기 |
| `GENERATED_FILE_TTL_SECONDS` | `86400` | `save_text` 결과 파일 TTL |
| `FILE_CLEANUP_INTERVAL_SECONDS` | `60` | 업로드/생성 파일 정리 주기 |
| `MEMORY_HIGH_WATER_TURNS` | `8` | 대화 compaction을 시작하는 Human turn high watermark |
| `MEMORY_LOW_WATER_TURNS` | `6` | compaction 후 Human turn low watermark |
| `MEMORY_HIGH_WATER_TOKENS` | `32000` | 대화 메모리 추정 token high watermark |
| `MEMORY_LOW_WATER_TOKENS` | `16000` | compaction 후 추정 token low watermark |
| `MEMORY_HIGH_WATER_BYTES` | `98304` | 대화 메모리 UTF-8 직렬화 byte high watermark |
| `MEMORY_LOW_WATER_BYTES` | `49152` | compaction 후 UTF-8 직렬화 byte low watermark |
| `MEMORY_HIGH_WATER_MESSAGES` | `18` | 대화 메시지 수 high watermark |
| `MEMORY_LOW_WATER_MESSAGES` | `14` | compaction 후 메시지 수 low watermark |
| `MEMORY_SUMMARY_MAX_TOKENS` | `256` | rolling summary 출력·저장 추정 token 상한 |
| `MEMORY_SUMMARY_MAX_BYTES` | `4096` | rolling summary UTF-8 byte 상한 |
| `MEMORY_HARD_MAX_BYTES` | `131072` | summary와 최근 메시지를 합친 durable snapshot 절대 byte 상한 |
| `SLACK_BOT_TOKEN` | 없음 | Slack 전송용 토큰 |
| `SLACK_DEFAULT_DM_EMAIL` | 없음 | 기본 DM 대상 이메일 |
| `SLACK_DEFAULT_USER_ID` | 없음 | 기본 DM 대상 사용자 |

모델별 reasoning effort override 참고:

- `gpt-5.4-nano`: none, low, medium, high, xhigh
- `gpt-5-nano`: minimal, low, medium, high

### 2.2 벤치마크 설정

기준 파일:

- 기본값 source of truth: `data/benchmarks/config.toml`
- 환경 변수 override 정의: `src/infra/settings.py`
- 예시 환경 파일: `.env.example` (config 값을 복사한 override 예시)
- 동기화 명령: `uv run python script/sync_env_example.py`

| 이름 | 기본값 | 설명 |
|---|---|---|
| `JUDGE_MODEL` | `gpt-5.4-mini` | benchmark judge 모델 override |
| `BENCHMARK_ENDPOINT` | `http://127.0.0.1:8000` | benchmark 대상 FastAPI 주소 override |
| `BENCHMARK_JUDGE_ENABLED` | `true` | judge 사용 여부 override |
| `BENCHMARK_SLACK_ENABLED` | `false` | benchmark live Slack 전송 opt-in |
| `BENCHMARK_SLACK_CHANNEL_ID` | 없음 | benchmark channel case 전송용 Slack channel id |
| `BENCHMARK_SLACK_USER_ID` | 없음 | benchmark DM case 전송용 Slack user id |
| `BENCHMARK_SLACK_EMAIL` | 없음 | benchmark DM case 전송용 Slack email |

## 3. 검색 소스와 파일 제약

### 3.1 공식 문서 검색

UI와 문서 검색 규칙은 아래 파일을 기준으로 관리합니다.

- `src/core/domain_docs.py`: Streamlit 소개 영역에 노출하는 기본 문서 목록
- `src/infra/config/agent_rules.toml`: docs allowlist, query hint, 저장·전송 요청 감지 규칙

`RULES_CONFIG_PATH`로 규칙 파일을 지정할 수 있습니다. 필요한 검색 출처는 업로드 가용성과 무관하게 LLM이 선택하고, 스키마 검증을 통과한 `PlannerOutput.tasks`를 기준으로 실행합니다. [planner 지침](../src/runtime/nodes/planner/prompt_builder.py)은 일반 기술 설명과 실제 파일 조회를 구분하고 출처 제외 지시를 반영합니다. 파일 조회에 필요한 업로드가 없으면 검색을 진행하지 않고 업로드를 안내합니다.

검색어 후처리는 공백만 정규화해 한국어 주제와 식별자를 보존합니다. LLM 호출이나 출력 검증이 실패하면 `planner_diagnostics.reason="planner_unavailable"`로 기록하고 재요청을 안내하며, 검색·저장·전송을 실행하지 않습니다.

현재 기본 문서 소스는 Python, Git, LangChain, Matplotlib, NumPy, pandas, PyTorch, Hugging Face, FastAPI, BeautifulSoup, Streamlit, Gradio, scikit-learn, Pydantic입니다.

`docs` route는 query 하나당 Tavily 요청을 한 번 수행합니다. 첫 검색으로 유효한 evidence나 필요한 identifier coverage를 확보하지 못하면 query hint의 fallback을 정의된 순서대로 하나씩 실행하고, 충분한 근거를 확보하는 즉시 중단합니다. `DOCS_SEARCH_TIMEOUT_SECONDS`는 이 개별 Tavily 요청 각각에 적용되며 route 전체를 하나의 deadline으로 제한하는 값은 아닙니다.

### 3.2 업로드 파일

- 허용 확장자: `.py`, `.ipynb`
- 허용 위치: `uploads/<session_id>/...`
- 검증 기준: `src/app/web/cleanup.py::validate_upload_file_path`
- 현재 업로드 검색은 세션에 연결된 단일 파일 컨텍스트만 사용합니다.

`upload` route는 업로드 파일에서 만든 세션별 임시 Chroma retriever를 검색합니다. 파일 기반 질문에는 해당 파일을 현재 세션에 업로드해야 합니다. 검색 근거는 `tool="upload_search"`, `kind="local"`로 반환되며, `kind`는 파일 근거의 유형이고 검색 route는 `upload`입니다.

### 3.3 생성 파일과 정리 정책

- `save_text` 결과: `output/save_text/*.txt`
- 다운로드 엔드포인트: `GET /download/{filename}`
- 세션 업로드 정리: `SESSION_TTL_SECONDS`
- 생성 파일 정리: `GENERATED_FILE_TTL_SECONDS`
- 정리 로직: `src/app/web/cleanup.py::RuntimeCleaner`

## 4. 대화 메모리 정책

### 4.1 저장 형태와 compaction

세션의 process-local conversation snapshot은 두 부분으로 구성됩니다.

- `memory_summary`: 고정 예산의 rolling replacement summary
- `messages`: 최근 Human turn과 각 turn의 canonical final AI 답변

`MEMORY_HIGH_WATER_*` 중 turn, 추정 token, UTF-8 직렬화 byte, message 수 하나라도 high watermark에 도달하면 compaction을 시작합니다. 가장 오래된 완결 Human turn부터 제거해 모든 `MEMORY_LOW_WATER_*` 조건을 만족하는 가장 긴 최근 suffix를 남깁니다. `MEMORY_HARD_MAX_BYTES`는 summary와 canonical messages를 직렬화한 최종 durable snapshot의 절대 backstop입니다. token 값은 모델 과금량의 정확한 계산이 아니라 UTF-8 byte 길이를 바탕으로 한 보수적 예산 추정치입니다.

LangGraph의 `messages`는 `add_messages` reducer이므로 요약 노드는 최근 리스트만 반환하지 않습니다. `RemoveMessage(REMOVE_ALL_MESSAGES)` 뒤에 retained suffix를 다시 추가해 퇴출된 Human/AI/Tool 원문을 실제 graph state에서 제거합니다.

새 summary는 다음 입력을 하나의 bounded memory로 다시 작성한 replacement입니다.

```text
기존 bounded summary + 이번에 새로 퇴출된 대화
→ 하나의 새 bounded summary
```

기존 summary 뒤에 새 문자열을 append하지 않습니다. ToolMessage와 SystemMessage는 summary transcript에서 제외합니다. summary LLM의 예외 또는 빈 출력에는 기존 memory의 앞부분과 새로 퇴출된 최근 문맥의 뒷부분을 함께 보존하는 deterministic bounded fallback을 사용합니다.

### 4.2 요청 종료 시 durable projection과 commit

graph가 반환한 전체 메시지는 현재 요청의 debug evidence와 save/Slack receipt 조립이 끝날 때까지 유지됩니다. 조립 성공 후 세션에 저장할 때는 다음 규칙을 적용합니다.

- HumanMessage와 각 Human turn의 마지막 canonical AIMessage만 content-only 객체로 저장
- ToolMessage, SystemMessage, tool-call 중간 AI, provider/usage metadata는 저장하지 않음
- 마지막 AI 내용은 graph 내부 초안이 아니라 사용자에게 실제 표시한 receipt 포함 최종 응답으로 정규화
- projection과 모든 hard-bound 검사가 끝난 뒤 `messages + memory_summary`를 immutable snapshot 하나로 commit

graph 실행, debug 수집, response assembly, projection 또는 budget 검사가 실패하면 이전 정상 conversation snapshot을 유지합니다. 이미 완료된 `save_text`나 Slack 전송 같은 외부 side effect는 이 대화 메모리 원자성의 rollback 범위가 아닙니다.

### 4.3 수명주기와 한계

- 같은 session 요청은 기존 per-session request lock 안에서 snapshot 읽기부터 최종 commit까지 직렬화됩니다.
- `exit`/`quit`/`q`, manager close, TTL/LRU eviction은 messages와 summary를 함께 초기화합니다.
- 서로 다른 session은 snapshot을 공유하지 않습니다.
- 현재 store는 in-memory이므로 process restart와 multi-worker 사이에서 대화 상태를 복원하지 않습니다.
- Streamlit의 새 대화는 새 session ID를 발급해 즉시 격리하지만 이전 backend entry는 TTL/LRU까지 남을 수 있습니다.
- planner/synthesis에는 summary를 비신뢰 과거 데이터로 전달하며, summary 안의 명령을 따르거나 retrieved evidence로 취급하지 않습니다.

compaction 진단은 debug `edge_decisions`와 구조화 로그에서 before/after turn·message·추정 token·byte, removed message 수, fallback 여부로 확인할 수 있습니다. 원문 query, summary, Tool payload는 이 진단 로그에 기록하지 않습니다.

## 5. API 계약

### 5.1 `POST /agent`

요청 예시:

```json
{
  "query": "업로드한 파일에서 groupby가 어디에 쓰였는지 보여줘",
  "session_id": "demo-session",
  "upload_file_path": "uploads/demo-session/sales_analysis.py",
  "include_debug": true,
  "slack_user_id": "U12345678",
  "slack_email": "user@example.com",
  "slack_channel_id": "C12345678"
}
```

주요 규칙:

- `query`는 공백이 아닌 문자열이어야 하며 최대 `8192`자와 `16384` UTF-8 byte를 모두 만족해야 합니다. 초과 입력은 truncate하지 않고 graph/session 생성 전에 HTTP `422`로 거절합니다.
- `session_id`는 세션 캐시 키로 사용됩니다.
- `upload_file_path`는 반드시 `uploads/<session_id>/...` 범위 안이어야 합니다.
- `include_debug=true`일 때만 debug payload가 내려옵니다.
- Slack 필드는 세션 메타데이터로 저장되며 후속 요청에서 재사용될 수 있습니다.

응답 구조:

```json
{
  "response": {
    "answer": "문장 단위 답변 [1]",
    "claims": [],
    "evidence": [],
    "confidence": 0.42,
    "sections": []
  },
  "trace": "Session ID: ..., Request ID: ..., Agent ID: ...",
  "file_path": null,
  "debug": null
}
```

`debug`에는 아래 정보가 포함될 수 있습니다.

- `schema_version`, `observability_status`, `missing_required_debug_fields`
- `tool_calls`, `tool_call_count`
- `latency_ms_server`, `latency_breakdown`
- `token_usage`, `model_name`, `models_used`, `model_usage_status`, `llm_calls`
- `errors`, `error_codes`, `validation_events`, `edge_decisions`
- `observed_evidence`
- `retry_context`
- `retrieval_diagnostics`
- `planner_diagnostics`
- `action_results`

실제 응답 스키마 기준 파일:

- `src/app/web/schemas.py`
- `src/core/answer_schema/`

### 5.2 `POST /agent/stream`

`POST /agent`와 같은 요청 스키마를 사용하지만, 응답은 `text/event-stream` 형식의 SSE로 반환합니다. Streamlit 클라이언트는 우선 이 엔드포인트를 호출하고, 스트리밍이 실패하면 일반 `/agent` 호출로 fallback합니다.

이벤트 이름:

- `request_started`
- `stage_started`
- `stage_completed`
- `heartbeat`
- `progress_snapshot`
- `final_response`
- `error`
- `done`

`final_response` 이벤트의 `data`는 일반 `POST /agent` 응답과 같은 `response`, `trace`, `file_path`, `debug` 구조를 담습니다.

### 5.3 `GET /download/{filename}`

- `save_text`가 만든 텍스트 파일을 다운로드합니다.
- 경로 순회와 절대 경로는 차단됩니다.
- 파일이 없으면 `404 Not Found`를 반환합니다.

## 6. 프로젝트 구조

```text
.
|-- archive/
|-- data/
|   `-- benchmarks/
|-- docs/
|   |-- assets/
|   |   |-- benchmark_history.svg
|   |   |-- demo-final.png
|   |   `-- demo-flow.gif
|   |-- benchmarking.md
|   |-- design_rationale.md
|   |-- error_codes.md
|   `-- runtime_reference.md
|-- output/
|   |-- benchmarks/
|   |-- runtime/
|   `-- save_text/
|-- script/
|   |-- check_encoding.py
|   `-- sync_env_example.py
|-- src/
|   |-- app/
|   |-- core/
|   |-- eval/
|   |-- infra/
|   `-- runtime/
|-- tests/
|   |-- core/
|   |-- eval/
|   |-- tools/
|   `-- web/
`-- uploads/
```

## 7. 운영 메모

- `src.app.service_manager`는 FastAPI와 Streamlit을 함께 띄우고 종료합니다.
- `src.app.web.session_store`는 세션별 단일 요청 직렬화 lock을 사용합니다.
- planner 구조화 요청은 OpenAI client의 요청별 30초 timeout과 최대 2회 SDK 재시도를 사용합니다. 이 값은 stage 전체 deadline이 아니므로 재시도와 SDK backoff를 포함한 총 실행 시간은 30초를 넘을 수 있습니다.
- synthesis 구조화 요청은 요청별 `SYNTHESIS_TIMEOUT_SECONDS`와 `SYNTHESIS_MAX_RETRIES`를 사용합니다. 재시도를 허용하면 primary synthesis의 총 실행 시간은 설정된 요청별 timeout을 넘을 수 있으며, compact fallback은 절반의 token·timeout budget과 SDK 재시도 0회를 사용합니다.
- startup의 `fastapi_runtime_settings` 로그에는 모델, Docs/Synthesis timeout, Synthesis SDK retry, memory high/low/hard policy가 포함됩니다.
- agent request 로그는 query 원문 대신 문자 수, UTF-8 byte 수, SHA-256 hash만 기록합니다.
- benchmark run 산출물과 `latest_release_run.txt`, `latest_smoke_run.txt` 포인터는 Git으로 추적하지 않는 `output/benchmarks/` 아래에 로컬로 유지합니다.
- run별 자동 판정과 상세 분석은 각각 `output/benchmarks/<run_id>/summary.json`, `output/benchmarks/<run_id>/report.md`에서 확인합니다.
- 저장소에 공개하는 release 요약의 정본은 [README의 검증 결과](../README.md#검증-결과)이며, 비교 추세는 [benchmark history SVG](assets/benchmark_history.svg)에 유지합니다.

## 8. 테스트 및 검증

기본 검증:

```bash
uv run pytest -q
uv run python script/check_encoding.py
uv run python script/sync_env_example.py --check
```

최신 회귀 테스트 결과는 [README의 검증 결과](../README.md#검증-결과)를 기준으로 합니다. 파일 검색은 업로드 유무, 일반 파일 API 설명과의 구분, 인용과 세션 격리, 과거 벤치마크 읽기 호환성을 검증합니다. bounded memory에는 compiled reducer 회귀, 반복 rolling summary, LLM 예외/빈 출력 fallback, cross-request persistence, response assembly rollback, message ownership 격리, ToolMessage projection, JSON escape-heavy byte fitting, policy envelope, TTL/세션 격리, query boundary, Hypothesis Unicode/property, 300-turn plateau 테스트가 포함됩니다.

벤치마크 관련 명령과 로컬 산출물 정책은 [벤치마크 가이드](benchmarking.md), 공개 release 요약은 [README의 검증 결과](../README.md#검증-결과), 비교 추세는 [benchmark history SVG](assets/benchmark_history.svg)를 참고하세요. benchmark CLI의 env override 우선순위는 `CLI > .env > OS env > config.toml`입니다.
