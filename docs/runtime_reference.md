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

### 1.4 로컬 노트북 인덱스 생성

```bash
uv run python -m src.app.rag_build
```

이 명령은 `data/`와 `uploads/` 아래의 `.ipynb` 파일을 스캔해 `data/index`에 Chroma 인덱스를 만듭니다.

### 1.5 웹 서비스 실행

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
| `TAIL_HEDGE_MAX_CONCURRENCY` | `8` | tail latency hedge max concurrency |
| `TAIL_HEDGE_MAX_ATTEMPTS` | `3` | tail latency hedge max attempts per call |
| `PLANNER_HEDGE_DELAY_SECONDS` | `0.5` | planner tail latency hedge delay |
| `DOCS_SEARCH_TIMEOUT_SECONDS` | `5` | Tavily 검색 timeout |
| `DOCS_SEARCH_HEDGE_DELAY_SECONDS` | `0.5` | Tavily 검색 tail latency hedge delay |
| `SYNTHESIS_TIMEOUT_SECONDS` | `20` | synthesis timeout |
| `SYNTHESIS_HEDGE_DELAY_SECONDS` | `0.2` | synthesis tail latency hedge delay |
| `SYNTHESIS_HEDGE_MAX_ATTEMPTS` | `4` | synthesis tail latency hedge max attempts |
| `SYNTHESIS_USE_RESPONSES_API` | `false` | synthesis Responses API 사용 여부 |
| `SYNTHESIS_MAX_RETRIES` | `0` | synthesis 자체 재시도 횟수 |
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
- `src/infra/config/agent_rules.toml`: docs allowlist, intent rule, query hint 규칙

현재 기본 문서 소스는 Python, Git, LangChain, Matplotlib, NumPy, pandas, PyTorch, Hugging Face, FastAPI, BeautifulSoup, Streamlit, Gradio, scikit-learn, Pydantic입니다.

### 3.2 업로드 파일

- 허용 확장자: `.py`, `.ipynb`
- 허용 위치: `uploads/<session_id>/...`
- 검증 기준: `src/app/web/cleanup.py::validate_upload_file_path`
- 현재 업로드 검색은 세션에 연결된 단일 파일 컨텍스트만 사용합니다.

### 3.3 생성 파일과 정리 정책

- `save_text` 결과: `output/save_text/*.txt`
- 다운로드 엔드포인트: `GET /download/{filename}`
- 세션 업로드 정리: `SESSION_TTL_SECONDS`
- 생성 파일 정리: `GENERATED_FILE_TTL_SECONDS`
- 정리 로직: `src/app/web/cleanup.py::RuntimeCleaner`

## 4. API 계약

### 4.1 `POST /agent`

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

### 4.2 `POST /agent/stream`

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

### 4.3 `GET /download/{filename}`

- `save_text`가 만든 텍스트 파일을 다운로드합니다.
- 경로 순회와 절대 경로는 차단됩니다.
- 파일이 없으면 `404 Not Found`를 반환합니다.

## 5. 프로젝트 구조

```text
.
|-- archive/
|-- data/
|   `-- benchmarks/
|-- docs/
|   |-- assets/
|   |-- benchmarking.md
|   |-- benchmark_results.md
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

## 6. 운영 메모

- `src.app.service_manager`는 FastAPI와 Streamlit을 함께 띄우고 종료합니다.
- `src.app.web.session_store`는 세션별 단일 요청 직렬화 lock을 사용합니다.
- `src.infra.rag_build`는 증분 인덱싱을 위해 `data/index/manifest.json`을 관리합니다.
- benchmark 최신 성능 정본은 `output/benchmarks/latest_release_run.txt`입니다.
- smoke 최신 런 포인터는 `output/benchmarks/latest_smoke_run.txt`로 별도 관리합니다.
- 공개용 benchmark 요약은 [벤치마크 결과](benchmark_results.md)에 별도 정리합니다.

## 7. 테스트 및 검증

기본 검증:

```bash
uv run pytest -q
uv run python script/check_encoding.py
uv run python script/sync_env_example.py --check
```

벤치마크 관련 명령은 [벤치마크 가이드](benchmarking.md), 최신 결과는 [벤치마크 결과](benchmark_results.md)를 참고하세요. benchmark CLI의 env override 우선순위는 `CLI > .env > OS env > config.toml`입니다.
