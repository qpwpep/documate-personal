# DocuMate
> 공식 문서 검색, 로컬 노트북 RAG, 업로드 파일 검색, 멀티턴 세션 메모리를 결합한 LangGraph 기반 학습 보조 에이전트

이 저장소는 2025년 부트캠프 팀 결과물을 기반으로 현재 런타임 경로와 평가 체계를 분리해 유지보수 중인 개인 프로젝트 버전입니다. 현재 버전은 구조화된 evidence 응답, 검증 후 1회 재시도, FastAPI/Streamlit 실행 관리, 온라인 벤치마크 리포팅을 중심으로 정리되어 있습니다.

## 1. 핵심 기능

| 기능 | 설명 |
|---|---|
| 멀티턴 세션 메모리 | 세션별 메시지 히스토리와 요약 메모리를 유지하고, FastAPI 레이어에서 TTL + LRU 캐시로 세션을 관리합니다. |
| 공식 문서 검색 | `tavily_search`가 `src/domain_docs.py`에 정의된 공식 문서 도메인 집합을 기본 화이트리스트로 사용합니다. |
| 로컬 노트북 RAG | `src/rag_build.py`가 `data/`와 `uploads/` 아래 `.ipynb`를 증분 인덱싱하고, `rag_search`가 `data/index`를 조회합니다. |
| 업로드 파일 검색 | 현재 세션의 업로드 파일 `.py`, `.ipynb`에 대해 임시 Chroma retriever를 구성하고 `upload_search`로 조회합니다. |
| 구조화된 evidence 응답 | `/agent`는 `response.answer`와 `response.evidence[]`를 반환하고, `debug.observed_evidence`와 함께 평가에 사용됩니다. |
| evidence 검증 후 재시도 | evidence가 없거나 점수가 낮으면 planner -> retrieval -> synthesis 흐름을 최대 1회 재시도하고, 실패 시 불확실성 응답을 반환합니다. |
| 후처리 도구 | 사용자가 요청하면 `save_text`로 답변을 `.txt` 파일로 저장하고 `slack_notify`로 Slack DM 또는 채널 전송을 수행합니다. |
| UTF-8 안전 실행 | `src/runtime_encoding.py`와 `src/main.py`가 UTF-8 모드 재실행과 표준 입출력 재설정을 처리합니다. |

## 2. 런타임 아키텍처

- Interface
  - `src/main.py`: CLI 실행과 FastAPI/Streamlit 백그라운드 서비스 시작, 중지
  - `src/web/main.py`: `/agent`, `/download/{filename}`를 제공하는 FastAPI 앱
  - `src/web/streamlit_app.py`: 웹 UI
- Orchestration
  - `src/agent_manager.py`: 세션별 LangGraph 실행 결과를 정리하고 evidence, debug 정보를 추출
  - `src/graph_builder.py`: LLM, 도구, 노드 조합
  - `src/node.py`: planner, retrieval dispatch, synthesis, validate_evidence, action_postprocess 구현
  - `src/planner_schema.py`: planner 출력 스키마와 route 제약
- Evidence / Retrieval
  - `src/evidence.py`: evidence 정규화, dedupe, payload 파싱
  - `src/domain_docs.py`: 공식 문서 도메인 기본 목록
  - `src/rag_build.py`: 로컬 노트북 인덱스 생성
  - `src/upload_helpers.py`: 업로드 파일용 임시 retriever 생성
- Runtime flow
  - 사용자 메시지 추가 -> 오래된 대화 요약 -> planner route 선택 -> docs/upload/local retrieval -> synthesis -> evidence 검증 -> 필요 시 1회 재시도 -> save/slack 후처리

## 3. 프로젝트 구조

```text
.
├── archive/
│   ├── legacy_code/
│   ├── team_docs/
│   └── README.md
├── data/
│   ├── benchmarks/
│   │   ├── config.toml
│   │   └── fixtures/
│   └── index/                  # src.rag_build.py가 생성하는 Chroma 인덱스
├── docs/
│   └── assets/
│       └── benchmark_history.svg
├── output/
│   ├── benchmarks/
│   └── save_text/
├── script/
│   ├── check_encoding.py
│   └── web_services_state.json
├── src/
│   ├── agent_manager.py
│   ├── domain_docs.py
│   ├── evidence.py
│   ├── graph_builder.py
│   ├── main.py
│   ├── node.py
│   ├── planner_schema.py
│   ├── rag_build.py
│   ├── runtime_encoding.py
│   ├── settings.py
│   ├── tools.py
│   ├── upload_helpers.py
│   ├── eval/
│   └── web/
├── tests/
│   ├── eval/
│   └── web/
├── uploads/
├── pyproject.toml
└── README.md
```

레거시 코드와 이전 팀 산출물은 [archive/README.md](archive/README.md)에 분리해 두었습니다.

## 4. 설치 및 실행

### 4.1 의존성 설치

```bash
uv sync
```

### 4.2 환경변수 파일 준비

```bash
cp .env.example .env
# Windows PowerShell
Copy-Item .env.example .env
```

필수 값
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

### 4.3 로컬 노트북 인덱스 생성

로컬 RAG를 사용하려면 먼저 `data/index`를 생성해야 합니다.

```bash
uv run python -m src.rag_build
```

이 명령은 `data/`와 `uploads/` 아래 `.ipynb` 파일을 증분 인덱싱합니다.

### 4.4 CLI 실행

```bash
uv run python -m src.main
```

`src.main`은 현재 인터프리터가 UTF-8 모드가 아니면 내부적으로 `-X utf8`로 재실행합니다.

### 4.5 웹 서비스 실행

```bash
uv run python -m src.main --mode startweb
uv run python -m src.main --mode stopweb
```

- FastAPI: `http://localhost:8000`
- Streamlit: `http://localhost:8501`
- 시작 시 프로세스 상태는 `script/web_services_state.json`에 기록됩니다.

### 4.6 FastAPI/Streamlit 직접 실행

```bash
uv run python -X utf8 -m uvicorn src.web.main:app --host 0.0.0.0 --port 8000
uv run python -X utf8 -m streamlit run src/web/streamlit_app.py --server.port 8501
```

직접 실행 시에는 `-X utf8` 또는 `PYTHONUTF8=1` 설정을 유지하는 편이 안전합니다.

## 5. 환경변수

| 이름 | 기본값 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | 없음 | OpenAI 호출에 필요 |
| `TAVILY_API_KEY` | 없음 | Tavily 검색에 필요 |
| `CHAT_MODEL` | `gpt-5-mini` | synthesis 모델 |
| `PLANNER_MODEL` | `gpt-5-nano` | planner 모델 |
| `SUMMARY_MODEL` | `gpt-5-mini` | 요약 모델 |
| `VERBOSE` | `true` | CLI 및 내부 로깅 상세도 |
| `FASTAPI_URL` | `http://localhost:8000` | Streamlit에서 사용할 API 주소 |
| `SESSION_TTL_SECONDS` | `1800` | 세션 TTL |
| `MAX_ACTIVE_SESSIONS` | `200` | 세션 캐시 최대 개수 |
| `SESSION_CLEANUP_INTERVAL_SECONDS` | `60` | 세션 cleanup 주기 |
| `GENERATED_FILE_TTL_SECONDS` | `86400` | `save_text` 생성 파일 TTL |
| `FILE_CLEANUP_INTERVAL_SECONDS` | `60` | 파일 cleanup 주기 |
| `SLACK_BOT_TOKEN` | 없음 | Slack 전송용 토큰 |
| `SLACK_DEFAULT_USER_ID` | 없음 | 기본 DM 대상 |
| `SLACK_DEFAULT_DM_EMAIL` | 없음 | 기본 DM 이메일 |
| `JUDGE_MODEL` | `gpt-5-mini` | 벤치마크 judge 모델 |
| `BENCHMARK_ENDPOINT` | `http://localhost:8000` | 벤치마크 기본 대상 |
| `BENCHMARK_JUDGE_ENABLED` | `true` | judge 사용 여부 |

## 6. 업로드, 저장, 세션 정책

- 업로드 파일은 `uploads/<session_id>/` 하위 경로만 허용됩니다.
- 허용 확장자는 `.py`, `.ipynb`입니다.
- `upload_search`는 현재 세션에 연결된 업로드 파일만 조회합니다.
- `save_text`가 생성한 파일은 `output/save_text/*.txt`에 저장됩니다.
- 업로드 디렉터리는 `SESSION_TTL_SECONDS`, 생성 파일은 `GENERATED_FILE_TTL_SECONDS` 기준으로 자동 정리됩니다.
- 만료된 저장 파일은 `/download/{filename}` 요청 시 `404 Not Found`가 반환될 수 있습니다.

## 7. API 계약

### 7.1 `POST /agent`

요청 예시:

```json
{
  "query": "NumPy broadcasting을 간단히 설명해줘",
  "session_id": "demo-session",
  "upload_file_path": "uploads/demo-session/sample_pipeline.ipynb",
  "include_debug": true,
  "slack_user_id": "U12345678",
  "slack_email": "user@example.com",
  "slack_channel_id": "C12345678"
}
```

응답 예시:

```json
{
  "response": {
    "answer": "NumPy broadcasting은 서로 다른 shape의 배열 연산을 가능하게 하는 규칙입니다.",
    "evidence": [
      {
        "kind": "official",
        "tool": "tavily_search",
        "source_id": "url:https://numpy.org/doc/stable/user/basics.broadcasting.html",
        "url_or_path": "https://numpy.org/doc/stable/user/basics.broadcasting.html",
        "title": "Broadcasting",
        "snippet": "Broadcasting provides a means of vectorizing array operations...",
        "score": 0.98
      }
    ]
  },
  "trace": "Session ID: demo-session, Request ID: abcd1234, Agent ID: 12345678",
  "file_path": "output/save_text/response_20260306_103000.txt",
  "debug": {
    "tool_calls": ["tavily_search", "save_text"],
    "tool_call_count": 2,
    "latency_ms_server": 1842,
    "token_usage": {
      "prompt_tokens": 642,
      "completion_tokens": 153,
      "total_tokens": 795
    },
    "model_name": "gpt-5-mini",
    "errors": [],
    "observed_evidence": [
      {
        "kind": "official",
        "tool": "tavily_search",
        "source_id": "url:https://numpy.org/doc/stable/user/basics.broadcasting.html",
        "url_or_path": "https://numpy.org/doc/stable/user/basics.broadcasting.html",
        "title": "Broadcasting",
        "snippet": "Broadcasting provides a means of vectorizing array operations...",
        "score": 0.98
      }
    ],
    "retry_context": {
      "attempt": 0,
      "max_retries": 1,
      "retry_reason": null,
      "retrieval_feedback": null,
      "evidence_start_index": 0,
      "retrieval_error_start_index": 0,
      "score_avg": null
    }
  }
}
```

### 7.2 `GET /download/{filename}`

- `save_text` 결과 파일 다운로드용 엔드포인트입니다.
- 절대 경로나 상위 디렉터리 탈출 경로는 거부됩니다.

## 8. 온라인 벤치마크

DocuMate는 `FastAPI /agent` 실경로를 기준으로 하는 온라인 벤치마크를 제공합니다.

### 8.1 케이스 생성

```bash
uv run python -m src.eval.main generate \
  --seed data/benchmarks/fixtures/cases.seed.jsonl \
  --regression-seed data/benchmarks/fixtures/cases.regression.seed.jsonl \
  --out data/benchmarks/fixtures/cases.generated.jsonl \
  --target 120
```

### 8.2 실행

```bash
uv run python -m src.eval.main run \
  --mode online \
  --fixtures data/benchmarks/fixtures/cases.generated.jsonl \
  --endpoint http://localhost:8000
```

### 8.3 리포트 재생성

```bash
uv run python -m src.eval.main report --run output/benchmarks/20260306_094033
```

### 8.4 결과 산출물

- `output/benchmarks/<run_id>/raw_results.jsonl`
- `output/benchmarks/<run_id>/summary.json`
- `output/benchmarks/<run_id>/report.md`
- `output/benchmarks/latest_run.txt`

### 8.5 Hard Gate

| Gate | Threshold |
|---|---:|
| `pass_rate` | 0.82 |
| `tool_precision` | 0.90 |
| `tool_recall` | 0.85 |
| `citation_compliance` | 0.88 |
| `p95_latency_ms` | 20000 |
| `avg_cost_per_case_usd` | 0.035 |

## 9. 최신 벤치마크 결과

기준 런은 `output/benchmarks/latest_run.txt`가 가리키는 `20260306_123220`입니다.

- run_id: `20260306_123220`
- generated_at_utc: `2026-03-06T13:25:46.653810+00:00`
- endpoint: `http://localhost:8000`
- fixtures: `data\benchmarks\fixtures\cases.generated.jsonl`
- overall: `FAIL`

### 9.1 Metrics

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

### 9.2 Hard Gates

| Gate | Threshold | Actual | Passed |
|---|---:|---:|:---:|
| pass_rate | 0.82 | 0.25 | N |
| tool_precision | 0.90 | 0.8824 | N |
| tool_recall | 0.85 | 0.20 | N |
| citation_compliance | 0.88 | 0.00 | N |
| p95_latency_ms | 20000 | 41446.1 | N |
| avg_cost_per_case_usd | 0.035 | 0.00058941 | Y |

최신 실패는 대부분 `score below threshold`입니다. 상세 목록은 [latest report](output/benchmarks/20260306_123220/report.md)를 참고하세요.

## 10. 최근 벤치마크 이력 및 추세

저장소에 남아 있는 3개 런 기준으로 보면, baseline 이후 정답률과 citation 품질은 하락한 상태로 유지되고 있다. 반면 `20260306_123220` 런은 직전 `20260306_094033` 대비 p95 latency를 크게 줄였지만, retrieval 품질 지표는 아직 회복되지 않았다.

| run_id | generated_at_utc | overall | pass_rate | tool_precision | tool_recall | citation_compliance | p50_latency_ms | p95_latency_ms | avg_cost_per_case_usd | 변화 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `20260303_134325` | `2026-03-03T15:24:50.806715+00:00` | `FAIL` | 0.3833 | 0.8049 | 0.4400 | 0.3056 | 49835.5 | 62063.0 | 0.00219042 | 기준 런 |
| `20260306_094033` | `2026-03-06T10:34:56.024712+00:00` | `FAIL` | 0.2500 | 0.8824 | 0.2000 | 0.0000 | 14969.0 | 53888.1 | 0.00058583 | `pass_rate -0.1333; tool_precision +0.0775; tool_recall -0.2400; citation_compliance -0.3056; p50_latency_ms -34866.5; p95_latency_ms -8174.9; avg_cost_per_case_usd -0.00160459` |
| `20260306_123220` | `2026-03-06T13:25:46.653810+00:00` | `FAIL` | 0.2500 | 0.8824 | 0.2000 | 0.0000 | 15905.5 | 41446.1 | 0.00058941 | `pass_rate +0.0000; tool_precision +0.0000; tool_recall +0.0000; citation_compliance +0.0000; p50_latency_ms +936.5; p95_latency_ms -12442.0; avg_cost_per_case_usd +0.00000358` |

![DocuMate benchmark history](docs/assets/benchmark_history.svg)

저장소에 남아 있는 3개 런 기준 trend chart입니다. 상세 수치는 [run 20260303_134325](output/benchmarks/20260303_134325/report.md), [run 20260306_094033](output/benchmarks/20260306_094033/report.md), [run 20260306_123220](output/benchmarks/20260306_123220/report.md)에서 다시 확인할 수 있습니다.

## 11. 테스트 및 검증

```bash
uv run python -m unittest discover -s tests
uv run python -m src.eval.main report --run output/benchmarks/20260306_094033
uv run python script/check_encoding.py
```

현재 테스트는 planner graph, evidence pipeline, API 스키마, 벤치마크 fixture 계약, scoring 규칙, request payload 전달을 다룹니다.

## 12. 인코딩 정책

- 텍스트 파일 기본 인코딩은 UTF-8 no BOM입니다.
- `.editorconfig`는 `charset = utf-8`을 기본값으로 사용합니다.
- Windows PowerShell 5.1에서는 콘솔 출력 인코딩 때문에 한글이 깨져 보일 수 있습니다.
- 직접 실행 시 UTF-8 모드를 강제하려면 `-X utf8` 또는 `PYTHONUTF8=1`을 사용하세요.

런타임 인코딩 점검 예시:

```bash
uv run python -X utf8 -c "import sys, locale; print(sys.flags.utf8_mode, sys.stdout.encoding, locale.getpreferredencoding(False))"
```

## 13. 참고 링크

- LangChain: https://docs.langchain.com/oss/python/langchain/overview
- Streamlit: https://docs.streamlit.io/
- FastAPI: https://fastapi.tiangolo.com/
- Pydantic: https://docs.pydantic.dev/latest/
