# DocuMate
> 공식 문서 검색(Tavily) + 로컬 예제 검색(RAG) + 멀티턴 대화를 결합한 학습 보조 AI 챗봇

이 저장소는 2025년 부트캠프 팀 결과물을 기반으로, 런타임 경로와 레거시 자료를 분리해 개인 프로젝트 관점으로 유지보수 중인 버전입니다.

## 1. 주요 기능

| 기능 | 설명 |
|---|---|
| 멀티턴 대화 | 세션별 메시지 히스토리와 요약 메모리를 활용해 대화 맥락 유지 |
| 공식 문서 검색 | Tavily 검색으로 최신 공식 문서 중심 응답 생성 |
| 로컬/업로드 파일 검색 | 노트북(`.ipynb`) 또는 업로드 파일(`.py`, `.ipynb`) 기반 답변 |
| 답변 저장 | `save_text` 도구로 응답을 `.txt` 파일로 저장하고 다운로드 제공 |
| Slack 연동 | 사용자가 요청하면 `slack_notify` 도구로 DM/채널 전송 |

## 2. 런타임 아키텍처 레이어

- Interface
  - `src/main.py` (CLI + 웹 서비스 start/stop)
  - `src/web/main.py` (FastAPI)
  - `src/web/streamlit_app.py` (Streamlit UI)
- Orchestration
  - `src/agent_manager.py`
  - `src/graph_builder.py`
  - `src/make_graph.py`
  - `src/node.py`
- Infra / Tooling
  - `src/tools.py`
  - `src/upload_helpers.py`
  - `src/slack_utils.py`
  - `src/settings.py`

## 3. 프로젝트 구조

```text
.
├── archive
│   ├── legacy_code
│   │   ├── baseline_code.py
│   │   └── router_experiment.py
│   ├── team_docs
│   │   └── Langchain_Project_Team_3.pdf
│   └── README.md
├── script
│   ├── start_services.sh
│   └── stop_services.sh
├── src
│   ├── main.py
│   ├── agent_manager.py
│   ├── graph_builder.py
│   ├── make_graph.py
│   ├── llm.py
│   ├── node.py
│   ├── prompts.py
│   ├── rag_build.py
│   ├── settings.py
│   ├── slack_utils.py
│   ├── tools.py
│   ├── upload_helpers.py
│   ├── util
│   │   └── util.py
│   └── web
│       ├── main.py
│       ├── schemas.py
│       └── streamlit_app.py
├── uploads
├── pyproject.toml
└── README.md
```

레거시 코드/문서는 [archive/README.md](archive/README.md)에 정리했습니다.

## 4. 설치 및 실행

### 4.1 의존성 설치

```bash
uv sync
```

### 4.2 CLI 실행

```bash
uv run python -m src.main
```

### 4.3 Web 실행 (단일 명령)

```bash
uv run python -m src.main --mode startweb
uv run python -m src.main --mode stopweb
```

### 4.4 Web 실행 (프로세스 분리)

```bash
uv run uvicorn src.web.main:app --reload --host 0.0.0.0 --port 8000
uv run streamlit run src/web/streamlit_app.py --server.port 8501
```

웹 UI: `http://localhost:8501`

### 4.5 파일 수명주기 정책

- `uploads/<session_id>/` 업로드 파일은 세션 기준 TTL(`SESSION_TTL_SECONDS`)을 따릅니다.
- `output/save_text/*.txt` 생성 파일은 `GENERATED_FILE_TTL_SECONDS`(기본 86400초, 24시간) 이후 자동 삭제됩니다.
- 파일 정리는 FastAPI 시작 시 1회 + `/agent` 요청 시 주기적으로 수행됩니다.
- 만료되어 삭제된 파일은 다운로드 시 `404 Not Found`가 반환될 수 있습니다.

## 5. 환경변수 설정

`.env.example`을 복사해서 `.env`를 생성하세요.

```bash
cp .env.example .env
# Windows (PowerShell): Copy-Item .env.example .env
```

필수 값
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

선택 값
- `CHAT_MODEL` (default: `gpt-5.2`)
- `SUMMARY_MODEL` (default: `gpt-5-mini`)
- `VERBOSE` (default: `true`)
- `FASTAPI_URL` (default: `http://localhost:8000`)
- `SESSION_TTL_SECONDS` (default: `1800`)
- `MAX_ACTIVE_SESSIONS` (default: `200`)
- `SESSION_CLEANUP_INTERVAL_SECONDS` (default: `60`)
- `GENERATED_FILE_TTL_SECONDS` (default: `86400`)
- `FILE_CLEANUP_INTERVAL_SECONDS` (default: `60`)
- `SLACK_BOT_TOKEN`
- `SLACK_DEFAULT_USER_ID`
- `SLACK_DEFAULT_DM_EMAIL`

## 6. 참고 링크

- LangChain: https://docs.langchain.com/oss/python/langchain/overview
- Streamlit: https://docs.streamlit.io/
- FastAPI: https://fastapi.tiangolo.com/ko/
