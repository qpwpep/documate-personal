# ErrorCode taxonomy

이 문서는 `src/core/contracts/debug.py`의 `ErrorCode` Literal을 기준으로, debug payload와 action result에 기록되는 오류 코드를 정리합니다. "재시도 가능"은 같은 요청을 다시 보내는 것만으로 회복될 가능성입니다. 설정, 인증, 파일 상태가 원인인 코드는 원인을 고친 뒤 재시도해야 합니다.

| ErrorCode | 언제 발생하나 | 사용자가 할 일 | 재시도 가능 |
|---|---|---|---|
| `PLANNER_SCHEMA_INVALID` | planner의 structured output이 schema 검증 또는 파싱에 실패했습니다. 시스템은 heuristic planner fallback으로 이어갈 수 있습니다. | 질문 의도, 필요한 문서 범위, 업로드 파일 사용 여부를 더 명확히 적습니다. 반복된다면 planner prompt/schema 변경 여부를 확인합니다. | 예. fallback이 적용될 수 있고, 재요청으로 정상 output이 나올 수 있습니다. |
| `PLANNER_TIMEOUT` | planner LLM 호출이 timeout 또는 timed out 오류로 종료됐습니다. | 질문을 줄이거나 다시 요청합니다. 운영자는 모델 지연, 네트워크, planner timeout 설정을 확인합니다. | 예. 일시적 지연이면 재시도 가능성이 높습니다. |
| `RETRIEVAL_DOCS_TIMEOUT` | 공식 문서 검색 route의 Tavily 호출이 `DOCS_SEARCH_TIMEOUT_SECONDS` 안에 끝나지 않았습니다. | 질문의 라이브러리/버전/기능명을 좁히고 다시 요청합니다. 운영자는 Tavily 상태와 timeout 설정을 확인합니다. | 예. 외부 검색 지연이면 재시도로 회복될 수 있습니다. |
| `RETRIEVAL_DOCS_FAILED` | Tavily 호출 실패, 예외, 예상과 다른 응답 타입, `results` payload 누락 등 공식 문서 검색이 실패했습니다. | 공식 문서 검색이 꼭 필요하면 다시 요청합니다. 운영자는 `TAVILY_API_KEY`, 네트워크, allowlist/domain rule을 확인합니다. | 조건부. 외부/API 문제면 원인 해소 후 재시도합니다. |
| `RAG_INDEX_MISSING` | local notebook RAG index 경로가 없어 `rag_search`를 사용할 수 없습니다. | 로컬 문서 기반 답변이 필요하면 index를 먼저 빌드합니다. | 아니요. index 생성 전에는 같은 요청을 반복해도 해결되지 않습니다. |
| `LOCAL_RAG_FAILED` | local 또는 upload retriever의 similarity search가 예외로 실패했습니다. | 로컬 index를 재빌드하거나 업로드 파일을 다시 올립니다. 운영자는 embedding/API key와 Chroma 상태를 확인합니다. | 조건부. index, 파일, API 설정을 고친 뒤 재시도합니다. |
| `UPLOAD_RETRIEVER_BUILD_FAILED` | 업로드된 `.py` 또는 `.ipynb` 파일로 임시 retriever를 만드는 단계가 실패했습니다. | 파일이 손상됐거나 너무 크지 않은지 확인하고 다시 업로드합니다. 운영자는 `OPENAI_API_KEY`와 업로드 파서/embedding 오류를 확인합니다. | 조건부. 파일 또는 설정을 고친 뒤 재시도합니다. |
| `LLM_STRUCTURED_EMPTY` | synthesis 단계의 structured output이 비어 있었습니다. | 질문을 더 작게 나누거나 다시 요청합니다. 운영자는 모델 응답/structured output adapter 로그를 확인합니다. | 예. 일시적 LLM 출력 실패일 수 있습니다. |
| `SYNTHESIS_TIMEOUT` | 최종 답변 생성 단계가 timeout 또는 timed out 오류로 종료됐습니다. | 질문 범위를 줄이거나 업로드/근거 요구를 좁혀 다시 요청합니다. 운영자는 `SYNTHESIS_TIMEOUT_SECONDS`와 모델 지연을 확인합니다. | 예. 다만 큰 context가 원인이면 요청을 줄인 뒤 재시도합니다. |
| `VALIDATION_UNSUPPORTED_CLAIMS` | validation이 근거로 뒷받침되지 않는 claim을 발견했고 retry 사유가 `unsupported_claims`로 기록됐습니다. | 더 구체적인 근거 문서나 업로드 파일을 제공하거나, 답변 범위를 좁혀 요청합니다. | 조건부. 시스템 retry가 한 번 적용될 수 있고, 추가 근거가 있으면 재시도 가치가 있습니다. |
| `HYBRID_SECTION_REPEATED` | hybrid 답변에서 공식 문서/업로드 파일 비교 섹션이 반복되거나 섹션 구성이 약하다고 validation이 판단했습니다. | 비교 질문이라면 공식 문서 기준과 업로드 파일 기준을 나누어 달라고 명시합니다. 운영자는 hybrid section assessment와 synthesis prompt를 확인합니다. | 조건부. 같은 evidence로 재합성하면 회복될 수 있습니다. |
| `HYBRID_UPLOAD_SETTING_MISSING` | hybrid 답변이 업로드 파일의 설정/구현 내용을 충분히 반영하지 못했습니다. | 업로드 파일을 다시 확인하고, 비교할 설정명이나 코드 위치를 더 구체적으로 적습니다. 운영자는 upload evidence 추출과 route coverage를 확인합니다. | 조건부. 업로드 evidence가 충분하면 재요청 또는 재합성으로 회복될 수 있습니다. |
| `HYBRID_COMPARISON_WEAK` | 공식 문서 근거와 업로드/로컬 근거를 함께 사용해야 하는 질문에서 비교 연결이 약하다고 validation이 판단했습니다. | "공식 문서 기준"과 "내 코드/업로드 파일 기준"을 함께 비교해 달라고 요청 범위를 좁힙니다. 운영자는 hybrid comparison assessment와 `error_code_histogram`을 확인합니다. | 조건부. 추가 근거나 더 구체적인 비교 축이 있으면 재시도 가치가 있습니다. |
| `DEBUG_NORMALIZATION_FAILED` | web API가 raw debug payload를 `AgentDebugInfo`로 정규화하는 중 latency/debug 구조 검증에 실패했습니다. | 답변 자체보다 관측성 정보가 불완전한 상태입니다. 운영자는 raw debug payload와 `schema_version`을 확인합니다. | 아니요. 같은 사용자 요청 반복보다 debug schema/normalizer 수정이 필요합니다. |
| `SLACK_AUTH_FAILED` | Slack token이 없거나 Slack API 호출이 인증/권한 문제로 실패했습니다. | `SLACK_BOT_TOKEN` 설정, 앱 설치, channel 접근 권한, 필요한 scope를 확인합니다. | 조건부. Slack 설정을 고친 뒤 재시도합니다. |
| `SLACK_DESTINATION_MISSING` | channel ID, user ID, email, 기본 Slack destination 중 어느 것도 유효하게 해석되지 않았습니다. | Slack 전송을 원하면 channel ID(`C/G/D...`), user ID, email 중 하나를 제공하거나 기본 destination env를 설정합니다. | 조건부. destination을 제공한 뒤 재시도합니다. |
| `UPLOAD_PATH_INVALID` | 요청의 `upload_file_path`가 비어 있지 않지만 session upload directory 밖이거나, `.py`/`.ipynb`가 아니거나, 파일이 없습니다. | 현재 세션에서 파일을 다시 업로드하고 지원 확장자만 사용합니다. 클라이언트가 임의 경로나 이전 세션 경로를 보내지 않는지 확인합니다. | 아니요. 올바른 업로드 경로로 다시 요청해야 합니다. |

## 운영 메모

- ErrorCode의 source of truth는 `src/core/contracts/debug.py`입니다.
- retrieval/action tool은 가능한 경우 payload의 `error_code`에 직접 기록합니다.
- planner/synthesis 계열 코드는 stage error 문자열을 정규화해서 debug payload의 `error_codes`에 합쳐집니다.
- benchmark histogram은 `src/eval/reporting/histograms.py`에서 같은 코드 집합을 bucket으로 집계합니다.
