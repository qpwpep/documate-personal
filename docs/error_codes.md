# ErrorCode taxonomy

이 문서는 `src/core/contracts/debug.py`의 `ErrorCode` Literal을 기준으로, debug payload와 action result에 기록되는 오류 코드를 정리합니다. "재시도 가능"은 같은 요청을 다시 보내는 것만으로 회복될 가능성입니다. 설정, 인증, 파일 상태가 원인인 코드는 원인을 고친 뒤 재시도해야 합니다.

| ErrorCode | 언제 발생하나 | 사용자가 할 일 | 재시도 가능 |
|---|---|---|---|
| `PLANNER_SCHEMA_INVALID` | planner의 structured output이 schema 검증 또는 파싱에 실패했습니다. `planner_unavailable`로 기록하고 검색·저장·전송 없이 재요청을 안내합니다. | 질문 의도, 필요한 문서 범위, 업로드 파일 사용 여부를 더 명확히 적습니다. 반복된다면 planner prompt/schema 변경 여부를 확인합니다. | 예. 재요청으로 정상 output이 나올 수 있습니다. |
| `PLANNER_TIMEOUT` | planner LLM 호출이 timeout 또는 timed out 오류로 종료됐습니다. | 질문을 줄이거나 다시 요청합니다. 운영자는 모델 지연, 네트워크, planner timeout 설정을 확인합니다. | 예. 일시적 지연이면 재시도 가능성이 높습니다. |
| `RETRIEVAL_DOCS_TIMEOUT` | 공식 문서 검색 route의 Tavily 호출이 `DOCS_SEARCH_TIMEOUT_SECONDS` 안에 끝나지 않았습니다. | 질문의 라이브러리/버전/기능명을 좁히고 다시 요청합니다. 운영자는 Tavily 상태와 timeout 설정을 확인합니다. | 예. 외부 검색 지연이면 재시도로 회복될 수 있습니다. |
| `RETRIEVAL_DOCS_FAILED` | Tavily 호출 실패, 예외, 예상과 다른 응답 타입, `results` payload 누락 등 공식 문서 검색이 실패했습니다. | 공식 문서 검색이 꼭 필요하면 다시 요청합니다. 운영자는 `TAVILY_API_KEY`, 네트워크, allowlist/domain rule을 확인합니다. | 조건부. 외부/API 문제면 원인 해소 후 재시도합니다. |
| `RAG_INDEX_MISSING` | 과거 local route의 인덱스 누락 기록을 읽기 위해 유지하는 코드이며 현재 런타임에서는 발생하지 않습니다. | 현재 파일 기반 질문은 해당 파일을 세션에 업로드해 요청합니다. | 아니요. 과거 실행 기록을 해석하는 코드입니다. |
| `LOCAL_RAG_FAILED` | upload retriever의 similarity search가 예외로 실패했습니다. 과거 결과에서는 local route 실패에도 사용됩니다. | 업로드 파일을 다시 올립니다. 운영자는 embedding/API key와 Chroma 상태를 확인합니다. | 조건부. 파일이나 API 설정을 고친 뒤 재시도합니다. |
| `UPLOAD_RETRIEVER_BUILD_FAILED` | 업로드된 `.py` 또는 `.ipynb` 파일로 임시 retriever를 만드는 단계가 실패했습니다. | 파일이 손상됐거나 너무 크지 않은지 확인하고 다시 업로드합니다. 운영자는 `OPENAI_API_KEY`와 업로드 파서/embedding 오류를 확인합니다. | 조건부. 파일 또는 설정을 고친 뒤 재시도합니다. |
| `LLM_STRUCTURED_EMPTY` | synthesis 단계에서 표시할 `AnswerDocument.blocks`가 비어 있었습니다. | 질문을 더 작게 나누거나 다시 요청합니다. 운영자는 모델 응답/structured output adapter 로그를 확인합니다. | 예. 일시적 LLM 출력 실패일 수 있습니다. |
| `SYNTHESIS_TIMEOUT` | 최종 답변 생성 단계가 timeout 또는 timed out 오류로 종료됐습니다. | 질문 범위를 줄이거나 업로드/근거 요구를 좁혀 다시 요청합니다. 운영자는 `SYNTHESIS_TIMEOUT_SECONDS`와 모델 지연을 확인합니다. | 예. 다만 큰 context가 원인이면 요청을 줄인 뒤 재시도합니다. |
| `VALIDATION_UNRESOLVED_REFERENCES` | 실제 표시 내용의 `refs`가 해당 synthesis packet에 없거나, 근거가 필요한 내용에 참조가 없습니다. 의미적 사실 판정은 아닙니다. | 더 구체적인 자료를 제공하거나 답변 범위를 좁힙니다. 운영자는 packet 선택과 출력 refs를 확인합니다. | 조건부. 확보한 검색 결과를 재사용해 본문과 참조를 함께 다시 생성할 수 있습니다. |
| `VALIDATION_MISSING_CONTENT` | 본문이 비었거나 요청한 코드·단계·체크리스트 형식 또는 출처 범위가 부족합니다. 원문 발췌가 실제 선택 범위와 일치하지 않는 경우도 포함합니다. | 원하는 결과를 구체적으로 적습니다. 운영자는 내용 단위의 checks, 요청 계약과 route coverage를 확인합니다. | 조건부. 재합성 후에도 부족하면 확인 가능한 원문 발췌와 제한을 제공합니다. |
| `DEBUG_NORMALIZATION_FAILED` | web API가 raw debug payload를 `AgentDebugInfo`로 정규화하는 중 latency/debug 구조 검증에 실패했습니다. | 답변 자체보다 관측성 정보가 불완전한 상태입니다. 운영자는 raw debug payload와 `schema_version`을 확인합니다. | 아니요. 같은 사용자 요청 반복보다 debug schema/normalizer 수정이 필요합니다. |
| `SLACK_AUTH_FAILED` | Slack token이 없거나 Slack API 호출이 인증/권한 문제로 실패했습니다. | `SLACK_BOT_TOKEN` 설정, 앱 설치, channel 접근 권한, 필요한 scope를 확인합니다. | 조건부. Slack 설정을 고친 뒤 재시도합니다. |
| `SLACK_DESTINATION_MISSING` | channel ID, user ID, email, 기본 Slack destination 중 어느 것도 유효하게 해석되지 않았습니다. | Slack 전송을 원하면 channel ID(`C/G/D...`), user ID, email 중 하나를 제공하거나 기본 destination env를 설정합니다. | 조건부. destination을 제공한 뒤 재시도합니다. |
| `UPLOAD_PATH_INVALID` | 요청의 `upload_file_path`가 비어 있지 않지만 session upload directory 밖이거나, `.py`/`.ipynb`가 아니거나, 파일이 없습니다. | 현재 세션에서 파일을 다시 업로드하고 지원 확장자만 사용합니다. 클라이언트가 임의 경로나 이전 세션 경로를 보내지 않는지 확인합니다. | 아니요. 올바른 업로드 경로로 다시 요청해야 합니다. |

## 운영 메모

- ErrorCode의 source of truth는 `src/core/contracts/debug.py`입니다.
- retrieval/action tool은 가능한 경우 payload의 `error_code`에 직접 기록합니다.
- planner/synthesis 계열 코드는 stage error 문자열을 정규화해서 debug payload의 `error_codes`에 합쳐집니다.
- benchmark histogram은 `src/eval/reporting/histograms.py`에서 같은 코드 집합을 bucket으로 집계합니다.
- 사용자에게 필요한 제한은 `AnswerResponse.issues`, 내용별 확인 결과는 `checks`, 저장·전송 실패는 `actions`에도 전달합니다. debug를 끈 상태에서도 이 정보는 유지됩니다.
- 참조가 `resolved`인 것과 설명이 원문으로 뒷받침되는 것은 구분합니다. 일반 설명의 `support_status`는 `not_evaluated`이며, 원문 발췌가 정확히 일치할 때만 `exact_match`입니다.
