# ErrorCode taxonomy

이 문서는 그래프의 debug/action 오류와 문서 첨부 API의 HTTP 오류를 구분해 정리합니다. "재시도 가능"은 같은 요청을 다시 보내는 것만으로 회복될 가능성입니다. 설정, 인증, 파일 상태가 원인인 코드는 원인을 고친 뒤 재시도해야 합니다.

## 그래프 debug·action ErrorCode

아래 코드는 [`src/core/contracts/debug.py`](../src/core/contracts/debug.py)의 `ErrorCode` Literal을 기준으로 debug payload와 action result에 기록됩니다. 뒤의 문서 첨부 HTTP 코드는 별도 계약이며 이 Literal이나 benchmark histogram의 코드 집합에 자동으로 포함되지 않습니다.

| ErrorCode | 언제 발생하나 | 사용자가 할 일 | 재시도 가능 |
|---|---|---|---|
| `PLANNER_SCHEMA_INVALID` | planner의 structured output이 schema 검증 또는 파싱에 실패했습니다. `planner_unavailable`로 기록하고 검색·저장·전송 없이 재요청을 안내합니다. | 질문 의도, 필요한 문서 범위, 업로드 파일 사용 여부를 더 명확히 적습니다. 반복된다면 planner prompt/schema 변경 여부를 확인합니다. | 예. 재요청으로 정상 output이 나올 수 있습니다. |
| `PLANNER_TIMEOUT` | planner LLM 호출이 timeout 또는 timed out 오류로 종료됐습니다. | 질문을 줄이거나 다시 요청합니다. 운영자는 모델 지연, 네트워크, planner timeout 설정을 확인합니다. | 예. 일시적 지연이면 재시도 가능성이 높습니다. |
| `RETRIEVAL_DOCS_TIMEOUT` | 공식 문서 검색 route의 Tavily 호출이 `DOCS_SEARCH_TIMEOUT_SECONDS` 안에 끝나지 않았습니다. | 질문의 라이브러리/버전/기능명을 좁히고 다시 요청합니다. 운영자는 Tavily 상태와 timeout 설정을 확인합니다. | 예. 외부 검색 지연이면 재시도로 회복될 수 있습니다. |
| `RETRIEVAL_DOCS_FAILED` | Tavily 호출 실패, 예외, 예상과 다른 응답 타입, `results` payload 누락 등 공식 문서 검색이 실패했습니다. | 공식 문서 검색이 꼭 필요하면 다시 요청합니다. 운영자는 `TAVILY_API_KEY`, 네트워크, allowlist/domain rule을 확인합니다. | 조건부. 외부/API 문제면 원인 해소 후 재시도합니다. |
| `RAG_INDEX_MISSING` | 과거 local route의 인덱스 누락 기록을 읽기 위해 유지하는 코드이며 현재 런타임에서는 발생하지 않습니다. | 현재 파일 기반 질문은 해당 파일을 세션에 업로드해 요청합니다. | 아니요. 과거 실행 기록을 해석하는 코드입니다. |
| `LOCAL_RAG_FAILED` | upload retriever의 similarity search가 예외로 실패했습니다. 과거 결과에서는 local route 실패에도 사용됩니다. | 업로드 파일을 다시 올립니다. 운영자는 embedding/API key와 Chroma 상태를 확인합니다. | 조건부. 파일이나 API 설정을 고친 뒤 재시도합니다. |
| `UPLOAD_RETRIEVER_BUILD_FAILED` | 기존 단일 업로드 실행 경로에서 `.py` 또는 `.ipynb`의 임시 retriever 준비가 실패했습니다. 첨부 목록 API의 후보 생성 실패는 별도 HTTP 오류로 반환합니다. | 파일이 손상됐거나 너무 크지 않은지 확인하고 다시 업로드합니다. 운영자는 `OPENAI_API_KEY`와 업로드 파서/embedding 오류를 확인합니다. | 조건부. 파일 또는 설정을 고친 뒤 재시도합니다. |
| `LLM_STRUCTURED_EMPTY` | synthesis 단계에서 표시할 `AnswerDocument.blocks`가 비어 있었습니다. | 질문을 더 작게 나누거나 다시 요청합니다. 운영자는 모델 응답/structured output adapter 로그를 확인합니다. | 예. 일시적 LLM 출력 실패일 수 있습니다. |
| `SYNTHESIS_TIMEOUT` | 최종 답변 생성 단계가 timeout 또는 timed out 오류로 종료됐습니다. | 질문 범위를 줄이거나 업로드/근거 요구를 좁혀 다시 요청합니다. 운영자는 `SYNTHESIS_TIMEOUT_SECONDS`와 모델 지연을 확인합니다. | 예. 다만 큰 context가 원인이면 요청을 줄인 뒤 재시도합니다. |
| `VALIDATION_UNRESOLVED_REFERENCES` | 실제 표시 내용의 `refs`가 해당 synthesis packet에 없거나, 근거가 필요한 내용에 참조가 없습니다. 의미적 사실 판정은 아닙니다. | 더 구체적인 자료를 제공하거나 답변 범위를 좁힙니다. 운영자는 packet 선택과 출력 refs를 확인합니다. | 조건부. 확보한 검색 결과를 재사용해 본문과 참조를 함께 다시 생성할 수 있습니다. |
| `VALIDATION_MISSING_CONTENT` | 본문이 비었거나 요청한 코드·단계·체크리스트 형식 또는 출처 범위가 부족합니다. 원문 발췌가 실제 선택 범위와 일치하지 않는 경우도 포함합니다. | 원하는 결과를 구체적으로 적습니다. 운영자는 내용 단위의 checks, 요청 계약과 route coverage를 확인합니다. | 조건부. 재합성 후에도 부족하면 확인 가능한 원문 발췌와 제한을 제공합니다. |
| `DEBUG_NORMALIZATION_FAILED` | web API가 raw debug payload를 `AgentDebugInfo`로 정규화하는 중 latency/debug 구조 검증에 실패했습니다. | 답변 자체보다 관측성 정보가 불완전한 상태입니다. 운영자는 raw debug payload와 `schema_version`을 확인합니다. | 아니요. 같은 사용자 요청 반복보다 debug schema/normalizer 수정이 필요합니다. |
| `SLACK_AUTH_FAILED` | Slack token이 없거나 Slack API 호출이 인증/권한 문제로 실패했습니다. | `SLACK_BOT_TOKEN` 설정, 앱 설치, channel 접근 권한, 필요한 scope를 확인합니다. | 조건부. Slack 설정을 고친 뒤 재시도합니다. |
| `SLACK_DESTINATION_MISSING` | channel ID, user ID, email, 기본 Slack destination 중 어느 것도 유효하게 해석되지 않았습니다. | Slack 전송을 원하면 channel ID(`C/G/D...`), user ID, email 중 하나를 제공하거나 기본 destination env를 설정합니다. | 조건부. destination을 제공한 뒤 재시도합니다. |
| `UPLOAD_PATH_INVALID` | 요청 경로가 현재 session upload directory 밖이거나, 소유할 수 없는 확장자이거나, 파일이 없습니다. 경로 소유권 검사와 현재 기능의 형식 접수 여부는 별도로 검사합니다. | 현재 세션에서 파일을 다시 업로드합니다. PDF·DOCX·이미지는 문서 기능을 활성화한 뒤 첨부 목록 API를 사용하고, 임의 경로나 이전 세션 경로를 보내지 않습니다. | 아니요. 올바른 업로드 경로로 다시 요청해야 합니다. |

## 문서 첨부 HTTP 오류

[`UploadService._build_candidate()`](../src/app/web/upload_service.py)는 변환·청킹·임베딩 경계의 `IngestionError`를 아래 HTTP 상태로 매핑합니다. 응답은 `detail.code`, `detail.message`, `detail.files`를 포함하며 파일을 특정할 수 있으면 `files`에 `name`, `code`, `message`가 들어갑니다. 후보가 실패하면 새 후보 원본·인덱스를 정리하고 이전 활성 첨부 목록을 유지합니다. 이 오류는 답변 생성 전의 첨부 요청에서 반환되므로 그래프의 `include_debug`와 무관합니다.

| HTTP 코드 | 상태 | 발생 조건 | 대응과 재시도 |
|---|---|---|---|
| `DOCUMENT_INVALID` | 422 | 파일 서명·DOCX 구조·이미지 형식 검증 실패, 손상된 문서, 여러 프레임 이미지 또는 Docling의 실패·건너뜀 결과입니다. | 파일을 다시 내보내거나 다중 프레임 이미지를 페이지별 파일·PDF로 바꿔 첨부합니다. 같은 잘못된 파일의 반복 요청으로 해결되지 않습니다. |
| `DOCUMENT_NO_SEARCHABLE_CONTENT` | 422 | 변환 결과에 검색 가능한 본문이나 표가 없습니다. 그림 자리표시자만 있는 결과도 해당합니다. | 원문에 텍스트가 있는지, OCR 설정과 스캔 해상도가 적절한지 확인한 뒤 다시 첨부합니다. |
| `DOCUMENT_PARTIAL_CONVERSION` | 422 | Docling이 부분 성공을 반환했거나, 성공 결과에도 오류 또는 원본 페이지 누락이 있습니다. 남은 본문만 공개하지 않습니다. | 원본을 확인하고 문제가 있는 페이지를 다시 내보내거나 파일을 나눠 첨부합니다. |
| `DOCUMENT_SOURCE_CHANGED` | 422 | 원본을 다시 읽을 수 없거나 검증한 크기·SHA-256과 현재 바이트가 다릅니다. | 파일을 다시 첨부해 새 관리 원본을 준비합니다. |
| `DOCUMENT_LIMIT_EXCEEDED` | 413 | PDF 페이지·이미지 픽셀·DOCX 압축 해제 크기·변환 출력·worker RSS·문서 청크 수 또는 표 한 행의 크기 한도를 초과했습니다. | 파일을 나누거나 해상도·내용량을 줄입니다. 운영자는 실제 자원 측정과 한도 설정을 함께 확인합니다. |
| `DOCUMENT_PROCESSING_TIMEOUT` | 504 | Docling 변환 시간, worker 실행 기한, 첨부 후보 전체 기한 또는 문서 임베딩 요청 시간을 초과했습니다. | 일시적 지연이면 다시 시도할 수 있습니다. 반복되면 파일을 나누고 변환·임베딩 지연과 시간 한도를 확인합니다. |
| `DOCUMENT_CONVERTER_BUSY` | 503 | 제한된 변환 동시 실행 자리를 확보하지 못했습니다. | 진행 중인 변환이 끝난 뒤 다시 시도합니다. |
| `DOCUMENT_CONVERTER_UNAVAILABLE` | 503 | 변환 기능·선택 의존성·사전 모델이 준비되지 않았거나 변환기가 종료된 상태입니다. | 기능 설정, 선택 패키지와 모델 경로를 확인하고 필요한 준비 또는 서비스 재시작 후 다시 첨부합니다. |
| `DOCUMENT_CONVERSION_FAILED` | 503 | worker가 결과 없이 종료되거나 변환 자원 준비·실행에서 처리하지 못한 오류가 발생했습니다. | 일시적 자원 부족이면 다시 시도할 수 있습니다. 반복되면 worker 종료·파일 시스템·실행 환경을 확인합니다. |
| `DOCUMENT_ADAPTER_ERROR` | 500 | 변환 요청·결과의 구조, 원본 식별자·파일 목록·설정, 페이지 수 메타데이터 또는 직렬화 계약이 일치하지 않습니다. | 운영자가 adapter·worker·캐시 계약과 설치 버전을 확인합니다. 같은 요청을 반복하기보다 원인을 수정합니다. |

형식 접수 단계에서 기능이 꺼져 있거나 지원하지 않는 확장자는 `UPLOAD_TYPE_INVALID`(422)입니다. 파일 자체의 크기 한도는 `UPLOAD_FILE_TOO_LARGE`(413), 일반 파일 내용 검증 실패는 `UPLOAD_VALIDATION_FAILED`(422), 변환 오류로 분류되지 않은 검색 인덱스 준비 실패는 `UPLOAD_INDEX_FAILED`(503)로 반환합니다. 모델 준비, 한도와 실패 후 자원 수명은 [문서 변환과 OCR](document_ingestion.md)을 참고합니다.

OCR의 글자 오인식이나 문단 누락이 항상 실패 상태로 검출되지는 않습니다. 변환이 성공했어도 인용의 `quality_issues`를 유지하며, 처리 완료·원본 발췌 일치·OCR 정확성을 서로 다른 신호로 해석합니다.

## 운영 메모

- 그래프 ErrorCode의 source of truth는 `src/core/contracts/debug.py`이며, 문서 첨부 HTTP 상태 매핑은 `src/app/web/upload_service.py`에서 관리합니다.
- retrieval/action tool은 가능한 경우 payload의 `error_code`에 직접 기록합니다.
- planner/synthesis 계열 코드는 stage error 문자열을 정규화해서 debug payload의 `error_codes`에 합쳐집니다.
- benchmark histogram은 `src/eval/reporting/histograms.py`에서 같은 코드 집합을 bucket으로 집계합니다.
- 사용자에게 필요한 제한은 `AnswerResponse.issues`, 내용별 확인 결과는 `checks`, 저장·전송 실패는 `actions`에도 전달합니다. debug를 끈 상태에서도 이 정보는 유지됩니다.
- 참조가 `resolved`인 것과 설명이 원문으로 뒷받침되는 것은 구분합니다. 일반 설명의 `support_status`는 `not_evaluated`이며, 원문 발췌가 정확히 일치할 때만 `exact_match`입니다.
