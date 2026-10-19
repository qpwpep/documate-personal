# ErrorCode taxonomy

이 문서는 질문 실행 결과, 그래프의 debug/action 오류, 문서 첨부 API의 HTTP 오류를 정리합니다. "재시도 가능"은 같은 요청을 다시 보내는 것만으로 회복될 가능성입니다. 설정, 인증, 파일 상태가 원인인 코드는 원인을 고친 뒤 재시도해야 합니다.

## 질문 실행 결과와 제공자 오류

질문의 SSE 스트림은 `final_response` 하나로 완료됩니다. 공통 `TurnResult`는 `status`, `response`, `message`, `problem`, `request_id`, `missing_slots`를 전달합니다. `completed`와 `partial`만 검증된 `AnswerResponse`를 포함합니다. `needs_input`은 실제 누락된 사용자 정보에 대한 질문이며, `failed`와 `refused`는 답변 문서를 만들지 않습니다. `partial`은 이미 확보하고 검증한 근거로 제한된 답변을 제공하며 실패 원인을 함께 표시합니다. 단순 확인이나 취소 완료는 `completed`입니다.

`problem`의 `code`, `stage`, 안전한 `message`, `next_action`, `retry_after_seconds`는 `include_debug=False`에서도 유지됩니다. 제공자가 정상 질문의 출력 스키마를 거부해도 질문을 고쳐 쓰라는 보충 질문으로 바뀌지 않습니다. 기술적 실패·거절·부분 답변은 세션의 이전 답변과 보류 작업을 덮어쓰지 않으며 UI rerun이 질문을 자동 재전송하지 않습니다.

| code | 원인 | 다음 행동과 자동 복구 |
|---|---|---|
| `provider_schema_invalid` | 로컬 스키마 컴파일 실패 또는 제공자가 `response_format`/`text.format` 스키마를 거부 | `fix_configuration`. 자동 재시도·재검색·축소 합성을 하지 않습니다. 사용자 질문 수정은 필요하지 않습니다. |
| `provider_configuration` | 인증·권한·모델/매개변수 설정 오류 또는 할당량·결제 한도 부족 | `fix_configuration`. 같은 요청을 자동 반복하지 않습니다. |
| `provider_rate_limited` | 일시적 429 요청 제한 | `retry_later`. `Retry-After`와 전체 호출 예산 안에서 제한적으로 재시도합니다. |
| `provider_unavailable` | timeout, 연결 장애, 408/409/5xx 등 일시적 제공자 실패 | `retry_later`. 제한된 재시도 후 종료하며 합성 timeout만 축소 context 복구를 사용할 수 있습니다. |
| `model_output_invalid` | JSON, wire schema, 도메인 계약 또는 바인딩 검증 실패. Decisions의 잘못된 확률·중복·누락·미지 판정 이름도 포함 | Planner/synthesis는 기존 사용자 제약을 유지한 출력 수정 요청을 최대 한 번 수행합니다. Decisions의 유효하지 않은 응답은 수정 요청 없이 차단합니다. 최종 실패는 `retry_later`이며 질문 의도를 다시 묻지 않습니다. |
| `model_output_incomplete` | 출력 토큰 한도 등으로 생성이 미완료 | `retry_later`. 잘린 JSON을 정상 문서로 복구하거나 보충 질문으로 바꾸지 않습니다. |
| `call_budget_exhausted` | 다음 호출 전에 역할별 호출 횟수 또는 처리 시간 예산이 소진됨 | `retry_later`. 추가 모델 호출 없이 종료하며 출력 오류로 오분류하지 않습니다. |
| `model_refusal` | 호출 전체에 대한 명시적 refusal·content filter, 또는 근거 부족 동반 여부와 관계없이 `request_ready`가 명시적으로 거절됨 | `none`. `refused`로 표시하고 재검색·재생성하지 않습니다. Decisions의 이름별 원문·요구·사후 판정 refusal은 아래 개별 판정 정책을 적용하며 곧바로 전체 `model_refusal`로 바꾸지 않습니다. |
| `internal_error` | 분류되지 않은 내부 실행 오류 | `none`. 예외 원문 대신 안전한 안내와 request ID를 전달하며 운영자가 진단을 확인합니다. |
| `evidence_insufficient` | 검색·근거 검증 후 답변에 필요한 근거를 확보하지 못함 | `supply_information`. 기술적 검색 실패와 실제 사용자 정보 누락을 구분합니다. |
| `upload_revision_conflict` | 질문에 포함된 첨부 revision이 현재 세션과 다름 | `none`. 현재 첨부 상태를 다시 확인한 뒤 요청합니다. 자동 재전송하지 않습니다. |

Planner와 synthesis는 각 역할별 한 질문에서 최대 3회, 45초의 공통 호출 예산을 사용합니다. 전송 재시도·출력 수정·합성의 축소 복구·그래프의 재진입이 같은 예산을 공유하며 SDK 자체 재시도는 0회입니다. 제공자의 대기 시간이 남은 예산보다 크면 기다린 뒤 무조건 호출하지 않고 종료합니다. 일반 대화 요약 모델은 이 구조화 출력 정책의 대상이 아닙니다.

OpenAI Decisions API는 `pre_synthesis_validation`에 최대 6회·90초, `post_synthesis_validation`에 최대 3회·45초의 예산을 사용합니다. 사전 검증의 원문별 판정·요구 충족 판정과 한 번의 재검색 후 검증, 요청당 최대 200개 질문의 배치, 그래프 재진입, 일시적 429·timeout·연결·5xx 재시도가 같은 단계 예산을 공유합니다. 각 HTTP timeout은 `VALIDATION_TIMEOUT_SECONDS` 기본 20초와 남은 예산 중 작은 값입니다. 인증·권한·할당량 오류와 유효하지 않은 판정 응답은 호출 전체를 실패 처리합니다. 한 항목의 refusal 뒤에 잘못된 이름·중복·누락·잘못된 확률이 있어도 검사를 끝까지 적용하며, 앞선 배치의 정상 결과만으로 성공을 선언하지 않습니다. 호출 실패는 `problem.stage`로 구분하고 답변 제공과 저장·전송을 차단합니다.

유효한 이름별 refusal은 `DecisionResult.refused_names`로 반환하고 다른 이름의 정상 확률은 `probabilities`에 유지합니다. 사전 원문 판정은 `(requirement_id, evidence_id)`별 `passed`·`failed`·`refused`를 기록하며, 실패·거절 원문만 제외합니다. `request_ready`가 명시적으로 거절되면 근거 부족을 동반해도 `model_refusal`로 재검색과 생성을 중단하며 부분 답변을 만들지 않습니다. 이 거절이 없고 남은 통과 원문으로 요구가 충족되면 재검색하지 않습니다. 요구가 부족하거나 요구 판정이 거절되면 기존 정책에 따라 지원하는 docs 또는 upload 주제·aspect 검색의 실패 요구만 기본 한 번 재검색하고, 이미 통과한 원문은 재사용합니다. 재검색 불가·한도 소진 후에도 부족하면 검증한 남은 근거로 `partial` 답변을 제공하되 저장·전송을 막으며, 근거가 없으면 `evidence_insufficient`로 종료합니다. `request_ready`의 수치 확률이 낮은 경우는 근거 요구가 부족할 때만 검색 복구를 판단하고, 근거 요구가 충족됐다면 `evidence_insufficient`로 생성과 재검색을 중단합니다.

사후 개별 refusal은 `semantic_validation.checks`의 `status="refused"`, `probability=null`로 기록합니다. 해당 내용이나 요청 충족을 미통과로 처리해 기존의 제한된 본문 보정·필터링·재검증 정책을 적용합니다. 거절을 확률 0으로 바꾸거나 통과로 간주하지 않으며, refusal 원문은 결과·로그에 남기지 않습니다.

시간 예산은 다음 호출의 시작 여부와 각 HTTP I/O timeout을 제한합니다. 진행 중인 동기 네트워크 작업을 정확히 45초 또는 90초에 강제 취소하는 전체 요청 deadline은 아닙니다.

개발자 진단은 `debug.llm_diagnostics`와 구조화 로그에 오류 코드, 단계, 모델, endpoint, 스키마 이름·hash, compiler·OpenAI·LangChain 버전, 제공자 status/code/type/param/request ID, 시도와 복구 결정을 기록합니다. 요청 본문·모델 원문 출력·API 키·제공자 예외 메시지를 진단 필드에 저장하지 않습니다. `debug.error_codes`도 예외 문자열 추측 대신 분류된 문제를 사용합니다.

## 그래프 debug·action ErrorCode

아래 코드는 [`src/core/contracts/debug.py`](../src/core/contracts/debug.py)의 `ErrorCode` Literal을 기준으로 debug payload와 action result에 기록됩니다. 뒤의 문서 첨부 HTTP 코드는 별도 계약이며 이 Literal이나 benchmark histogram의 코드 집합에 자동으로 포함되지 않습니다.

| ErrorCode | 언제 발생하나 | 사용자가 할 일 | 재시도 가능 |
|---|---|---|---|
| `RETRIEVAL_DOCS_TIMEOUT` | 공식 문서 검색 route의 Tavily 호출이 `DOCS_SEARCH_TIMEOUT_SECONDS` 안에 끝나지 않았습니다. | 질문의 라이브러리/버전/기능명을 좁히고 다시 요청합니다. 운영자는 Tavily 상태와 timeout 설정을 확인합니다. | 예. 외부 검색 지연이면 재시도로 회복될 수 있습니다. |
| `RETRIEVAL_DOCS_FAILED` | Tavily 호출 실패, 예외, 예상과 다른 응답 타입, `results` payload 누락 등 공식 문서 검색이 실패했습니다. | 공식 문서 검색이 꼭 필요하면 다시 요청합니다. 운영자는 `TAVILY_API_KEY`, 네트워크, allowlist/domain rule을 확인합니다. | 조건부. 외부/API 문제면 원인 해소 후 재시도합니다. |
| `RAG_INDEX_MISSING` | 과거 local route의 인덱스 누락 기록을 읽기 위해 유지하는 코드이며 현재 런타임에서는 발생하지 않습니다. | 현재 파일 기반 질문은 해당 파일을 세션에 업로드해 요청합니다. | 아니요. 과거 실행 기록을 해석하는 코드입니다. |
| `LOCAL_RAG_FAILED` | upload retriever의 similarity search가 예외로 실패했습니다. 과거 결과에서는 local route 실패에도 사용됩니다. | 업로드 파일을 다시 올립니다. 운영자는 embedding/API key와 Chroma 상태를 확인합니다. | 조건부. 파일이나 API 설정을 고친 뒤 재시도합니다. |
| `UPLOAD_RETRIEVER_BUILD_FAILED` | 과거 단일 업로드 실행 경로의 retriever 준비 실패 기록입니다. 현재 질문 런타임은 인덱스를 생성하지 않으며 첨부 후보 실패는 HTTP 오류로 반환합니다. | 과거 결과의 실패 원인을 읽는 코드입니다. 현재 첨부 실패는 아래 HTTP 코드로 확인합니다. | 아니요. 현재 런타임에서는 발생하지 않습니다. |
| `VALIDATION_UNRESOLVED_REFERENCES` | 실제 표시 내용의 `refs`가 해당 synthesis packet에 없거나, 근거가 필요한 내용에 참조가 없습니다. 의미적 사실 판정은 아닙니다. | 더 구체적인 자료를 제공하거나 답변 범위를 좁힙니다. 운영자는 packet 선택과 출력 refs를 확인합니다. | 조건부. 확보한 검색 결과를 재사용해 본문과 참조를 함께 다시 생성할 수 있습니다. |
| `VALIDATION_MISSING_CONTENT` | 본문이 비었거나 요청한 형식·출처 범위가 부족합니다. 원문 발췌 불일치와 Decisions의 내용 근거·요청 충족 판정이 임계값에 못 미치는 경우도 포함합니다. | 운영자는 로컬 `checks`, `semantic_validation`, 요청 계약과 요구별 근거를 확인합니다. 사용자는 부족한 자료나 조건을 보완할 수 있습니다. | 조건부. 같은 근거와 요청으로 본문을 보정합니다. 끝내 통과하지 못하면 실패 또는 제한된 원문 안내를 반환하고 저장·전송을 차단합니다. |
| `DEBUG_NORMALIZATION_FAILED` | web API가 raw debug payload를 `AgentDebugInfo`로 정규화하는 중 latency/debug 구조 검증에 실패했습니다. | 답변 자체보다 관측성 정보가 불완전한 상태입니다. 운영자는 raw debug payload와 `schema_version`을 확인합니다. | 아니요. 같은 사용자 요청 반복보다 debug schema/normalizer 수정이 필요합니다. |
| `SLACK_RECIPIENT_MISSING` | 수신자와 기본값이 없습니다. | 수신자 하나를 입력합니다. | 입력 후 |
| `SLACK_RECIPIENT_INVALID` | 명시한 수신자를 검증하지 못했습니다. | ID 또는 이메일을 확인합니다. | 명시 교정 후 |
| `SLACK_RECIPIENT_AMBIGUOUS` | 수신자 입력이 모호합니다. | 대상 하나를 지정합니다. | 명시 교정 후 |
| `SLACK_RECIPIENT_CONFLICT` | 자연어와 요청 필드의 수신자가 충돌합니다. | 두 입력을 같은 대상으로 맞춥니다. | 명시 교정 후 |
| `SLACK_CONFIGURATION_ERROR` | 기본값이 복수이거나 잘못된 형식입니다. | 기본 사용자 ID 또는 이메일 하나만 설정합니다. | 설정 수정 후 |
| `SLACK_TARGET_NOT_FOUND` | 명시한 이메일/사용자에 해당하는 활성 대상을 조회하지 못했습니다. | 대상과 워크스페이스를 확인합니다. | 같은 대상 또는 명시 교정 후 |
| `SLACK_TARGET_UNAVAILABLE` | 대상이 없거나 현재 앱이 접근할 수 없습니다. | 대상 ID·채널 접근을 확인합니다. | 접근 확인 후 같은 대상 |
| `SLACK_PERMISSION_DENIED` | Slack 앱 권한이 부족합니다. | 필요 scope와 앱의 채널 접근을 확인합니다. | 권한 수정 후 같은 대상 |
| `SLACK_AUTHENTICATION_FAILED` | 토큰이 없거나 인증에 실패했습니다. | SLACK_BOT_TOKEN과 앱 설치를 확인합니다. | 설정 수정 후 같은 대상 |
| `SLACK_RATE_LIMITED` | Slack이 요청을 제한했습니다. | retry_after_seconds 이후 같은 대상에 재시도합니다. | 같은 대상 |
| `SLACK_TEMPORARY_FAILURE` | 조회 또는 DM 개설에 일시적 장애가 발생했습니다. | 동일 선택자로 다시 시도합니다. | 같은 대상 |
| `SLACK_DELIVERY_UNKNOWN` | 전송 요청의 결과를 확인하지 못했습니다. | Slack에서 전달 여부를 확인합니다. | 자동 재전송 금지 |
| `SLACK_PROTOCOL_ERROR` | Slack 응답이 필요한 대상 정보를 충족하지 못했습니다. | 응답과 앱 설정을 확인합니다. | 결과의 next_action 확인 |
| `UPLOAD_PATH_INVALID` | 과거 질문에 포함된 업로드 경로가 세션 경계를 벗어나는 등의 검증 실패 기록입니다. 현재 질문은 경로를 받지 않으며 같은 이름의 첨부 HTTP 오류와 기록 위치가 다릅니다. | 과거 결과의 진단은 유지하고 현재 첨부 요청은 아래 HTTP 코드를 확인합니다. | 아니요. 현재 그래프에서는 발생하지 않습니다. |

## 문서 첨부 HTTP 오류

[`UploadService._build_candidate()`](../src/app/web/upload_service.py)는 변환·청킹·임베딩 경계의 `IngestionError`를 아래 HTTP 상태로 매핑합니다. 응답은 `detail.code`, `detail.message`, `detail.files`를 포함하며 파일을 특정할 수 있으면 `files`에 `name`, `code`, `message`가 들어갑니다. 후보가 실패하면 새 후보 원본·인덱스를 정리하고 이전 활성 첨부 목록을 유지합니다. 이 오류는 답변 생성 전의 첨부 요청에서 반환되므로 그래프의 `include_debug`와 무관합니다.

| HTTP 코드 | 상태 | 발생 조건 | 대응과 재시도 |
|---|---|---|---|
| `UPLOAD_PATH_INVALID` | 422 | 추가 경로가 현재 세션의 업로드 디렉터리 밖이거나 허용된 파일 참조가 아닙니다. 경로 소유권과 현재 기능의 형식 접수 여부는 별도로 검사합니다. | 현재 세션에서 파일을 다시 준비하고 임의 경로나 다른 세션 경로를 보내지 않습니다. |
| `UPLOAD_CONTENT_CHANGED` | 422 | 추가 요청의 필수 `content_hash`와 서버가 읽은 파일 bytes의 SHA-256이 다릅니다. | 변경된 bytes로 staging과 새 요청을 만들고 교체 대상을 다시 확인합니다. 같은 operation의 요청 내용만 바꾸지 않습니다. |
| `UPLOAD_REVISION_CONFLICT` | 409 | 첨부 변경의 epoch 또는 신규 작업의 expected revision이 현재 세션과 다릅니다. 질문 context 충돌은 SSE `error`로 전달합니다. | GET으로 현재 manifest를 확인하고 변경을 다시 검토합니다. 질문은 자동 재전송하지 않습니다. |
| `UPLOAD_OPERATION_CONFLICT` | 409 | 최근 성공 이력에 있는 operation ID를 다른 요청 fingerprint로 재사용했습니다. | 결과 확인에는 원래 요청을 그대로 사용하며, 새 변경에는 새 operation ID를 사용합니다. |
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
- planner/synthesis와 Decisions 검증의 제공자 오류는 `src/core/llm_errors.py`의 분류된 문제를 사용합니다. 의미 판정이 임계값에 못 미치는 경우와 API 자체의 실패를 구분합니다. 과거 대문자 LLM 오류 코드는 기존 실행 기록을 읽는 용도로만 남아 있습니다.
- benchmark histogram은 `src/eval/reporting/histograms.py`에서 같은 코드 집합을 bucket으로 집계합니다.
- 사용자에게 필요한 제한은 `AnswerResponse.issues`, 로컬 확인 결과는 `checks`, 의미 판정은 `semantic_validation`, 저장·전송 실패는 `actions`에도 전달합니다. debug를 끈 UI·API 응답에서도 이 정보는 유지됩니다. 의미 판정의 확률은 저장 파일과 Slack 본문에는 추가하지 않습니다.
- 참조가 `resolved`인 것과 설명이 원문으로 뒷받침되는 것은 구분합니다. 로컬 `checks.support_status`는 일반 설명에서 `not_evaluated`, 정확한 원문 발췌에서 `exact_match`입니다. 별도의 `semantic_validation`은 본문 hash와 근거 fingerprint에 연결된 `grounded`·`request_fulfilled`의 판정 상태와 확률을 보존합니다. 모든 항목이 `evaluated`이고 각 확률이 임계값 이상이어야 의미 검증을 통과합니다. `refused` 항목이나 보고서가 없는 원문 안내는 의미 검증 완료로 표시하지 않습니다.
