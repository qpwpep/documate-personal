# Design Rationale

DocuMate의 설계 판단과 기술적 선택

## 1. 문서 목적

이 문서는 DocuMate의 문서·검색·답변·검증 경계를 나눈 이유와 현재 구현의 트레이드오프를 설명합니다. 제품의 목적은 공식 문서와 사용자 자료를 이해하고 설명하며, 사용자가 설명에서 당시 원문으로 돌아가 확인할 수 있게 하는 것입니다.

DocuMate는 LangGraph 기반 학습 보조 에이전트입니다. 현재 구조는 공식 문서 검색, 세션 업로드 파일 검색, 구조화된 응답, 저장 및 Slack 전송 액션을 FastAPI와 Streamlit 런타임 위에서 함께 제공합니다. 실행 기준 코드는 `src/app`, `src/core`, `src/infra`, `src/runtime`, `src/eval` 계층으로 분리되어 있고, `archive`는 현재 실행 경로가 아니라 팀 프로젝트 원형과 참고 자료를 보관하는 영역입니다.

## 2. 핵심 설계 판단

### 단계형 LangGraph 파이프라인

초기 구조처럼 모델의 tool call 흐름에만 실행을 맡기면 검색, 검증, 액션의 책임 경계가 흐려지기 쉽습니다. 그래서 현재 그래프는 `add_user_message`, `summarize_old_messages`, `planner`, `retrieve_dispatch`, `pre_synthesis_validation`, `synthesize`, `post_synthesis_validation`, `action_postprocess` 단계로 나누었습니다.

이 구조의 목표는 각 단계가 명확한 상태 계약을 주고받게 만드는 것입니다. `GraphState`는 `runtime`, `planner`, `retrieval`, `retry`, `response`, `debug` 영역으로 나뉘며, boundary adapter가 dict와 Pydantic 모델 사이의 상태를 정규화합니다. planner는 확인 질문이 필요한지 또는 어떤 근거 요구를 검색할지 결정하고, retrieval은 각 `requirement_id`에 대응하는 `SearchHit`와 diagnostics를 모읍니다. synthesis는 실제 제공받은 원문 범위로 본문을 생성하고, validation은 그 본문의 원문 참조·발췌 일치·요구별 근거와 관찰 가능한 출력 조건을 확인합니다.

### 유한한 장기 대화 메모리

대화 메모리는 `bounded rolling summary + bounded recent canonical messages`라는 두 부분으로 나눴습니다. `ConversationMemoryPolicy`가 Human turn, 추정 token, UTF-8 직렬화 byte, message 수의 high/low watermark와 최종 hard byte limit을 한곳에서 관리합니다. 어느 high watermark든 도달하면 가장 오래된 완결 turn부터 퇴출해 모든 low watermark를 만족하는 가장 긴 최근 suffix를 남깁니다. 정확한 모델 tokenizer가 없어도 실제 크기 상한이 유지되도록 token 값은 보수적인 추정치로 사용하고, UTF-8 직렬화 byte를 독립적인 backstop으로 검사합니다.

`GraphState.messages`는 LangGraph의 `add_messages` reducer를 사용하므로 최근 리스트만 반환해서는 누락된 과거 메시지가 삭제되지 않습니다. 요약 노드는 `RemoveMessage(id=REMOVE_ALL_MESSAGES)`와 retained suffix를 함께 반환해 reducer에 전체 교체 의도를 명시합니다. router와 요약 노드는 같은 pure compaction plan을 사용하므로 trigger 판단과 실제 퇴출 범위가 어긋나지 않습니다.

summary도 별도로 bounded합니다. 새 summary는 `기존 bounded summary + 이번에 새로 퇴출된 대화`를 하나의 replacement memory로 다시 작성한 결과이며, 이전 summary 뒤에 새 문자열을 append하지 않습니다. `SUMMARY_MAX_TOKENS`는 LLM 생성 여유를, `MEMORY_SUMMARY_MAX_TOKENS`와 `MEMORY_SUMMARY_MAX_BYTES`는 저장 길이를 관리하므로 생성 예산 조정이 대화 메모리 보존량을 암묵적으로 바꾸지 않습니다. Tool/System payload는 요약 입력에서 제외합니다. summarizer 예외나 빈 출력에는 기존 summary와 새 transcript의 head/tail을 함께 보존하는 deterministic fallback을 적용하고, 결과를 token·byte 상한에 다시 맞춥니다. fallback 사용 여부와 before/after 크기는 기록하지만 대화 내용은 로그에 남기지 않습니다.

planner와 synthesis에 전달되는 summary는 과거 사용자 입력에서 유래한 비신뢰 데이터입니다. 고정 System policy가 그 안의 명령을 따르거나 검색 evidence로 취급하지 말라고 명시하고, 실제 summary payload는 system instruction이 아닌 별도 assistant data message로 전달합니다.

### 검색 소스와 근거 요구의 분리

검색 소스는 `docs`, `upload` route로 분리했습니다. 공식 문서 검색과 세션 업로드 파일 검색은 데이터 출처와 확인 기준이 다르기 때문입니다. route는 도구 선택 기준이고, 답변에 필요한 근거의 단위는 `RetrievalTask`입니다. NumPy와 pandas를 비교하는 질문처럼 같은 route에 여러 대상이 있어도 task를 합치지 않습니다. 계획은 최대 8개 독립 task를 유지하며 `requirement_id`로 검색·재시도·본문 근거를 연결합니다.

`RetrievalRequirement`에는 소유 라이브러리 `library`, 정확한 대상 `symbols`, 명시 버전 `version`, 확인할 구체 식별자·매개변수 `aspects`, 근거 종류 `match`를 둡니다. `topic`은 넓은 설명, `symbol`은 API·코드 사용, `definition`은 업로드 함수·클래스 구현입니다. 검색 문자열 `query`를 바꾸더라도 이 요구가 사라지지 않게 하는 것이 별도 계약을 둔 이유입니다. 일반적인 설명·표현 지시는 query에 남기고, 사용자 대화에 명시된 literal anchor만 aspects의 필수 조건으로 인정합니다. 모델이 예상한 옵션값까지 필수 조건으로 만들면 올바른 문서도 거절할 수 있으므로, 검색 가설과 사용자 제약을 구분합니다. task의 `k`도 도구에 전달합니다.

필요한 출처는 업로드 가용성과 무관하게 LLM이 판단합니다. 다만 대화로도 지시 대상이나 비교 버전을 정하지 못하면 `clarification_question`과 검색 없는 계획을 반환합니다. 파일이 없다는 실행 제약, 질문이 미해결인 상태, planner 호출·출력 실패를 서로 구분합니다. planner의 검색어 후처리는 공백만 정규화하며, docs 도구의 별칭 정규화도 원래 대상·버전 요구를 유지합니다.

`docs` route는 Tavily와 [agent_rules.toml](../src/infra/config/agent_rules.toml)의 허용 source를 사용합니다. 명시 라이브러리의 domain을 먼저 고르고, URL·path prefix·원문 유효성과 함께 심볼의 문서 소유권·요청 버전·발췌의 aspect를 검사합니다. 다른 API를 설명하는 문서에 요청 이름이 언급됐다는 이유만으로 해당 API 근거를 확보한 것으로 취급하지 않습니다. `upload` route는 현재 세션의 단일 `.py` 또는 `.ipynb`에 한정합니다. 일반 질문은 Chroma 후보를 검색하고, 명시 코드 심볼은 같은 retriever의 전체 source registry를 AST로 조회합니다.

`retrieve_dispatch`는 독립 task를 병렬 실행한 뒤 planner 순서로 결과를 정렬합니다. 성공과 실패를 requirement ID로 구분하므로, 같은 docs route의 한 대상만 실패해도 다른 대상의 근거는 재사용할 수 있습니다. 실행 `status`와 별도로 `answerability`를 `covered`, `partial`, `missing`, `unknown`으로 기록해, 결과 개수나 route 실행 성공을 답변 가능성과 혼동하지 않습니다.

### 문서 원문과 검색 결과의 분리

원문의 정체성과 검색 편의를 서로 다른 계약으로 관리합니다. [`documents.py`](../src/core/documents.py)의 `DocumentSnapshot`은 출처 URI·유형, 내용 hash, parser 이름·버전·설정, 수집 범위를 식별합니다. `ParsedDocument`와 `DocumentElement`는 부모 관계·제목 level·읽기 순서·제목 경로, 코드와 표 셀을 표현합니다. `SourceAnchor`는 여러 페이지·영역·원본 줄·Notebook cell 등 확보한 위치만 기록합니다. 좌표가 없으면 원문 요소나 문서 수준으로 남기며 정밀 위치를 추정해 채우지 않습니다.

[`evidence.py`](../src/core/evidence.py)의 `EvidenceRef`는 snapshot, 당시 원문 요소 전체, 해당 요소 안의 문자 또는 표 셀 선택을 묶습니다. ID는 이 데이터에 따라 결정되므로 같은 경로의 파일을 바꾸거나 검색 chunk 크기를 바꾸어도 이전 인용이 새 내용으로 조용히 연결되지 않습니다. `SearchHit`는 여기에 `requirement_id`, 질의별 순위와 `RetrievalScore`를 붙인 검색 결과입니다. 같은 원문이 여러 요구를 지원해도 검색 유래를 각각 보존합니다. 검색 점수는 답변의 정답 확률로 표시하지 않습니다.

현재 `.py`·`.ipynb` 경로는 기존 파서와 Chroma 검색·AST 조회를 사용하면서 이 모델로 원문을 전달합니다. 검색용 제목·옵션 요약을 추가하더라도 인용 발췌는 저장한 원문 요소의 범위에서 얻습니다. synthesis의 문자 예산 때문에 범위를 줄이면 줄인 범위를 가리키는 새 근거 ID를 만들고, `evidence_requirement_map`도 실제 packet ID에 연결합니다. 원래 source element를 보관하고 있더라도 모델에 보내지 않은 범위를 해당 답변의 근거로 인정하지 않습니다.

전체 원문을 모든 chunk의 metadata에 복제하면 작은 파일도 인덱스와 메모리를 크게 늘릴 수 있습니다. [`ChunkedDocument`](../src/infra/chunking.py)는 `ParsedDocument`를 한 번 보관하고, Chroma에는 chunk 텍스트와 snapshot·element·범위 참조만 저장합니다. vector 검색 범위는 원문과 대조해 `EvidenceRef`로 복원하고, 정확한 코드 심볼은 같은 source registry에서 범위를 선택합니다. 이 보관은 현재 세션의 process-local retriever 수명에 한정되지만 이미 반환된 근거는 독립적으로 원문 요소를 보존하므로 인덱스 cleanup이 과거 답변을 손상하지 않습니다.

웹 검색 결과는 provider의 `content`·`raw_content` 모두 `provider_excerpt`입니다. 외부 URL의 전체 문서나 변환 결과를 확보한 것으로 간주하지 않습니다. 사용자는 수집 당시의 발췌와 현재 URL을 구분해서 볼 수 있습니다.

### 표시하는 내용을 유일한 응답 본문으로 사용

LLM은 [`AnswerDocument`](../src/core/answer_schema/models.py)의 `blocks`만 생성합니다. 문단·제목·목록·코드·표 안의 각 `ContentUnit`에 실제 표시할 `text`, 표현 성격인 `basis`, 사용한 근거 `refs`를 붙입니다. 문단은 지지 근거나 표현 성격이 달라지는 지점에서 나누고, UI에서는 한 문단으로 읽을 수 있게 이어 붙입니다. 독립적인 주장 목록이나 요약 문자열을 별도로 생성하지 않습니다.

[`iter_content_units()`](../src/core/answer_schema/models.py)는 문장뿐 아니라 제목·표 헤더·표 셀·코드도 같은 읽기 순서로 순회합니다. 검증, 인용 번호, UI, export가 이 본문을 기준으로 동작합니다. 모델이 사실을 다른 필드에 반복해 쓰거나 검증한 내용과 다른 문자열을 마지막에 선택하는 경로를 두지 않습니다.

서버는 `AnswerResponse`에 `content`, `citations`, `checks`, `issues`, `actions`, `content_hash`, `retrieval_required`를 구성합니다. `citations`에는 실제 사용한 원문 참조만 남기고 최초 사용 순서로 번호를 붙입니다. 검사는 내용 위치 ID에 연결하고 본문 hash로 revision을 확인합니다. `retrieval_required`는 재검증에 필요한 출처 요구 조건을 보존합니다. 저장·Slack·대화용 문자열은 같은 본문에서 export하며 별도 생성 텍스트를 저장하지 않습니다.

### 원문 연결 확인과 의미적 지지 평가의 구분

참조 ID가 존재한다고 설명이 사실이거나 근거가 충분하다고 단정할 수는 없습니다. 서버의 기본 검사는 이번 synthesis에 실제 제공한 근거에 참조가 연결되는지, 문자 범위·표 셀이 유효한지, `excerpt` 내용이 하나의 원문 발췌와 정확히 일치하는지를 확인합니다. 일반 설명·해석은 `support_status="not_evaluated"`로 남고, 정확히 일치한 원문 발췌만 `exact_match`입니다.

`basis`의 `source`, `inference`, `example`, `interaction`, `excerpt`는 모델이 제안한 표현 구분입니다. 이를 사실 판정이나 검증 면제 인증으로 사용하지 않습니다. 프롬프트는 사실을 `interaction`으로 위장하지 않도록 지시하지만, 현재 런타임이 모든 자연어 사실 주장을 의미적으로 판별하지는 않습니다. 이 한계는 UI의 “근거 확인 범위”에 명시하고 benchmark의 LLM judge에서 별도로 평가합니다.

전체 confidence 수치는 제공하지 않습니다. 원문 수집 범위·품질 진단, 참조 연결 상태, 원문 발췌 일치, 생성 실패·불완전함을 각각 전달합니다. 생성 코드에는 예시임을 표시하며, 코드 실행 도구를 호출하지 않은 응답을 실행 검증된 결과로 간주하지 않습니다.

### 본문 가까이에서 원문을 확인하는 UI

Streamlit은 `AnswerResponse` 전체를 채팅 기록에 보존하고, 각 블록 아래의 인용 popover에서 발췌와 원문 요소를 보여줍니다. 제목 경로·줄·Notebook cell·페이지 메타데이터·내용 hash가 발췌와 연결됩니다. 표는 병합과 선택 셀을 보존하며 코드의 줄바꿈·들여쓰기를 유지합니다. 원문을 현재 업로드 파일에서 다시 읽는 방식이 아니므로 파일이 바뀐 뒤에도 이전 답변의 근거를 확인할 수 있습니다.

저장과 전송 결과는 `ActionReceipt`로 분리합니다. 성공·실패·보류 상태와 다운로드를 본문과 구분해 표시하고, 도구 실행 결과를 답변 문장에 끼워 넣지 않습니다. 후속 저장·전송 요청은 typed `previous_response`를 사용해 직전 본문과 근거를 함께 전달합니다.

### 검증과 선택적 재시도

검색 결과가 있어도 필요한 대상이나 조건이 모두 포함되는 것은 아닙니다. synthesis 전에는 requirement별 결과·가용성·도구 오류와 `answerability`를 검사합니다. `covered`는 명시 대상·제약에 대응하는 근거 확보, `partial`은 일부 미충족, `missing`은 현재 검색에서 근거 미확보, `unknown`은 충족이나 부재를 판단할 수 없는 상태입니다. upload에서 대상 부재를 단정하려면 전체 source를 검사해야 합니다. 명시 요구는 `covered`가 필요하지만 제약 없는 일반 topic에는 유효한 검색 후보로 설명을 생성할 수 있습니다. 이 구분은 검색 점수 임계값으로 대체하지 않습니다.

synthesis 후에는 실제 packet과 본문이 인용한 범위에서 참조·발췌 일치, requirement 연결, literal aspect와 요청한 출력 구조를 확인합니다. 모델 입력에도 요구별 근거 ID·남은 aspect·빠진 aspect와 각 source의 `is_partial`을 전달합니다. map은 검색 유래를 기록하므로, 그 ID 하나를 인용한 사실만으로 모든 aspect가 남아 있다고 보지 않습니다. 코드 예제는 코드 블록, 단계는 순서 있는 목록, 체크리스트는 목록처럼 관찰 가능한 형식을 검사하며, 자연어 설명의 의미적 충족은 별도 평가 영역으로 남깁니다. 고정된 출처별 섹션을 강제하지 않습니다.

`RetryState`는 현재 시도 시작 위치, 원래 task, `failed_requirement_ids`, `preserved_hits`, 보존 진단과 retry scope를 관리합니다. `refresh_routes`에서는 실패한 요구의 query를 다시 계획하고 같은 route의 성공한 다른 요구도 보존합니다. 재계획에는 실제 query·실패 진단·이미 시도한 query를 제공하며, 원래 ID·라이브러리·심볼·버전·aspect는 유지합니다. docs 도구는 같은 요구의 부분 후보도 새 후보와 합쳐 동일한 대상·버전·source 범위로 다시 검사합니다. 재사용은 점수가 높았다는 이유로 검사를 생략하는 경로가 아닙니다.

본문 참조나 요구 coverage 문제가 있고 원문 검색 결과가 남아 있으면 `reuse_hits_resynthesize`에서 planner·검색을 반복하지 않고 synthesis로 직접 돌아갑니다. docs·upload·hybrid에 같은 제한된 재합성 정책을 적용합니다. 기본 재시도 상한은 1회이고 `max_retries=0`도 존중합니다. 재시도 후 잘못된 내용 단위를 제거하면 참조·인용뿐 아니라 남은 requirement coverage도 다시 검사합니다. 필요한 내용을 충족하지 못하면 확인 가능한 발췌와 불완전함을 표시합니다.

planner 호출이나 출력 검증 실패는 `planner_unavailable`, 해석되지 않은 질문은 `clarification_question`, 업로드 부재는 파일 안내로 구분합니다. 필요한 입력을 해결하기 전에는 검색·저장·Slack 전송을 진행하지 않아, 실패 안내나 이전 답변이 요청한 결과물로 전달되지 않도록 합니다.

### FastAPI + Streamlit 런타임 분리

FastAPI는 실제 API 실행과 세션 관리를 담당하고, Streamlit은 사용자가 흐름을 확인하는 인터페이스 역할을 합니다. 웹 요청은 `AgentRequestService`에서 시작해 `InMemorySessionStore`와 세션별 `AgentFlowManager`를 거쳐 그래프를 실행합니다. `AgentFlowManager`는 `ExecutionRunner`, `DebugCollector`, `ResponseAssembler`, `SessionContext`를 묶는 facade 역할을 합니다.

세션별 manager cache, TTL/LRU 기반 정리, 요청 lock, SSE progress, 업로드/생성 파일 cleanup을 포함해 데모 UI와 실제 실행 경로가 같은 런타임을 바라보게 했습니다. `/agent/stream`은 `ProgressEmitter`로 request, stage, progress snapshot, final response, error, done 이벤트를 내보냅니다.

`SessionContext`는 최근 messages와 `memory_summary`를 하나의 immutable conversation snapshot으로 소유합니다. graph가 반환한 전체 메시지는 먼저 debug와 response assembly가 사용합니다. 사용자가 볼 응답까지 정상적으로 조립된 뒤에만 Tool/System/중간 AI를 제거하고 각 Human turn과 canonical final AI를 남겨 summary와 함께 단일 참조 교체로 commit합니다. graph, debug, assembly, projection 중 하나라도 실패하면 이전 정상 snapshot은 그대로 유지됩니다. 이 원자성은 대화 메모리에 한정되며 이미 실행된 파일 저장·Slack 전송 같은 외부 side effect까지 rollback하지는 않습니다.

Streamlit과 online benchmark는 `POST /agent/stream`으로 같은 진행 이벤트와 최종 응답을 받습니다. 전송 계층은 `final_response.data`의 `response`, `trace`, `debug` 전체를 전달하고, benchmark는 이 값들을 결과에 보존합니다. Streamlit 대화 기록에는 최종 `AnswerResponse`와 오류 메시지를 유지합니다. HTTP `200`이나 `done`만으로 성공을 판정하지 않으며, 실행 중 오류 뒤에 최종 응답이 오는 경우에도 오류 진단을 보존합니다. Streamlit은 통신 실패 때 요청을 자동으로 다시 보내지 않아 이미 실행된 저장·Slack 전송이 중복되지 않도록 합니다.

## 3. 주요 트레이드오프

### 단순한 tool agent보다 명시적 graph를 선택

단순한 tool agent는 구현이 빠르고 코드가 짧습니다. 대신 route 선택, evidence 정규화, 재시도 조건, 액션 후처리 같은 정책이 프롬프트와 런타임 곳곳에 흩어질 수 있습니다.

DocuMate는 포트폴리오 프로젝트이지만, 검색 품질과 근거 검증을 핵심 역량으로 보여주는 것이 중요했습니다. 그래서 구현량이 늘어나더라도 단계별 graph와 node 책임을 명시하는 방향을 선택했습니다. 일반 기술 설명과 사용자 파일 조회에는 같은 단어가 등장하므로, 출처의 지정·제외와 문맥 해석은 planner LLM에 맡깁니다. 후처리는 스키마와 실행 가용성을 확인하며 정규식으로 출처를 추가하거나 제거하지 않습니다. 의미 판단은 모델에 의존하지만, 계획 실패 시 명확한 안내로 종료해 다른 판별 규칙이 출처를 추측하며 실행을 이어 가지 않도록 했습니다.

### 여러 검색 소스를 하나로 합치지 않음

공식 문서와 업로드 파일을 하나의 retriever처럼 다루면 인터페이스는 단순해집니다. 하지만 답변이 어떤 근거를 사용했는지 설명하기 어렵고, 실패 원인을 route별로 추적하기도 어렵습니다.

현재 구조는 source별 처리와 요구별 상태 관리가 필요하지만, evidence 출처와 실패 원인을 명확히 남기는 쪽을 우선했습니다. `RetrievalDiagnostic`에는 route·requirement ID, 실행 status, answerability, 미충족 요구, 후보 수, 시도 query, 재사용 여부와 provider·URL 검증 시간, 필터 수, 경고가 남습니다. 같은 route의 여러 대상 중 무엇이 실패했는지 구분할 수 있으며, 이 신호를 의미적 정답률과 혼동하지 않습니다.

### 원문 전체 보존보다 bounded rolling memory를 선택

요약은 본질적으로 손실 압축이므로 오래된 원문을 전부 보존하는 것과 같은 의미 충실도를 보장하지 않습니다. 대신 process-local session memory와 다음 prompt 크기, 요약 호출 비용에 명시적인 상한을 둘 수 있습니다. high/low watermark를 사용해 한 번의 compaction에서 target까지 내리므로 window가 찬 뒤 매 요청마다 한 turn만 요약하는 진동도 줄였습니다.

ToolMessage 원문과 provider metadata를 durable snapshot에 저장하지 않는 선택 역시 같은 트레이드오프입니다. 현재 요청의 response assembly와 debug에는 전체 payload를 사용하지만 다음 요청에는 사용자 질문과 실제 표시된 assistant 답변만 넘깁니다. 완전한 event replay 가능성은 줄어드는 대신 검색 원문·파일 내용·tool receipt가 장기 대화에 반복 주입되는 비용과 개인정보 노출 면적을 줄입니다.

### 테스트와 benchmark에 운영 비용을 투자

개인 프로젝트에서 120-case release benchmark와 pytest 기반 회귀 테스트를 유지하는 것은 비용이 있습니다. fixture 관리, judge 설정, latency 및 비용 지표 확인이 필요하기 때문입니다.

대신 변경 후 품질을 감으로 판단하지 않아도 됩니다. 회귀 테스트의 실제 결과와 기록된 release benchmark는 [README의 검증 결과](../README.md#검증-결과)에서 확인합니다. 문서에 결과를 중복 복사하지 않으며, 평가 계약 변경 전의 기록을 새 계약의 검증 결과로 간주하지 않습니다.

### 작은 내부 모델과 parser 경계

Docling의 제목 계층·표·코드·provenance를 수용할 수 있도록 문서 요소와 위치 모델을 갖췄지만, Docling 자체를 설치하거나 연동하지는 않았습니다. 향후 adapter가 `ParsedDocument`를 반환하도록 연결하며 DoclingDocument 타입을 답변·API·검색 계약에 노출하지 않습니다. parser별 세부 정보는 element metadata와 snapshot parser 설정에 남길 수 있습니다.

현재 구현은 사용한 원문 요소를 응답에 보존합니다. 원본 bytes를 영구 저장하는 artifact store, 전체 문서 조회 API, PDF 페이지 이미지 뷰어는 아직 없습니다. 장기 보관·다중 문서 탐색이 필요해지면 원본 보관 수명과 소유권을 포함해 확장해야 합니다. 현재 규모에서는 별도 그래프 DB, 문장 주장 그래프, 범용 문서 편집 AST, parser 플러그인 프레임워크를 두지 않습니다.

## 4. 가장 어려웠던 문제: Latency와 Retrieval 품질

DocuMate에서 가장 까다로웠던 문제는 "더 빠른 응답"과 "더 충분한 근거"가 자주 반대 방향으로 움직인다는 점이었습니다. 관련 근거를 넉넉히 모으면 답변을 뒷받침할 가능성은 높아지지만, 검색 시간이 늘고 synthesis prompt가 무거워집니다. 반대로 속도만 보고 route나 context를 줄이면 필요한 근거를 놓쳐 tool recall과 최종 답변 품질이 흔들릴 수 있습니다. 근거 개수나 높은 검색 점수가 의미적 정확성을 보장하는 것은 아닙니다.

그래서 이 문제를 단순 최적화가 아니라, latency와 retrieval quality 사이의 균형을 계측 가능한 시스템 문제로 다시 정의했습니다. 전체 응답 시간을 하나의 숫자로 보지 않고 `summarize`, `planner`, `retrieval`, `pre_synthesis_validation`, `synthesis`, `post_synthesis_validation`, `action_postprocess` 단계로 나누어 latency trace를 남겼습니다. retrieval도 route별 latency와 status를 기록해 `docs`, `upload` 중 어느 경로가 병목인지, no result인지, timeout인지 debug payload와 benchmark output에서 바로 추적할 수 있게 했습니다.

독립 retrieval task는 `ThreadPoolExecutor`로 병렬 실행하고 결과는 planner 순서대로 다시 정렬합니다. 이 실행 단위는 서로 다른 route뿐 아니라 같은 docs route의 복수 대상도 포함합니다. 외부 검색은 각 Tavily 요청마다 `DOCS_SEARCH_TIMEOUT_SECONDS`를 적용하고 timeout 원인을 diagnostics에 남깁니다.

planner와 synthesis는 구조화 모델 호출 경로를 사용하며 provider의 요청별 timeout과 SDK retry를 호출 경계에 명시합니다. 요청별 timeout은 stage 전체 deadline이 아니므로 총 실행 시간은 더 길 수 있습니다. docs 도구는 최초 query와 같은 대상을 유지한 재구성 query를 최대 한 번씩 계획하고, `covered`를 확보하거나 해당 query를 이미 시도했다면 추가 호출을 생략합니다. 고정 fallback 목록을 순회하며 주제나 버전 제약을 약화하지 않습니다.

재시도는 requirement 단위로 비용을 제한합니다. 같은 docs route에서 NumPy 근거를 확보하고 pandas만 실패했다면 NumPy는 그대로 사용하고 pandas query만 재계획합니다. 같은 요구의 부분 후보도 다음 후보와 함께 원래 라이브러리·심볼·버전·aspect에 맞는지 재평가합니다. 완료된 동일 요청은 fingerprint로 재사용하고 실제 시도 query도 추적합니다. 응답의 참조·구성 문제는 검색을 다시 구매하지 않고 제한된 재합성으로 처리합니다.

docs 품질 검사는 공식 domain/path prefix와 URL·원문 유효성에 더해 대상 문서 소유권, 요청 버전, 실제 발췌의 aspect를 확인합니다. 단순한 이름 언급과 대상 API의 문서를 구분하고, 부분 근거를 확보한 상태와 요구 전체를 충족한 상태를 구분합니다. 이 검사는 모든 설명의 의미적 충분성을 보증하지 않으므로 자연어 평가와 함께 해석합니다.

upload의 일반 검색은 vector 후보를 identifier·keyword·parameter signal로 rerank합니다. 정확한 심볼 요청은 전체 source registry의 AST 조회로 바꾸어 top-k 후보 밖의 정의도 찾습니다. 사용과 정의를 구분하고, 정의의 decorator·본문·Notebook cell 및 정확한 원문 offset을 유지합니다. source 전체를 확인할 수 없으면 부재를 단정하지 않고 `unknown`으로 남깁니다. 이 경로는 임의의 유사도 임계값 없이 이름이 없는 함수의 오발췌를 막으며, source에 명시된 정적 코드 범위만 확인한다는 한계를 갖습니다.

synthesis 근거 수는 planner 최대 독립 요구 수와 같은 상수를 사용해 일반·hybrid 모두 최대 8건을 허용합니다. 총 excerpt 예산은 일반 6,000자·hybrid 8,000자를 유지하며, 개별 범위는 설정된 snippet 상한을 따릅니다. 요구별 대표 후보를 먼저 배분하고 같은 source의 떨어진 aspect는 별도 passage로 선택합니다. 긴 문서·코드의 앞부분을 일괄 자르지 않고 관련 문단·문장·코드 행을 원문 offset으로 전달하며, 예산으로 빠진 aspect와 부분 선택을 모델과 최종 검증에 남깁니다. 저장·전송 요청은 근거 예산을 줄이지 않습니다. structured synthesis timeout 시 compact 호출을 사용하고, 실패하면 불완전함을 명시한 원문 발췌 fallback을 제공합니다. 각 경로의 검사 기준은 해당 호출에 실제 제공한 packet입니다.

출력 한도는 LLM registry 생성부에서만 설정하고 synthesis 입력 프로필에서는 재지정하지 않습니다. 일반 출력과 snippet의 초기 조정값은 각각 4,096토큰과 1,800자이며, compact는 독립된 960토큰·900자 설정과 축소된 총 excerpt 예산을 사용합니다. 두 설정의 변경은 HTTP 요청·원문 범위에서 따로 검증하며 실제 답변 품질·지연·비용은 별도 온라인 평가가 필요합니다. 이 분리는 일반 응답의 생성 여유를 늘릴 때 복구 비용이나 저장 메모리까지 함께 늘어나는 것을 피하기 위한 것입니다.

최종적으로 이 문제의 성공 기준은 "빠르다" 하나가 아니었습니다. release pass rate, tool precision, tool recall, citation compliance, p95 latency, 평균 cost를 함께 보며 변경을 평가했습니다. latency를 줄이는 변경이 근거 품질을 훼손하지 않는지, retrieval 필터링을 강화한 변경이 recall을 떨어뜨리지 않는지 benchmark로 확인하는 흐름을 만든 것이 이 프로젝트에서 가장 중요한 엔지니어링 판단이었습니다.

## 5. 구현 기준

### 실행 경로를 기준으로 문서화

문서는 의도한 구조가 아니라 실제 동작하는 코드 기준으로 작성합니다. README에서도 주요 기준 경로를 `src/runtime/graph_builder.py`, `src/runtime/make_graph.py`, `src/infra/tools/*`, `src/runtime/nodes/*`, `src/app/web/*`, `src/eval/*`로 제시합니다.

현재 코드 구조는 앱 진입점(`src/app`), 도메인 계약(`src/core`), 인프라 도구(`src/infra`), LangGraph 런타임(`src/runtime`), 평가 파이프라인(`src/eval`)을 분리합니다. 이 기준 덕분에 웹 요청 처리, route 판단, 외부 도구 호출, benchmark scoring이 서로 다른 책임 경계 안에서 유지됩니다.

### 사용자 요청과 시스템 진단을 분리

사용자에게는 간결한 답변을 제공하되, `include_debug=true`에서는 latency, planner/retrieval diagnostics, retry context, LLM call metadata를 확인할 수 있게 했습니다. 일반 응답 품질과 개발자 관측성을 같은 메시지에 섞지 않기 위한 기준입니다.

현재 debug schema version은 `6`입니다. debug payload에는 tool call, token usage, model usage status, validation events, edge decisions, `observed_hits`, action results, stage별 latency, retrieval route latency, synthesis attempt mode가 포함됩니다. `observed_hits`는 검색 과정에서 본 자료이며, 사용자 응답의 `citations`는 실제 표시 내용이 채택한 원문입니다. 이 정보는 일반 사용자 답변이 아니라 회귀 분석과 benchmark 해석을 위한 진단 계층입니다.

대화 compaction은 `edge_decisions`에 trigger 차원, before/after turn·message·추정 token·직렬화 byte, removed message 수, fallback 여부를 남깁니다. fallback은 `validation_events`에도 degraded 신호로 기록합니다. 이 진단과 구조화 로그에는 원문 query, summary, ToolMessage content를 포함하지 않습니다.

### 세션 단위 격리

업로드 파일 검색과 대화 상태는 세션 단위로 다룹니다. 세션별 manager cache, TTL/LRU 정리, 요청 lock을 두어 한 사용자의 업로드나 실행 상태가 다른 흐름과 섞이지 않게 관리합니다. close, exit, TTL/LRU eviction은 messages와 summary를 함께 제거합니다. 현재 store는 process-local in-memory 구현이므로 서버 재시작이나 여러 worker 사이에서 대화 상태를 복원하지는 않습니다.

업로드 파일은 `uploads/<session_id>/...` 아래의 `.py` 또는 `.ipynb`만 허용합니다. 세션 디렉터리 밖 경로는 `validate_upload_file_path()`에서 차단하고, 다운로드도 `output/save_text` 아래 상대 경로만 허용합니다. 업로드 retriever는 세션별 Chroma collection으로 만들고, 세션 종료나 파일 교체 시 cleanup합니다.

### 검증 가능한 결과를 우선

기능 추가 자체보다 release gate를 통과하는 재현 가능한 상태를 우선합니다. benchmark CLI와 `uv run pytest -q` 결과를 문서화해, 프로젝트가 어느 기준에서 정상 동작하는지 확인할 수 있게 했습니다.

평가 파이프라인은 실제 FastAPI `POST /agent/stream`을 호출하고 최종 응답 수신까지의 latency를 측정하는 online benchmark를 기준으로 합니다. HTTP 오류, SSE 오류, 연결 단절, 최종 응답 누락을 구분하고, 수신한 최종 응답의 debug도 평가에 유지합니다. `docs_only`, `rag_only`, `hybrid`, `tool_action` category를 나누고, rule 기반 지표와 LLM judge를 함께 사용합니다. `rag_only`는 fixture의 분류명이며 현재 업로드 검색을 평가합니다. deterministic `reference_coverage`는 표시한 내용의 참조가 실제 검색 결과에 연결되는지를 측정하고, 설명의 의미적 지지는 judge가 평가합니다. `not_evaluated`를 근거가 없는 답변의 점수로 취급하지 않습니다. hard gate는 `data/benchmarks/config.toml`에서 관리하며 자세한 지표는 [벤치마크 가이드](benchmarking.md)에 정리했습니다.

## 6. 개선 방향

DocuMate의 다음 개선 방향은 더 많은 기능을 붙이는 것보다, 현재 구조의 품질 신호를 더 정교하게 만드는 쪽입니다.

- judge minimum score audit에서 기준을 넘지 못한 docs/hybrid 케이스를 분석해 답변 품질 개선 후보로 관리합니다.
- retrieval route별 warning, error code, latency breakdown을 더 쉽게 비교할 수 있게 report를 정리합니다.
- 실제 문서 표본으로 Docling adapter를 검증하고, 추출한 제목 계층·표 병합·여러 페이지 위치가 현재 문서 모델과 정확히 연결되는지 확인합니다.
- 원본 보관 수명·소유권과 문서 조회 계약을 정의한 뒤 PDF 페이지 뷰어와 위치 강조를 추가합니다.
- 설명의 의미적 지지 검사를 런타임에 추가할 경우 검사 비용·범위·실패 상태를 명시하고, 인용 연결 확인과 분리해 평가합니다.
- upload retriever build와 synthesis fallback의 비용/지연을 benchmark summary에서 더 세밀하게 분리합니다.
- benchmark fixture를 주기적으로 보강해 공식 문서 검색, 업로드 검색, tool action 흐름의 회귀 범위를 넓힙니다.
- rolling summary의 사실 보존율을 장기 대화 전용 eval fixture로 계측하고, 모델별 tokenizer를 알 수 있을 때 현재 보수적 추정기를 교정합니다.
- 인증·소유권과 암호화를 포함한 외부 session store가 필요해지면 process restart와 multi-worker를 지원하는 별도 persistence 계층을 도입합니다.
- Streamlit의 새 대화 동작이 이전 backend session을 TTL까지 남겨 두지 않고 즉시 폐기하도록 reset API의 동시성·멱등성 계약을 설계합니다.
