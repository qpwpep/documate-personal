# Design Rationale

DocuMate의 설계 판단과 기술적 선택

## 1. 문서 목적

이 문서는 DocuMate를 원본 팀 프로젝트에서 단계형 LangGraph 런타임과 120-case benchmark 체계를 갖춘 포트폴리오 개선본으로 재설계하면서, 어떤 기준으로 구조를 바꾸고 기능을 확장했는지 설명합니다. 단순히 기능 목록을 나열하기보다, 실행 경로를 안정적으로 만들기 위해 어떤 문제를 분리했고 어떤 트레이드오프를 받아들였는지 기록하는 데 목적이 있습니다.

DocuMate는 LangGraph 기반 학습 보조 에이전트입니다. 현재 구조는 공식 문서 검색, 세션 업로드 파일 검색, 구조화된 응답, 저장 및 Slack 전송 액션을 FastAPI와 Streamlit 런타임 위에서 함께 제공합니다. 실행 기준 코드는 `src/app`, `src/core`, `src/infra`, `src/runtime`, `src/eval` 계층으로 분리되어 있고, `archive`는 현재 실행 경로가 아니라 팀 프로젝트 원형과 참고 자료를 보관하는 영역입니다.

## 2. 핵심 설계 판단

### 단계형 LangGraph 파이프라인

초기 구조처럼 모델의 tool call 흐름에만 실행을 맡기면 검색, 검증, 액션의 책임 경계가 흐려지기 쉽습니다. 그래서 현재 그래프는 `add_user_message`, `summarize_old_messages`, `planner`, `retrieve_dispatch`, `pre_synthesis_validation`, `synthesize`, `post_synthesis_validation`, `action_postprocess` 단계로 나누었습니다.

이 구조의 목표는 각 단계가 명확한 상태 계약을 주고받게 만드는 것입니다. `GraphState`는 `runtime`, `planner`, `retrieval`, `retry`, `response`, `debug` 영역으로 나뉘며, boundary adapter가 dict와 Pydantic 모델 사이의 상태를 정규화합니다. planner는 검색 필요 여부와 route를 결정하고, retrieval은 route별 evidence와 diagnostics를 모으며, validation은 근거 품질과 route coverage를 확인하고, synthesis는 최종 답변을 구조화합니다.

### 유한한 장기 대화 메모리

대화 메모리는 `bounded rolling summary + bounded recent canonical messages`라는 두 부분으로 나눴습니다. `ConversationMemoryPolicy`가 Human turn, 추정 token, UTF-8 직렬화 byte, message 수의 high/low watermark와 최종 hard byte limit을 한곳에서 관리합니다. 어느 high watermark든 도달하면 가장 오래된 완결 turn부터 퇴출해 모든 low watermark를 만족하는 가장 긴 최근 suffix를 남깁니다. 정확한 모델 tokenizer가 없어도 실제 크기 상한이 유지되도록 token 값은 보수적인 추정치로 사용하고, UTF-8 직렬화 byte를 독립적인 backstop으로 검사합니다.

`GraphState.messages`는 LangGraph의 `add_messages` reducer를 사용하므로 최근 리스트만 반환해서는 누락된 과거 메시지가 삭제되지 않습니다. 요약 노드는 `RemoveMessage(id=REMOVE_ALL_MESSAGES)`와 retained suffix를 함께 반환해 reducer에 전체 교체 의도를 명시합니다. router와 요약 노드는 같은 pure compaction plan을 사용하므로 trigger 판단과 실제 퇴출 범위가 어긋나지 않습니다.

summary도 별도로 bounded합니다. 새 summary는 `기존 bounded summary + 이번에 새로 퇴출된 대화`를 하나의 replacement memory로 다시 작성한 결과이며, 이전 summary 뒤에 새 문자열을 append하지 않습니다. Tool/System payload는 요약 입력에서 제외합니다. summarizer 예외나 빈 출력에는 기존 summary와 새 transcript의 head/tail을 함께 보존하는 deterministic fallback을 적용하고, 결과를 token·byte 상한에 다시 맞춥니다. fallback 사용 여부와 before/after 크기는 기록하지만 대화 내용은 로그에 남기지 않습니다.

planner와 synthesis에 전달되는 summary는 과거 사용자 입력에서 유래한 비신뢰 데이터입니다. 고정 System policy가 그 안의 명령을 따르거나 검색 evidence로 취급하지 말라고 명시하고, 실제 summary payload는 system instruction이 아닌 별도 assistant data message로 전달합니다.

### 검색 route 분리

검색 소스는 `docs`, `upload` route로 분리했습니다. 공식 문서 검색과 세션 업로드 파일 검색은 데이터 출처와 신뢰 기준이 다르기 때문입니다. 필요한 출처는 업로드 가용성과 무관하게 LLM이 판단하고, 스키마 검증을 통과한 `PlannerOutput.tasks`를 기준으로 실행합니다. 검색어 후처리는 공백만 정규화해 주제와 식별자를 언어에 관계없이 보존합니다.

`docs` route는 Tavily 검색을 사용하되 [agent_rules.toml](../src/infra/config/agent_rules.toml)의 allowlist, query hint, domain/path prefix, URL 검증, topic purity, exact identifier coverage를 통과한 결과만 evidence로 사용합니다. `upload` route는 현재 세션에 업로드된 `.py` 또는 `.ipynb` 파일에서 만든 임시 Chroma retriever만 사용합니다. 파일 기반 근거의 범위를 사용자가 해당 세션에 제공한 파일로 한정하는 것이 이 경계의 기준입니다.

route를 분리하면 응답 단계에서 evidence의 출처를 더 명확히 다룰 수 있고, 특정 소스가 실패해도 전체 흐름을 바로 중단하지 않고 다른 route 결과를 활용할 수 있습니다. 현재 `retrieve_dispatch`는 여러 route task가 필요한 경우 `ThreadPoolExecutor`로 병렬 실행하고, 결과는 planner task 순서대로 다시 정렬합니다.

### Grounded 응답 스키마

최종 응답은 자연어 하나로만 끝내지 않고 `answer`, `claims`, `evidence`, `confidence`, `sections`를 포함하는 구조화된 페이로드로 반환하도록 설계했습니다. 현재 기준 모델은 `AgentResponsePayloadModel`이며, API 응답에서는 `AgentResponsePayload`로 노출됩니다.

이 선택은 답변 품질을 사람이 읽는 느낌에만 맡기지 않기 위한 것입니다. claim과 evidence를 함께 유지하면, 이후 검증 로직과 벤치마크에서 답변이 실제 검색 결과에 근거하는지 확인할 수 있습니다. 저장이나 Slack 전송처럼 액션이 붙는 경우에도 `ResponseAssembler`가 최종 답변과 receipt를 같은 payload 흐름 안에서 정리합니다.

### 검증과 선택적 재시도

검색 결과가 있더라도 최종 답변이 항상 충분히 grounded하다고 볼 수는 없습니다. 그래서 synthesis 전후에 validation 단계를 두고, evidence 품질이나 unsupported claim이 문제가 될 때 planner로 되돌아가 재검색할 수 있게 했습니다.

이 흐름은 모든 실패를 무조건 재시도하지 않습니다. `RetryState`는 failed route, preserved evidence, preserved retrieval diagnostics, retry scope를 보존합니다. 그래서 일부 route만 실패한 경우에는 성공한 route의 evidence를 재사용하고 실패 route만 다시 호출합니다. unsupported claim이나 section 누락처럼 검색 실패보다 response repair에 가까운 문제는 기존 evidence를 기준으로 payload를 결정적으로 보정합니다.

LLM 호출이나 출력 검증이 실패하면 `planner_unavailable`로 기록하고 재요청을 안내합니다. 필요한 업로드 파일의 retriever가 없으면 파일 업로드를 안내합니다. 두 경우 모두 검색·저장·Slack 전송을 중단해, 실패 안내나 이전 답변이 요청한 결과물로 전달되지 않도록 합니다.

### FastAPI + Streamlit 런타임 분리

FastAPI는 실제 API 실행과 세션 관리를 담당하고, Streamlit은 사용자가 흐름을 확인하는 인터페이스 역할을 합니다. 웹 요청은 `AgentRequestService`에서 시작해 `InMemorySessionStore`와 세션별 `AgentFlowManager`를 거쳐 그래프를 실행합니다. `AgentFlowManager`는 `ExecutionRunner`, `DebugCollector`, `ResponseAssembler`, `SessionContext`를 묶는 facade 역할을 합니다.

세션별 manager cache, TTL/LRU 기반 정리, 요청 lock, SSE progress, 업로드/생성 파일 cleanup을 포함해 데모 UI와 실제 실행 경로가 같은 런타임을 바라보게 했습니다. `/agent/stream`은 `ProgressEmitter`로 request, stage, progress snapshot, final response, error, done 이벤트를 내보냅니다.

`SessionContext`는 최근 messages와 `memory_summary`를 하나의 immutable conversation snapshot으로 소유합니다. graph가 반환한 전체 메시지는 먼저 debug와 response assembly가 사용합니다. 사용자가 볼 응답까지 정상적으로 조립된 뒤에만 Tool/System/중간 AI를 제거하고 각 Human turn과 canonical final AI를 남겨 summary와 함께 단일 참조 교체로 commit합니다. graph, debug, assembly, projection 중 하나라도 실패하면 이전 정상 snapshot은 그대로 유지됩니다. 이 원자성은 대화 메모리에 한정되며 이미 실행된 파일 저장·Slack 전송 같은 외부 side effect까지 rollback하지는 않습니다.

이 구조는 포트폴리오 데모와 백엔드 검증을 분리하지 않기 위한 선택입니다. 화면에서 보이는 동작이 테스트 및 benchmark 대상인 `POST /agent` 흐름과 이어져 있어야 유지보수 기준이 단순해집니다.

## 3. 주요 트레이드오프

### 단순한 tool agent보다 명시적 graph를 선택

단순한 tool agent는 구현이 빠르고 코드가 짧습니다. 대신 route 선택, evidence 정규화, 재시도 조건, 액션 후처리 같은 정책이 프롬프트와 런타임 곳곳에 흩어질 수 있습니다.

DocuMate는 포트폴리오 프로젝트이지만, 검색 품질과 근거 검증을 핵심 역량으로 보여주는 것이 중요했습니다. 그래서 구현량이 늘어나더라도 단계별 graph와 node 책임을 명시하는 방향을 선택했습니다. 일반 기술 설명과 사용자 파일 조회에는 같은 단어가 등장하므로, 출처의 지정·제외와 문맥 해석은 planner LLM에 맡깁니다. 후처리는 스키마와 실행 가용성을 확인하며 정규식으로 출처를 추가하거나 제거하지 않습니다. 의미 판단은 모델에 의존하지만, 계획 실패 시 명확한 안내로 종료해 다른 판별 규칙이 출처를 추측하며 실행을 이어 가지 않도록 했습니다.

### 여러 검색 소스를 하나로 합치지 않음

공식 문서와 업로드 파일을 하나의 retriever처럼 다루면 인터페이스는 단순해집니다. 하지만 답변이 어떤 근거를 사용했는지 설명하기 어렵고, 실패 원인을 route별로 추적하기도 어렵습니다.

현재 구조는 route별 처리 비용이 조금 더 들지만, evidence 출처와 진단 정보를 명확히 남기는 쪽을 우선했습니다. `RetrievalDiagnostic`에는 route status, error code, provider time, URL validation time, filtering count, warning이 남기 때문에 benchmark와 debug payload에서 실패 원인을 좁히기 쉽습니다.

### 원문 전체 보존보다 bounded rolling memory를 선택

요약은 본질적으로 손실 압축이므로 오래된 원문을 전부 보존하는 것과 같은 의미 충실도를 보장하지 않습니다. 대신 process-local session memory와 다음 prompt 크기, 요약 호출 비용에 명시적인 상한을 둘 수 있습니다. high/low watermark를 사용해 한 번의 compaction에서 target까지 내리므로 window가 찬 뒤 매 요청마다 한 turn만 요약하는 진동도 줄였습니다.

ToolMessage 원문과 provider metadata를 durable snapshot에 저장하지 않는 선택 역시 같은 트레이드오프입니다. 현재 요청의 response assembly와 debug에는 전체 payload를 사용하지만 다음 요청에는 사용자 질문과 실제 표시된 assistant 답변만 넘깁니다. 완전한 event replay 가능성은 줄어드는 대신 검색 원문·파일 내용·tool receipt가 장기 대화에 반복 주입되는 비용과 개인정보 노출 면적을 줄입니다.

### 테스트와 benchmark에 운영 비용을 투자

개인 프로젝트에서 120-case release benchmark와 pytest 기반 회귀 테스트를 유지하는 것은 비용이 있습니다. fixture 관리, judge 설정, latency 및 비용 지표 확인이 필요하기 때문입니다.

대신 변경 후 품질을 감으로 판단하지 않아도 됩니다. 현재 문서화된 최신 release benchmark는 `20260509_043436` 런 기준 120개 중 116개 케이스 통과, release pass rate `0.9667`, tool precision `0.9677`, tool recall `1.0000`, citation compliance `0.9556`, p95 latency `9435.9 ms`, 평균 cost `$0.00523362`를 기록했고, 테스트는 `429 passed, 56 subtests passed`로 검증되었습니다.

## 4. 가장 어려웠던 문제: Latency와 Retrieval 품질

DocuMate에서 가장 까다로웠던 문제는 "더 빠른 응답"과 "더 믿을 수 있는 근거"가 자주 반대 방향으로 움직인다는 점이었습니다. evidence를 넉넉히 모으면 citation compliance와 답변 신뢰도는 좋아지지만, 검색 시간이 늘고 synthesis prompt가 무거워집니다. 반대로 속도만 보고 route나 context를 줄이면 필요한 근거를 놓쳐 tool recall과 최종 답변 품질이 흔들릴 수 있습니다.

그래서 이 문제를 단순 최적화가 아니라, latency와 retrieval quality 사이의 균형을 계측 가능한 시스템 문제로 다시 정의했습니다. 전체 응답 시간을 하나의 숫자로 보지 않고 `summarize`, `planner`, `retrieval`, `pre_synthesis_validation`, `synthesis`, `post_synthesis_validation`, `action_postprocess` 단계로 나누어 latency trace를 남겼습니다. retrieval도 route별 latency와 status를 기록해 `docs`, `upload` 중 어느 경로가 병목인지, no result인지, timeout인지 debug payload와 benchmark output에서 바로 추적할 수 있게 했습니다.

응답 속도 개선은 "덜 찾기"보다 "필요한 것을 동시에, 제한 시간 안에서 찾기"에 가깝게 접근했습니다. hybrid 질문에서 여러 retrieval task가 필요할 때는 `ThreadPoolExecutor`로 route fan-out을 병렬 실행하고, 결과는 planner task 순서대로 다시 정렬합니다. 외부 검색인 docs route에는 개별 Tavily 요청마다 `DOCS_SEARCH_TIMEOUT_SECONDS`를 적용하고, timeout은 `RETRIEVAL_DOCS_TIMEOUT` error code와 diagnostics로 남겨 원인 분석이 가능하게 했습니다.

planner와 synthesis는 각각 하나의 구조화 모델 호출 경로를 사용하며 provider의 요청별 timeout과 순차 SDK retry 정책을 호출 경계에 명시합니다. 요청별 timeout은 stage 전체 deadline이 아니므로, 재시도를 허용한 호출의 총 실행 시간은 해당 timeout보다 길 수 있습니다. docs search도 query 하나당 Tavily 요청을 한 번만 보내고, 첫 결과의 근거 품질이나 identifier coverage가 부족할 때만 query hint의 fallback을 정의된 순서대로 실행합니다. 충분한 evidence를 확보하면 남은 fallback은 실행하지 않습니다.

재시도 전략도 latency 관점에서 다시 설계했습니다. validation 실패 후 모든 route를 매번 다시 호출하면 품질을 올리려는 시도가 곧바로 비용과 지연으로 이어집니다. 그래서 retry context에 failed route, preserved evidence, preserved retrieval diagnostics를 보존하고, 실패하지 않은 route의 evidence는 재사용합니다. 예를 들어 `docs + upload` hybrid 흐름에서 docs만 실패하면 upload evidence는 유지하고 docs route만 다시 시도합니다. unsupported claim이나 section 누락처럼 검색 실패가 아니라 response repair에 가까운 문제는 기존 evidence를 기준으로 claim을 필터링하고 답변이나 section을 결정적으로 보정합니다.

retrieval 품질은 "높은 score의 결과를 많이 가져오기"가 아니라 "답변에 실제로 쓸 수 있는 근거만 남기기"로 정의했습니다. docs route는 공식 문서 domain/path prefix를 통과한 결과만 evidence로 사용하고, query hint와 fallback query로 라이브러리별 검색 범위를 좁힙니다. 이후 topic purity, exact identifier coverage, chrome-only page 여부를 확인해 근거로 쓰기 어려운 결과를 제거합니다.

upload route에는 vector score에 lexical signal을 결합했습니다. query의 identifier, keyword, parameter hint를 기준으로 검색 결과를 rerank하고, 긴 chunk는 질문 토큰이 실제로 등장하는 주변 window로 압축합니다. 코드 추출처럼 원문 보존이 중요한 질문은 예외로 처리해, prompt budget을 줄이면서도 사용자가 찾는 코드 맥락은 잃지 않게 했습니다.

synthesis 단계에서는 category별 prompt budget을 적용했습니다. `docs_only`, `upload_only`, `hybrid`, `tool_action`에 따라 evidence 개수, snippet 길이, 출력 token 상한을 다르게 두었습니다. hybrid 답변은 source coverage가 핵심이므로 docs와 upload evidence를 균형 있게 남기고, 단일 route나 action 중심 요청은 더 작은 budget으로 불필요한 context를 줄였습니다. structured synthesis가 timeout되면 compact structured fallback 또는 deterministic grounded fallback으로 내려가도록 해, 빈 응답이나 과도한 실패 전파를 줄였습니다.

최종적으로 이 문제의 성공 기준은 "빠르다" 하나가 아니었습니다. release pass rate, tool precision, tool recall, citation compliance, p95 latency, 평균 cost를 함께 보며 변경을 평가했습니다. latency를 줄이는 변경이 근거 품질을 훼손하지 않는지, retrieval 필터링을 강화한 변경이 recall을 떨어뜨리지 않는지 benchmark로 확인하는 흐름을 만든 것이 이 프로젝트에서 가장 중요한 엔지니어링 판단이었습니다.

## 5. 구현 기준

### 실행 경로를 기준으로 문서화

문서는 의도한 구조가 아니라 실제 동작하는 코드 기준으로 작성합니다. README에서도 주요 기준 경로를 `src/runtime/graph_builder.py`, `src/runtime/make_graph.py`, `src/infra/tools/*`, `src/runtime/nodes/*`, `src/app/web/*`, `src/eval/*`로 제시합니다.

현재 코드 구조는 앱 진입점(`src/app`), 도메인 계약(`src/core`), 인프라 도구(`src/infra`), LangGraph 런타임(`src/runtime`), 평가 파이프라인(`src/eval`)을 분리합니다. 이 기준 덕분에 웹 요청 처리, route 판단, 외부 도구 호출, benchmark scoring이 서로 다른 책임 경계 안에서 유지됩니다.

### 사용자 요청과 시스템 진단을 분리

사용자에게는 간결한 답변을 제공하되, `include_debug=true`에서는 latency, planner/retrieval diagnostics, retry context, LLM call metadata를 확인할 수 있게 했습니다. 일반 응답 품질과 개발자 관측성을 같은 메시지에 섞지 않기 위한 기준입니다.

현재 debug schema version은 `5`입니다. debug payload에는 tool call, token usage, model usage status, validation events, edge decisions, observed evidence, action results, stage별 latency, retrieval route latency, synthesis attempt mode가 포함됩니다. 이 정보는 일반 사용자 답변이 아니라 회귀 분석과 benchmark 해석을 위한 진단 계층입니다.

대화 compaction은 `edge_decisions`에 trigger 차원, before/after turn·message·추정 token·직렬화 byte, removed message 수, fallback 여부를 남깁니다. fallback은 `validation_events`에도 degraded 신호로 기록합니다. 이 진단과 구조화 로그에는 원문 query, summary, ToolMessage content를 포함하지 않습니다.

### 세션 단위 격리

업로드 파일 검색과 대화 상태는 세션 단위로 다룹니다. 세션별 manager cache, TTL/LRU 정리, 요청 lock을 두어 한 사용자의 업로드나 실행 상태가 다른 흐름과 섞이지 않게 관리합니다. close, exit, TTL/LRU eviction은 messages와 summary를 함께 제거합니다. 현재 store는 process-local in-memory 구현이므로 서버 재시작이나 여러 worker 사이에서 대화 상태를 복원하지는 않습니다.

업로드 파일은 `uploads/<session_id>/...` 아래의 `.py` 또는 `.ipynb`만 허용합니다. 세션 디렉터리 밖 경로는 `validate_upload_file_path()`에서 차단하고, 다운로드도 `output/save_text` 아래 상대 경로만 허용합니다. 업로드 retriever는 세션별 Chroma collection으로 만들고, 세션 종료나 파일 교체 시 cleanup합니다.

### 검증 가능한 결과를 우선

기능 추가 자체보다 release gate를 통과하는 재현 가능한 상태를 우선합니다. benchmark CLI와 `uv run pytest -q` 결과를 문서화해, 프로젝트가 어느 기준에서 정상 동작하는지 확인할 수 있게 했습니다.

평가 파이프라인은 실제 FastAPI `POST /agent`를 호출하는 online benchmark를 기준으로 합니다. `docs_only`, `rag_only`, `hybrid`, `tool_action` category를 나누고, rule 기반 지표와 LLM judge를 함께 사용합니다. 평가의 `rag_only`는 기존 fixture와 결과를 읽기 위해 유지하는 분류명이며, 현재 fixture에서는 업로드 검색을 평가합니다. 검색 route 및 인용 유형과의 구분은 [벤치마크 가이드](benchmarking.md)에 정리했습니다. hard gate는 `data/benchmarks/config.toml`에서 관리하며, report와 history 산출물은 `src/eval`에서 생성합니다.

## 6. 개선 방향

DocuMate의 다음 개선 방향은 더 많은 기능을 붙이는 것보다, 현재 구조의 품질 신호를 더 정교하게 만드는 쪽입니다.

- judge minimum score audit에서 기준을 넘지 못한 docs/hybrid 케이스를 분석해 답변 품질 개선 후보로 관리합니다.
- retrieval route별 warning, error code, latency breakdown을 더 쉽게 비교할 수 있게 report를 정리합니다.
- Streamlit 데모에서 evidence와 claim의 관계를 더 직관적으로 확인할 수 있는 표시 방식을 개선합니다.
- upload retriever build와 synthesis fallback의 비용/지연을 benchmark summary에서 더 세밀하게 분리합니다.
- benchmark fixture를 주기적으로 보강해 공식 문서 검색, 업로드 검색, tool action 흐름의 회귀 범위를 넓힙니다.
- rolling summary의 사실 보존율을 장기 대화 전용 eval fixture로 계측하고, 모델별 tokenizer를 알 수 있을 때 현재 보수적 추정기를 교정합니다.
- 인증·소유권과 암호화를 포함한 외부 session store가 필요해지면 process restart와 multi-worker를 지원하는 별도 persistence 계층을 도입합니다.
- Streamlit의 새 대화 동작이 이전 backend session을 TTL까지 남겨 두지 않고 즉시 폐기하도록 reset API의 동시성·멱등성 계약을 설계합니다.
