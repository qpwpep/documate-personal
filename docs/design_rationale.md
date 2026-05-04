# Design Rationale

DocuMate의 설계 판단과 기술적 선택

## 1. 문서 목적

이 문서는 DocuMate를 개인 포트폴리오 프로젝트로 정리하면서 어떤 기준으로 구조를 바꾸고 기능을 확장했는지 설명합니다. 단순히 기능 목록을 나열하기보다, 실행 경로를 안정적으로 만들기 위해 어떤 문제를 분리했고 어떤 트레이드오프를 받아들였는지 기록하는 데 목적이 있습니다.

DocuMate는 LangGraph 기반 학습 보조 에이전트입니다. 현재 구조는 공식 문서 검색, 로컬 노트북 RAG, 세션 업로드 파일 검색, 구조화된 응답, 저장 및 Slack 전송 액션을 FastAPI와 Streamlit 런타임 위에서 함께 제공합니다.

## 2. 핵심 설계 판단

### 단계형 LangGraph 파이프라인

초기 구조처럼 모델의 tool call 흐름에만 실행을 맡기면 검색, 검증, 액션의 책임 경계가 흐려지기 쉽습니다. 그래서 현재 그래프는 `planner`, `retrieval`, `validation`, `synthesis`, `action` 단계로 나누었습니다.

이 구조의 목표는 각 단계가 명확한 상태 계약을 주고받게 만드는 것입니다. planner는 검색 필요 여부와 route를 결정하고, retrieval은 route별 evidence를 모으며, validation은 근거 품질을 확인하고, synthesis는 최종 답변을 구조화합니다.

### 검색 route 분리

검색 소스는 `docs`, `local`, `upload` route로 분리했습니다. 공식 문서 검색, 로컬 노트북 RAG, 세션 업로드 파일 검색은 데이터 출처와 신뢰 기준이 다르기 때문입니다.

route를 분리하면 응답 단계에서 evidence의 출처를 더 명확히 다룰 수 있고, 특정 소스가 실패해도 전체 흐름을 바로 중단하지 않고 다른 route 결과를 활용할 수 있습니다.

### Grounded 응답 스키마

최종 응답은 자연어 하나로만 끝내지 않고 `answer`, `claims`, `evidence`, `confidence`, `sections`를 포함하는 구조화된 페이로드로 반환하도록 설계했습니다.

이 선택은 답변 품질을 사람이 읽는 느낌에만 맡기지 않기 위한 것입니다. claim과 evidence를 함께 유지하면, 이후 검증 로직과 벤치마크에서 답변이 실제 검색 결과에 근거하는지 확인할 수 있습니다.

### 검증과 선택적 재시도

검색 결과가 있더라도 최종 답변이 항상 충분히 grounded하다고 볼 수는 없습니다. 그래서 synthesis 전후에 validation 단계를 두고, evidence 품질이나 unsupported claim이 문제가 될 때 planner로 되돌아가 재검색할 수 있게 했습니다.

이 흐름은 모든 실패를 무조건 재시도하지 않습니다. 사용자의 의도가 불명확하거나 근거가 부족한 경우에는 후속 질문으로 전환할 수 있도록 두어, 잘못된 확신을 가진 답변을 줄이는 방향을 선택했습니다.

### FastAPI + Streamlit 런타임 분리

FastAPI는 실제 API 실행과 세션 관리를 담당하고, Streamlit은 사용자가 흐름을 확인하는 인터페이스 역할을 합니다. 세션별 `AgentFlowManager`, TTL/LRU 기반 정리, 요청 lock, SSE progress, 업로드/생성 파일 cleanup을 포함해 데모 UI와 실제 실행 경로가 같은 런타임을 바라보게 했습니다.

이 구조는 포트폴리오 데모와 백엔드 검증을 분리하지 않기 위한 선택입니다. 화면에서 보이는 동작이 테스트 및 benchmark 대상인 `POST /agent` 흐름과 이어져 있어야 유지보수 기준이 단순해집니다.

## 3. 주요 트레이드오프

### 단순한 tool agent보다 명시적 graph를 선택

단순한 tool agent는 구현이 빠르고 코드가 짧습니다. 대신 route 선택, evidence 정규화, 재시도 조건, 액션 후처리 같은 정책이 프롬프트와 런타임 곳곳에 흩어질 수 있습니다.

DocuMate는 포트폴리오 프로젝트이지만, 검색 품질과 근거 검증을 핵심 역량으로 보여주는 것이 중요했습니다. 그래서 구현량이 늘어나더라도 단계별 graph와 node 책임을 명시하는 방향을 선택했습니다.

### 여러 검색 소스를 하나로 합치지 않음

공식 문서, 로컬 RAG, 업로드 파일을 하나의 retriever처럼 다루면 인터페이스는 단순해집니다. 하지만 답변이 어떤 근거를 사용했는지 설명하기 어렵고, 실패 원인을 route별로 추적하기도 어렵습니다.

현재 구조는 route별 처리 비용이 조금 더 들지만, evidence 출처와 진단 정보를 명확히 남기는 쪽을 우선했습니다.

### 테스트와 benchmark에 운영 비용을 투자

개인 프로젝트에서 120-case release benchmark와 pytest 기반 회귀 테스트를 유지하는 것은 비용이 있습니다. fixture 관리, judge 설정, latency 및 비용 지표 확인이 필요하기 때문입니다.

대신 변경 후 품질을 감으로 판단하지 않아도 됩니다. 현재 문서화된 최신 release benchmark는 120개 케이스 통과, release pass rate `1.0000`, citation compliance `1.0000`, p95 latency `9354.0 ms`를 기록했고, 테스트는 `347 passed, 49 subtests passed`로 검증되었습니다.

## 4. 가장 어려웠던 문제: Latency와 Retrieval 품질

DocuMate에서 가장 까다로웠던 문제는 "더 빠른 응답"과 "더 믿을 수 있는 근거"가 자주 반대 방향으로 움직인다는 점이었습니다. evidence를 넉넉히 모으면 citation compliance와 답변 신뢰도는 좋아지지만, 검색 시간이 늘고 synthesis prompt가 무거워집니다. 반대로 속도만 보고 route나 context를 줄이면 필요한 근거를 놓쳐 tool recall과 최종 답변 품질이 흔들릴 수 있습니다.

그래서 이 문제를 단순 최적화가 아니라, latency와 retrieval quality 사이의 균형을 계측 가능한 시스템 문제로 다시 정의했습니다. 전체 응답 시간을 하나의 숫자로 보지 않고 `planner`, `retrieval`, `pre_synthesis_validation`, `synthesis`, `post_synthesis_validation`, `action_postprocess` 단계로 나누어 latency trace를 남겼습니다. retrieval도 route별 latency와 status를 기록해 `docs`, `upload`, `local` 중 어느 경로가 병목인지, no result인지, timeout인지 debug payload와 benchmark output에서 바로 추적할 수 있게 했습니다.

응답 속도 개선은 "덜 찾기"보다 "필요한 것을 동시에, 제한 시간 안에서 찾기"에 가깝게 접근했습니다. hybrid 질문에서 여러 retrieval task가 필요할 때는 `ThreadPoolExecutor`로 route fan-out을 병렬 실행하고, 결과는 planner task 순서대로 다시 정렬합니다. 외부 검색인 docs route에는 `DOCS_SEARCH_TIMEOUT_SECONDS`를 두어 Tavily 호출이 전체 agent 응답을 과도하게 붙잡지 않도록 했고, timeout은 `RETRIEVAL_DOCS_TIMEOUT` error code와 diagnostics로 남겨 원인 분석이 가능하게 했습니다.

재시도 전략도 latency 관점에서 다시 설계했습니다. validation 실패 후 모든 route를 매번 다시 호출하면 품질을 올리려는 시도가 곧바로 비용과 지연으로 이어집니다. 그래서 retry context에 failed route, preserved evidence, preserved retrieval diagnostics를 보존하고, 실패하지 않은 route의 evidence는 재사용합니다. 예를 들어 `docs + upload` hybrid 흐름에서 docs만 실패하면 upload evidence는 유지하고 docs route만 다시 시도합니다. unsupported claim이나 section 누락처럼 검색 실패가 아니라 synthesis repair에 가까운 문제는 기존 evidence를 바탕으로 다시 합성하도록 분리했습니다.

retrieval 품질은 "높은 score의 결과를 많이 가져오기"가 아니라 "답변에 실제로 쓸 수 있는 근거만 남기기"로 정의했습니다. docs route는 공식 문서 domain/path prefix를 통과한 결과만 evidence로 사용하고, query hint와 fallback query로 라이브러리별 검색 범위를 좁힙니다. 이후 topic purity, exact identifier coverage, chrome-only page 여부를 확인해 근거로 쓰기 어려운 결과를 제거합니다.

local/upload route에는 vector score에 lexical signal을 결합했습니다. query의 identifier, keyword, parameter hint를 기준으로 검색 결과를 rerank하고, 긴 chunk는 질문 토큰이 실제로 등장하는 주변 window로 압축합니다. 코드 추출처럼 원문 보존이 중요한 질문은 예외로 처리해, prompt budget을 줄이면서도 사용자가 찾는 코드 맥락은 잃지 않게 했습니다.

synthesis 단계에서는 category별 prompt budget을 적용했습니다. `docs_only`, `rag_only`, `upload_only`, `hybrid`, `tool_action`에 따라 evidence 개수, snippet 길이, 출력 token 상한을 다르게 두었습니다. hybrid 답변은 source coverage가 핵심이므로 docs와 upload/local evidence를 균형 있게 남기고, 단일 route나 action 중심 요청은 더 작은 budget으로 불필요한 context를 줄였습니다.

최종적으로 이 문제의 성공 기준은 "빠르다" 하나가 아니었습니다. release pass rate, tool precision, tool recall, citation compliance, p95 latency, 평균 cost를 함께 보며 변경을 평가했습니다. latency를 줄이는 변경이 근거 품질을 훼손하지 않는지, retrieval 필터링을 강화한 변경이 recall을 떨어뜨리지 않는지 benchmark로 확인하는 흐름을 만든 것이 이 프로젝트에서 가장 중요한 엔지니어링 판단이었습니다.

## 5. 구현 기준

### 실행 경로를 기준으로 문서화

문서는 의도한 구조가 아니라 실제 동작하는 코드 기준으로 작성합니다. README에서도 주요 기준 경로를 `src/runtime/graph_builder.py`, `src/runtime/make_graph.py`, `src/infra/tools/*`, `src/runtime/nodes/*`, `src/app/web/*`, `src/eval/*`로 제시합니다.

### 사용자 요청과 시스템 진단을 분리

사용자에게는 간결한 답변을 제공하되, `include_debug=true`에서는 latency, planner/retrieval diagnostics, retry context, LLM call metadata를 확인할 수 있게 했습니다. 일반 응답 품질과 개발자 관측성을 같은 메시지에 섞지 않기 위한 기준입니다.

### 세션 단위 격리

업로드 파일 검색과 대화 상태는 세션 단위로 다룹니다. 세션별 manager cache, TTL/LRU 정리, 요청 lock을 두어 한 사용자의 업로드나 실행 상태가 다른 흐름과 섞이지 않게 관리합니다.

### 검증 가능한 결과를 우선

기능 추가 자체보다 release gate를 통과하는 재현 가능한 상태를 우선합니다. benchmark CLI와 `uv run pytest -q` 결과를 문서화해, 프로젝트가 어느 기준에서 정상 동작하는지 확인할 수 있게 했습니다.

## 6. 개선 방향

DocuMate의 다음 개선 방향은 더 많은 기능을 붙이는 것보다, 현재 구조의 품질 신호를 더 정교하게 만드는 쪽입니다.

- judge minimum score audit에서 기준을 넘지 못한 케이스를 분석해 답변 품질 개선 후보로 관리합니다.
- retrieval route별 실패 원인과 latency breakdown을 더 쉽게 비교할 수 있게 정리합니다.
- Streamlit 데모에서 evidence와 claim의 관계를 더 직관적으로 확인할 수 있는 표시 방식을 개선합니다.
- benchmark fixture를 주기적으로 보강해 공식 문서 검색, 로컬 RAG, 업로드 검색, tool action 흐름의 회귀 범위를 넓힙니다.
