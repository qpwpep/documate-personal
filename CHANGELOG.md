# Changelog

이 프로젝트는 Keep a Changelog에 가까운 형식으로 변경 이력을 정리합니다. `0.2.0`은 현재 `pyproject.toml`에 선언된 패키지 버전이며, 아직 tag/release가 없는 기준선입니다.

## [Unreleased]

### Changed

- `src/tools/docs_search.py`가 공식 문서 검색 결과를 도메인 + 경로 prefix allowlist로 제한하고, 일부 API/심볼 질의에 대해 라이브러리별 query hint와 fallback query를 적용하도록 정리했습니다.
- `src/nodes/validation.py`, `src/nodes/retry.py`, `src/nodes/retrieval.py`가 `docs`, `upload`, `local` 경로별 evidence 검증과 선택적 재시도 규칙을 사용하도록 재구성했습니다.
- `docs + upload` 혼합 retrieval에서 `docs`만 실패한 경우, 성공한 upload evidence와 진단 정보를 보존한 채 `docs`만 재시도하도록 조정했습니다.
- `src/nodes/synthesis.py`가 upload 중심 1~2개 evidence에 대해 `deterministic_grounded_direct` 경로로 grounded payload를 직접 생성하도록 변경했습니다.
- `src/agent_manager.py`가 구조화된 LLM call trace가 없을 때 현재 턴 `AIMessage`의 response/usage metadata로 `debug.llm_calls`를 보강하도록 정리했습니다.
- `README.md`, `docs/benchmarking.md`를 현재 런타임 구조, 환경변수 기본값, FastAPI 공개 응답 스키마 기준으로 다시 동기화했습니다.

## [0.2.0] - current untagged baseline

### Changed

- LangGraph 런타임을 조립 진입점(`src/graph_builder.py`), 그래프 토폴로지(`src/make_graph.py`), 모델 레지스트리(`src/llm.py`), 노드 구현(`src/nodes/*`)으로 분리했습니다.
- planner -> retrieval -> synthesis -> validation -> postprocess 흐름을 현재 상태 타입과 라우팅 규칙 기준으로 재구성했습니다.
- 구조화된 evidence 응답과 1회 재시도 컨텍스트를 FastAPI 응답 스키마에 맞춰 정리했습니다.

### Added

- FastAPI/Streamlit 런타임 관리, 세션 TTL/LRU 캐시, 생성 파일 및 업로드 정리 루틴을 현재 `src/web/app.py`와 `src/service_manager.py` 기준으로 유지합니다.
- 온라인 벤치마크 실행, 리포트 재생성, 이력 요약을 위한 `src.eval` 도구 체계를 포함합니다.
- 레거시 코드와 팀 산출물을 `archive/` 아래로 분리해 현재 유지보수 대상 경로를 명확히 했습니다.
