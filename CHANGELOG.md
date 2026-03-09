# Changelog

이 프로젝트는 Keep a Changelog에 가까운 형식으로 변경 이력을 정리합니다. `0.2.0`은 현재 `pyproject.toml`에 선언된 패키지 버전이며, 아직 tag/release가 없는 기준선입니다.

## [Unreleased]

### Changed

- README를 실제 런타임 구조(`src/graph_builder.py`, `src/make_graph.py`, `src/llm.py`, `src/nodes/*`) 기준으로 다시 정리했습니다.
- 벤치마크 상세를 `docs/benchmarking.md`로 분리하고, README에는 검증 현황 요약만 남기도록 문서 구조를 조정했습니다.

## [0.2.0] - current untagged baseline

### Changed

- LangGraph 런타임을 조립 진입점(`src/graph_builder.py`), 그래프 토폴로지(`src/make_graph.py`), 모델 레지스트리(`src/llm.py`), 노드 구현(`src/nodes/*`)으로 분리했습니다.
- planner -> retrieval -> synthesis -> validation -> postprocess 흐름을 현재 상태 타입과 라우팅 규칙 기준으로 재구성했습니다.
- 구조화된 evidence 응답과 1회 재시도 컨텍스트를 FastAPI 응답 스키마에 맞춰 정리했습니다.

### Added

- FastAPI/Streamlit 런타임 관리, 세션 TTL/LRU 캐시, 생성 파일 및 업로드 정리 루틴을 현재 `src/web/main.py`와 `src/main.py` 기준으로 유지합니다.
- 온라인 벤치마크 실행, 리포트 재생성, 이력 요약을 위한 `src.eval` 도구 체계를 포함합니다.
- 레거시 코드와 팀 산출물을 `archive/` 아래로 분리해 현재 유지보수 대상 경로를 명확히 했습니다.
