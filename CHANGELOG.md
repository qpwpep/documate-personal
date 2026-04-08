# Changelog

이 프로젝트는 Keep a Changelog 스타일에 가깝게 변경 이력을 관리합니다. 현재 패키지 버전 기준은 `pyproject.toml`의 `0.2.0`입니다.

## [Unreleased]

### Changed

- `README.md`를 현재 패키지 구조, 실행 흐름, 환경 변수, API 계약 기준으로 다시 정리했습니다.
- `docs/benchmarking.md`를 `src/eval/main.py`와 `data/benchmarks/config.toml` 기준으로 정리했습니다.
- `archive/README.md`와 `.github/ISSUE_TEMPLATE/task.md`를 현재 저장소 상태에 맞게 갱신했습니다.

## [0.2.0] - current untagged baseline

### Added

- FastAPI + Streamlit 런타임과 세션 TTL/LRU 캐시, 업로드/생성 파일 cleanup 루프를 추가했습니다.
- 온라인 benchmark CLI와 report/history 생성 파이프라인을 추가했습니다.
- 로컬 노트북 RAG 인덱싱과 업로드 파일 전용 검색 흐름을 추가했습니다.

### Changed

- LangGraph 실행 경로를 `graph_builder`, `make_graph`, `nodes/*`, `tools/*`, `web/*` 중심의 구조로 정리했습니다.
- 응답 스키마를 `answer`, `claims`, `evidence`, `confidence`, `sections` 중심의 구조화된 payload로 정리했습니다.
- planner, retrieval, synthesis, validation, action postprocess 단계를 기준으로 디버그/관측성 정보를 수집하도록 정리했습니다.
- 공식 문서 검색은 docs allowlist와 query hint 규칙을 기준으로 동작하도록 정리했습니다.
