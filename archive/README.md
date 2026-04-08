# Archive Notes

`archive/`는 현재 실행 경로에 포함되지 않는 과거 코드와 참고 문서를 보관하는 디렉터리입니다. 유지보수와 기능 수정은 기본적으로 `src/` 기준으로 진행해야 합니다.

## 현재 포함된 항목

| 경로 | 설명 |
|---|---|
| `archive/legacy_code/baseline_code.py` | 과거 베이스라인 실험 코드 |
| `archive/legacy_code/router_experiment.py` | 예전 라우팅 실험 코드 |
| `archive/team_docs/Langchain_Project_Team_3.pdf` | 프로젝트 발표/팀 문서 보관본 |

## 사용 원칙

- `archive/` 아래 파일은 현재 서비스 import 경로에 포함되지 않습니다.
- 현재 동작 확인, 기능 수정, 테스트 추가는 `src/`와 `tests/` 기준으로 진행합니다.
- 과거 구현 의도나 비교가 필요할 때만 참고 자료로 사용합니다.
