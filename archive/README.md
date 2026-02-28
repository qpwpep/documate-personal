# Archive Notes

이 디렉터리는 현재 런타임에서 사용하지 않는 레거시 코드/문서를 보관합니다.

## 이관 사유

- `src/`에는 현재 서비스 실행 경로만 남기고, 실험/참고 코드는 분리해 아키텍처 경계를 명확히 하기 위함
- 개인 프로젝트 유지보수 관점에서 런타임 코드와 과거 팀 산출물을 분리하기 위함

## 이동 매핑

| 이전 위치 | 현재 위치 | 상태 |
|---|---|---|
| `src/baseline_code.py` | `archive/legacy_code/baseline_code.py` | 런타임 미사용(참고용) |
| `src/router.py` | `archive/legacy_code/router_experiment.py` | 런타임 미사용(실험 코드) |
| `docs/Langchain_Project_Team_3.pdf` | `archive/team_docs/Langchain_Project_Team_3.pdf` | 레거시 발표자료 |

## 사용 원칙

- `archive/` 하위 파일은 런타임 import/실행 경로에 포함하지 않습니다.
- 기능 구현/수정은 `src/` 하위 현재 구조를 기준으로 진행합니다.
