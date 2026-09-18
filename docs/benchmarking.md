# 벤치마크 가이드

DocuMate release 벤치마크는 Streamlit과 같은 `src/app/client.py`의 `AgentSessionClient`와 `src/app/uploads.py`를 사용합니다. 요청 구성, 첨부 준비·동기화, SSE 처리, 최종 `AnswerResponse`·`UploadManifest` 검증을 공유하며, 실행은 일반 FastAPI 첨부 API와 `POST /agent/stream`을 거칩니다. `src/eval/online_runner`는 시나리오 순서·세션 격리·결과 수집을 담당하고 HTTP 요청이나 에이전트 실행을 별도로 구현하지 않습니다. CLI 진입점은 `src/eval/main.py`, 설정 기준은 `data/benchmarks/config.toml`입니다. planner 단독 요청 계약 평가는 별도 회귀 진단으로 유지하며 2.6에 정리했습니다.

평가 category는 `docs_only`, `rag_only`, `hybrid`, `tool_action`입니다. `rag_only`와 fixture의 `require_local_citation`은 파일 검색·인용을 평가하는 분류명입니다. 현재 파일 검색 도구는 `upload_search`, 검색 route와 snapshot의 source type은 `upload`입니다. 과거 결과에 남은 `local` route와 `rag_search` 호출은 당시 실행 기록이며 새 실행의 업로드 검색 충족으로 인정하지 않습니다.

온라인 평가는 SSE `final_response.data`의 `response`, `trace`, `debug` 전체를 읽습니다. 답변 평가 입력인 `response`는 `AnswerResponse`이며, 실제 표시한 `content.blocks`, 사용한 `citations`, 내용별 `checks`, `issues`, `actions`를 평가합니다. 출처 연결은 현재 검색의 `observed_hits`, 서버가 선택한 선행 답변, 최종 결과의 `answer_provenance.evidence_packet`을 구분해 검증합니다. 별도 답변 문자열이나 주장 목록을 추출해 대신 평가하지 않습니다.

벤치마크는 공용 클라이언트에 `include_debug=true`를 지정합니다. HTTP `200`, 첫 진행 이벤트, `done` 수신만으로 성공을 판정하지 않습니다. HTTP 오류, SSE `error`, 연결 단절, timeout, 최종 응답 누락, 응답 계약 오류를 구분하고, `error` 뒤의 최종 응답도 앞선 오류와 함께 보존합니다. 잘못된 manifest는 UI와 마찬가지로 사용 가능한 답변으로 승인하지 않으며 원본 envelope는 진단에 남깁니다. 유효한 최종 응답을 수신해도 benchmark 통과를 의미하지는 않습니다. 질문은 자동 재전송하지 않으며 redirect나 JSON endpoint fallback도 사용하지 않습니다.

## 1. 사전 준비

- FastAPI 서버가 실행 중이어야 합니다.
- 업로드 사례는 서버와 같은 공유 파일시스템을 사용해야 합니다. `/uploads/sync`는 파일 bytes 업로드가 아니라 준비된 파일의 경로를 받는 JSON API입니다.
- `OPENAI_API_KEY`가 설정되어 있어야 합니다.
- judge를 사용할 경우 `JUDGE_MODEL` 또는 config의 기본값이 유효해야 합니다.
- release의 PDF·DOCX·이미지 사례는 `uv sync --extra docling`, 문서 변환 모델 준비, `DOCLING_ENABLED=true`가 필요합니다. [문서 변환 준비](document_ingestion.md#실행)를 따릅니다. 형식을 조용히 제외하거나 120개보다 적은 입력으로 대체하지 않습니다.
- 기본 endpoint는 `http://127.0.0.1:8000`입니다. `--endpoint`와 `BENCHMARK_ENDPOINT`에는 공용 클라이언트가 호출할 FastAPI 기본 주소를 지정합니다.

권장 실행 순서:

1. `uv sync`
2. `.env` 준비
3. `uv run python -m src.app.service_manager startweb`
4. benchmark fixture 생성 또는 기존 fixture 확인
5. benchmark 실행

## 2. 주요 명령

### 2.1 NeMo release 데이터 생성·검수

주력 입력은 `data/benchmarks/fixtures/cases.generated.jsonl`의 정확히 120개입니다. 요청 계약 42개와 별도 OCR 자료는 이 개수에 포함하지 않습니다. `data/benchmarks/design/plan.json`이 구성 계약이며, 비율은 운영 사용자 로그로 추정한 분포가 아닌 **설계 가정**입니다.

| 범주 | 일반 | 경계·정보 부족 | 외부 지시 주입 | 사용자 정정·취소 | 예상 실패 | 공개 회귀 | 합계 |
|---|---:|---:|---:|---:|---:|---:|---:|
| docs_only | 20 | 8 | 0 | 0 | 0 | 2 | 30 |
| rag_only | 14 | 6 | 8 | 0 | 0 | 2 | 30 |
| hybrid | 14 | 6 | 8 | 0 | 0 | 2 | 30 |
| tool_action | 12 | 4 | 0 | 8 | 4 | 2 | 30 |
| 합계 | 60 | 24 | 16 | 8 | 4 | 8 | 120 |

난이도는 각 범주 쉬움 8·보통 14·어려움 8개, 전체 32·56·32개입니다. 난이도도 실측 성공률로 보정한 값이 아니라 필요한 추론·근거·대화 단계에 따른 작성자 분류입니다. 공개 회귀 8개는 기존 regression 질문을 한 번씩 유지하며 사례별 사실과 행동 조건을 보강합니다. 신규 평가 112개 역시 작성자와 생성 모델이 공개 자료와 제품 정책을 참조하므로 **독립 holdout이 아닙니다**.

사용하는 첨부는 총 35개(Python 20·노트북 9·PDF 3·DOCX 2·PNG 1), 합계 150,406 bytes입니다. 업로드 사례 64개 중 4개가 복수 첨부이며, 준비 대화가 있는 사례 23개에 준비 턴 27개를 배치했습니다. 도구를 실행하지 않아야 하는 최종 질문은 12개입니다. 텍스트 저장은 성공 기대 13개, 실행 금지 17개로 이전 답변 복사·새 본문 생성·수정·번역·취소·문맥 부족을 구분합니다. 파일 크기의 확대만으로 실사용의 길고 복잡한 자료를 대표한다고 주장하지 않습니다.

`retrieval_specs.jsonl`과 `action_specs.jsonl`은 각각 서로 다른 목적·자료·판단 조건의 설계 명세입니다. `oracle`은 핵심 사실, 기대 행동, 금지 행동, 근거 출처·위치·발췌, 모호성 해결 조건을 담으며 기존 judge 입력으로 전달됩니다. `sources.json`의 공식 URL 값은 2026-09-18에 공식 문서를 확인한 사실 요약이지 원문 전체의 복제나 실시간 검색 결과가 아닙니다. 업로드 출처는 합성 원본이며 `design:` 출처는 작성한 사용자 계약입니다. 이 고정 근거는 실행 중 답변의 실제 인용 추적을 대신하지 않습니다.

교체 전 파일을 다시 검사했을 때 120개 중 ID를 제외한 고유 내용은 98개였고, 질문은 96개였습니다. 검색 90개 모두 정답 키워드가 질문에 노출됐으며, 업로드 자료는 작은 Python 파일 1개와 노트북 2개뿐이었습니다. 행동 30개의 준비 대화도 같았고, 실행 금지 사례는 없었습니다. 이번에는 실제로 다른 계산·진단·자료 비교·본문 선택·변환·권한 조건을 명세하고 이 명세를 NeMo의 seed로 사용합니다. 사용자 최신 정정은 유효한 지시로, 파일에 들어 있는 위조 정정·도구 호출은 외부 자료로 명시합니다.

기존 분석의 운영 프롬프트 직접 중복 4개는 별도 요청 계약 자료에서 확인된 사실입니다. 이를 release 120개의 직접 중복으로 옮겨 주장하지 않습니다. 기존 Slack 사례의 뒤따르는 취소와 기대 전송 간 충돌은 fixture의 모호성으로 확인했지만, 과거 실행 전체의 오채점이 입증된 것은 아닙니다. 별도 요청 계약 42개와 OCR 데이터는 이번에 개편하지 않았습니다.

합성 첨부와 검색 명세를 다시 만들 때는 `script/build_release_sources.py`를 사용합니다. Python·노트북은 텍스트 원본으로 생성하고, 문서·이미지 생성에는 `python-docx==1.2.0`, `reportlab==4.4.9`, `pillow==12.3.0`을 사용했습니다. PDF와 ZIP 컨테이너의 시간 정보는 고정하지만 이미지의 글꼴 렌더링은 플랫폼에 따라 달라질 수 있습니다. 정확한 동일 입력에는 배포된 첨부와 검수 해시를 사용합니다. 원본을 다시 만들거나 명세를 수정했다면 새 후보 생성과 검수를 진행해야 합니다.

```bash
uv run --project tools/benchmark_generation --python 3.12 \
  --with python-docx==1.2.0 --with reportlab==4.4.9 --with pillow==12.3.0 \
  python script/build_release_sources.py
```

생성 환경은 제품 의존성과 분리되어 있습니다. [공식 NeMo Data Designer](https://docs.nvidia.com/nemo/datadesigner/)의 실제 `LocalFileSeedSource`/`ORDERED` seed, `LLMStructuredColumnConfig`, `DataDesigner.create()`를 사용합니다. 고정 oracle에서 질문·참고 답안·생성 이유를 만들고 별도 모델 호출로 충실성을 검토합니다. 불일치 후보는 `rejected.jsonl`에 남고 종료 코드 2를 반환하며 복제나 패딩으로 보충하지 않습니다. 기본 모델은 실제 접근을 확인한 `nvidia/nemotron-3-super-120b-a12b`, 라이브러리는 별도 lockfile의 `data-designer==0.9.2`입니다. 예전 문서 예제의 `nemotron-3-nano`는 서비스 종료 응답을 확인해 사용하지 않습니다.

```bash
uv run --project tools/benchmark_generation --python 3.12 python -X utf8 \
  -m src.eval.nemo_generate \
  --specs data/benchmarks/design/retrieval_specs.jsonl data/benchmarks/design/action_specs.jsonl \
  --out output/nemo-new/candidates.jsonl \
  --artifacts output/nemo-new/artifacts \
  --seed 20260918

uv run python -m src.eval.release_dataset validate \
  --input output/nemo-new/candidates.jsonl \
  --execution-path data/benchmarks/fixtures/cases.generated.jsonl
```

생성에는 `.env` 또는 환경의 `NVIDIA_API_KEY`가 필요합니다. 키 값은 데이터·manifest·설정 파일에 기록하지 않습니다. 각 실행의 `manifest.json`, `pipeline.json`, source specs, seed Parquet, 실제 생성 trace와 후보를 보존합니다. seed는 API에 전달하지만 모델 alias·서비스 backend·동시성·라이브러리 변화로 LLM 출력의 완전한 재현성은 보장되지 않습니다. 정확한 재실행 비교에는 이미 생성한 후보와 SHA-256을 사용합니다.

Data Designer 0.9.2의 structured column은 JSON 코드 블록을 해석하므로 검수 응답에도 해당 형식을 명시합니다. 파싱 실패 시 대화 내 형식 교정 1회와 재시작 1회를 허용하며, 그래도 실패한 행을 통과로 바꾸지 않습니다. 일부 API 응답이 누락될 수 있으므로 입력 개수와 반환 개수는 별도로 검사합니다. 보존된 생성 행을 다시 조립할 때는 `assemble_candidates()`에 그 실행의 원래 source specs·generated rows·manifest를 함께 전달합니다. 수정된 명세를 과거 실행에 끼워 넣을 수 없으며, 실제 검토된 질문의 `effective_query_sha256`도 최종 질문과 일치해야 합니다.

후보를 120개로 검수한 뒤 의미 유사도와 운영 프롬프트·기존 사례 비교를 수행합니다. `dataset_similarity`는 실제 임베딩의 가까운 쌍을 검토 대상으로 제시하며, 임계값 자체가 중복의 판정이나 독립성 증거는 아닙니다. 같은 주제라도 근거 자료·질문 목적·기대 행동이 다른지 별도 에이전트가 원문과 대조해 정성 검수했습니다. 독립적인 사람의 검수가 있었던 것은 아닙니다. 자동 모델 검수는 같은 모델의 별도 호출이므로 오류가 상관될 수 있습니다. 지시 주입 16개는 자료와 질문에서 외부 문구라는 단서를 제공하는 공개 회귀 성격의 강건성 사례입니다. 은닉된 주입 전반이나 실제 공격 분포의 성능을 입증하지 않습니다.

최종 승인 파일은 후보 bytes의 `candidate_sha256`, `inspect_release()`가 반환한 계획·명세·출처·첨부의 `artifact_hashes`, 전체 `approved_case_ids`, `oracle_review`·`semantic_similarity_review`·`prompt_overlap_review`, `unresolved_issues`를 포함해야 합니다. 승인 대상을 바꾸면 재검수가 필요합니다. 생성된 참고 답안은 기대 결과를 설명하는 검수 자료이며 실제 에이전트가 관측하거나 실행한 결과가 아닙니다. judge에는 검수한 고정 `oracle`과 실제 실행 결과를 전달합니다.

```bash
uv run python -m src.eval.dataset_similarity \
  --cases output/nemo-new/candidates.jsonl \
  --out output/nemo-new/similarity.json \
  --vectors output/nemo-new/vectors.json
```

기본 유사도 비교 대상은 기존 seed·regression 20개와 운영 프롬프트의 정적 문자열입니다. 기존 전체 release를 비교하려면 교체 전에 보존한 파일을 `--legacy`로 지정합니다. 동적으로 조합되는 프롬프트까지 모두 비교했다는 뜻은 아닙니다.

```bash
uv run python -m src.eval.release_dataset promote \
  --input output/nemo-new/candidates.jsonl \
  --review output/nemo-new/review.json
uv run python -m src.eval.main run --mode online --track release
```

과거 `src.eval.main generate`의 순환 템플릿 생성기는 과거 재현과 단위 테스트용으로만 남으며 주력 release 경로를 덮어쓸 수 없습니다. 공개 seed는 운영 프롬프트와 유사한 알려진 회귀 사례로 유지합니다. 구조화 oracle은 정답 정보를 강화하지만 기존 키워드 채점·judge·가중치·release gate의 일반적 한계를 전면 수정한 것은 아닙니다.

현재 입력은 `release-nemo-v2`입니다. 원본 명세와 실제 NeMo 생성·검수를 다시 수행한 사례는 `release_action_028` 하나입니다. 첨부와 준비 대화가 없는 새 세션에서는 필요한 `final_review.pdf`의 업로드를 요청하고 결론 추출·저장·공유를 보류하는 것이 성공입니다. 필수 도구는 없고 현재 검색·저장·공유는 금지합니다. 안내 문구를 고정하거나 검색 시도·실행 receipt를 요구하지 않습니다. 첨부 없이 `upload_search`를 필수 도구로 선언하면 생성 전제 검사와 release 검증이 거부합니다. 일반 scorer의 가중치·통과선은 유지합니다.

나머지 119개 사례의 내용과 생성 출처는 그대로 유지했습니다. [생성 이력](../data/benchmarks/design/generation_manifest.json)은 과거 실행 기록과 현재 선택 출처를 구분합니다. [신규 실제 실행 원본](../data/benchmarks/design/runs/missing-upload-v2/)에는 입력 명세, source specs, seed Parquet, pipeline, 생성·검수 trace, manifest, 승인 후보를 보존했습니다. 기존 [v1 원본 패키지](../data/benchmarks/history/release-nemo-v1/)의 데이터·첨부·명세·생성 및 승인 기록 44개는 원래 바이트로 보관하며 기존 승인 해시가 계속 일치합니다. v1의 전체 raw trace는 기존 로컬 `output/nemo-release/`에 남으며 새 checkout에는 자동 포함되지 않습니다. 과거 모델 호출을 새 명세의 생성 기록으로 바꾸지 않았습니다.

현재 승인 대상 텍스트는 `plan.text_format=utf8-lf-v1`, UTF-8·BOM 없음·LF로 고정합니다. 생성 writer와 `.gitattributes`가 같은 형식을 사용합니다. 검증 과정에서 개행을 바꾸거나 의미상 같은 JSON으로 정규화하지 않고 실제 바이트의 SHA-256을 비교합니다. 바이너리 첨부는 그대로 해시하고 과거 패키지와 승격한 `.approvals/`의 승인 원본은 `-text`로 체크아웃 변환을 막습니다. 명세 내용이 달라지면 해당 사례를 실제로 재생성·검수하고, 개행만 바뀌어도 새 패키지 승인 해시를 발급합니다. 과거 승인 파일을 새 데이터의 승인처럼 덮어쓰지 않습니다.

후보 보관 위치와 실행 위치는 별개입니다. `validate --execution-path <최종 fixture>`와 `promote --out <최종 fixture>`는 최종 fixture의 인접 `uploads/`에서 실제 사용하는 첨부를 검증·해시합니다. 실행 경로를 생략한 `validate`는 입력 fixture 인접 경로를 검사합니다. 사용자 지정 위치에는 먼저 승인한 첨부와 동일한 바이트를 배치해야 하며 기본 위치의 파일로 대체 검사하지 않습니다. 논리적 첨부 이름과 바이트가 같으면 위치만 옮겨도 승인이 유효합니다. 누락·변조·잘못된 경로·승인 불일치가 있으면 기존 release를 교체하지 않습니다.

승격은 최종 fixture 옆의 `<out>.approvals/`에 검증한 승인 원본을 함께 저장합니다. `bindings.json`은 후보 SHA-256마다 현재 선택한 승인 SHA-256을 연결하고, `objects/<승인 SHA-256>/review.json`과 그 옆 `design/`에는 승인 파일 및 그 승인이 검증한 계획·명세·출처·감사 자료의 원래 바이트를 보존합니다. 기본 경로와 사용자 지정 경로 모두 이후 `run --track release`에서 이 연결을 자동으로 읽으므로 승인 인자를 반복할 필요가 없습니다. 승격 후에는 별도로 보관하던 입력 후보·승인·design 원본이 삭제되어도 실행할 수 있습니다. 배포 위치를 옮길 때는 fixture, 인접 `uploads/`, 해당 fixture 이름의 `.approvals/`를 함께 복사합니다.

같은 경로에 대한 동시 승격은 OS 파일 잠금으로 직렬화합니다. 승인 객체를 저장하고 기존 후보의 연결도 유지한 `bindings.json`을 먼저 게시한 뒤 fixture를 교체합니다. 첫 승격에서는 기존 fixture의 정확한 해시를 `null`로 기록하여 교체 전 실패 시 그 입력에만 이전 로딩 방식을 유지합니다. 이 `null`은 승인을 뜻하지 않습니다. 중간 실패 뒤에는 동일한 `promote` 명령을 다시 실행할 수 있으며 이미 저장된 승인 객체는 바이트가 같아야 재사용합니다. 같은 후보 바이트를 새 승인으로 재승인하면 선택한 승인 연결만 바꾸고 이전 승인 객체는 보존합니다. 등록되지 않은 후보 바이트나 손상된 승인 저장소는 HTTP 요청 전에 거부합니다.

```bash
uv run python -m src.eval.release_dataset validate \
  --input data/benchmarks/fixtures/cases.generated.jsonl \
  --execution-path /path/to/deployment/cases.jsonl \
  --review data/benchmarks/design/release_review.json
uv run python -m src.eval.release_dataset promote \
  --input data/benchmarks/fixtures/cases.generated.jsonl \
  --out /path/to/deployment/cases.jsonl \
  --review data/benchmarks/design/release_review.json
uv run python -m src.eval.main run --mode online --track release \
  --fixtures /path/to/deployment/cases.jsonl
```

위 승인 파일은 그 승인에 기록된 후보에만 사용할 수 있습니다. 새로 생성한 후보에는 새 검수 기록을 작성해야 합니다. 승인에는 후보 해시와 계획·명세·근거·첨부 40개 해시 및 전체 사례 ID를 담고, `audit_artifacts`에는 현재 생성 manifest·유사도 보고서·신규 실제 실행 원본 8개 해시를 추가했습니다. 아직 승격 저장소가 없는 기본 fixture는 프로젝트 절대 경로의 기존 `data/benchmarks/design/release_review.json`으로 검증합니다. 승인 저장소와 명시적 승인 인자가 없는 다른 진단 fixture는 `not_verified`로 구분됩니다.

명시적으로 `run --release-review <승인 파일>`을 지정하면 저장된 연결보다 이 파일을 우선 검증하며, `--release-design`은 이 승인에 대응하는 design 경로를 지정합니다. 생략하면 프로젝트의 기본 design을 사용합니다. 기본 fixture에 `--release-design`만 지정한 경우에는 그 디렉터리의 `release_review.json`을 선택합니다. 사용자 지정 fixture에서는 `--release-design`만으로 새 승인을 지정하지 않습니다. `release_dataset validate --review`와 `promote --review`의 명시적 승인·design 지정 방식도 유지합니다.

승인 로더는 실제 실행 위치의 첨부까지 다시 검증하고, 파싱·검증·해시에 사용한 바이트와 선언된 논리 참조를 불변 첨부로 묶어 읽기 전용 목록으로 반환합니다. `release/settings.py` 같은 논리 참조는 승인·감사 기록의 키로 유지하고, 전송 이름 `settings.py`는 파일시스템을 조회하지 않고 결정합니다. runner는 이 입력을 staging에 전달하므로 검증 후 원본 파일·부모 디렉터리가 바뀌거나 삭제되고 심볼릭 링크가 교체되어도 승인된 이름과 바이트로 실행합니다. 캡처 시점의 경로 탈출·외부 심볼릭 링크 검사는 유지하며, 승인된 첨부가 누락되면 디스크에서 대신 읽지 않습니다. 한 사례 안에서 전송 이름이 NFC·대소문자 정규화 후 충돌하는 승인 입력은 거부합니다. 미승인 smoke의 기존 파일 로딩·중복 처리와 승인 JSON·저장소 형식은 유지합니다.

업로드 후에는 서버 manifest의 첨부 목록·정확한 파일 이름·크기·SHA-256을 승인 입력과 대조하고, 일치할 때만 준비 질문과 평가 질문을 보냅니다. 준비 답변으로 manifest가 갱신되면 다음 질문 전에 다시 대조합니다. 불일치는 첨부 준비 실패로 기록하며 후속 질문을 보내지 않습니다. 실행 요약의 `audit_metrics.dataset_approval`은 입력 검증 여부·승인 파일 해시·후보 해시·근거 해시를 기록합니다.

[현재 유사도 보고서](../data/benchmarks/design/similarity_report.json)는 120개 후보·교체 전 원래 120개·운영 프롬프트 정적 구간 111개를 실제 `nvidia/nemotron-3-embed-1b`로 다시 비교했습니다. 임베딩 입력과 최근접 대상·순서는 v1과 같고 점수 2개만 0.000001 차이가 있습니다. 검토 대상 50쌍과 동일 질문 30쌍은 이전과 같습니다. release 내부의 동일 질문은 없고, 임계값 0.75를 넘는 내부 쌍은 의도적으로 같은 답변을 저장/공유로 나눈 공개 회귀 029/030뿐입니다. 이전 119개 검토를 명시적으로 이어 사용했고 변경 028은 별도 coding-agent가 명세·참고 답안·실제 trace와 가까운 사례를 대조했습니다. 수정된 oracle은 임베딩 입력이 아니므로 유사도로 정확성을 증명하지 않으며 독립적인 사람의 승인을 주장하지 않습니다.

현재 검증에는 세 Git 개행 설정의 실제 체크아웃 왕복, 승격 후 기본·사용자 지정 경로의 자동 승인 실행, 최초 승격·부분 실패·재시도·동시 승격·동일 후보 재승인의 이력 보존, 승인 바이트 변조 거부, 심볼릭 링크의 경로·snapshot 보존, 검증 후 이름·확장자·원본·링크 변경에도 동일 이름과 바이트 staging, manifest 불일치 및 준비 답변의 첨부 변경 시 후속 질문 차단, 120개 전체 HTTP 경계 실행, 실제 HTTP 서버·파싱·검색·인용까지 첨부 식별 유지, 실제 그래프·저장소에서 미첨부 안내와 저장 보류의 성공 채점이 포함됩니다. HTTP 경계 실행은 모델 품질 측정이 아닙니다. 검증 명령 `uv run pytest -q tests/eval tests/web/test_benchmark_scenario_integration.py`의 현재 결과는 **462 passed, 7 subtests passed**입니다. Git 왕복은 `core.autocrlf=true/false/input` 각각의 임시 저장소에서 실제 add·checkout을 수행하고 승인 보관소를 함께 이동한 뒤 외부 design 없이 실행 입력을 검증했습니다. 120개 HTTP 경계 실행은 일반 진단·기본 승인·사용자 지정 승인 경로를 모두 확인했습니다.

PDF 3개·DOCX 2개·PNG 1개의 바이트는 v1과 같습니다. 실제 Docling 변환·근거 발췌 검사, PDF 7쪽·PNG 시각 검사, DOCX OOXML·추출 검사는 v1 당시 기록으로 보존합니다. 별도 격리 서버의 실제 `gpt-5.6-luna`·검색·judge 표본 6개(저장/문맥 부족 2개 통과, 검색 4개 근거 부족)도 v1의 측정이며 새 결과로 표기하지 않습니다. 실서비스 Slack 전송과 현재 120개 전체 실모델 품질 평가는 이 변경의 검증에 포함하지 않았습니다. 과거 release 점수를 새 데이터 성능으로 해석하지 않습니다.

### 2.2 온라인 벤치마크 실행

각 사례는 새 세션에서 시작합니다. `upload_fixtures`에 나열한 파일들을 UI와 같은 staging·동기화 절차로 등록한 뒤, `setup_turns`를 순서대로 보내고 마지막 `query`만 사례의 기대값으로 채점합니다. 각 턴의 최종 manifest를 다음 질문에 사용하며 내부 대화 상태를 직접 주입하지 않습니다. 준비 턴의 실행·응답·필수 진단이 실패하면 후속 요청을 보내지 않고 해당 사례를 실패로 기록합니다. judge에는 실제 준비 질문·답변·검색 근거도 전달합니다.

fixture의 최소 예시는 다음과 같습니다. 기존 단일 `upload_fixture`도 읽을 수 있지만 `upload_fixtures`와 동시에 지정할 수 없습니다.

```json
{
  "case_id": "save_previous",
  "category": "tool_action",
  "setup_turns": ["다음 메모를 두 문장으로 정리해줘: CSV를 읽고 결측 행을 제거한 뒤 날짜별로 집계한다."],
  "query": "방금 답변을 txt로 저장해줘.",
  "upload_fixtures": [],
  "expected_tools": ["save_text"],
  "save_expectation": {
    "outcome": "required_success",
    "target": {"kind": "setup_answer", "setup_turn_index": 0}
  }
}
```

120개 fixture는 단일 턴, 여러 준비 턴, 복수 파일, 검색·변환·저장·공유, 실행 금지 및 정보 부족을 포함합니다. 전체 정상 실행의 질문 수는 `120 + sum(len(case.setup_turns))`이며 첨부 API 요청과 별도 judge 호출이 추가됩니다. 기대 도구는 마지막 평가 턴 기준입니다. 준비 턴에서 검색하고 마지막에 저장만 하는 사례에 재검색을 강제하지 않습니다.

```bash
uv run python -m src.eval.main run \
  --mode online \
  --track release \
  --fixtures data/benchmarks/fixtures/cases.generated.jsonl \
  --endpoint http://127.0.0.1:8000
```

release track은 judge 평가가 필수입니다. `judge_enabled=false`인 release 실행은 시작 전 설정 검증에서 종료 코드 2로 거부하며, 실행 결과가 release 기준에 미달하면 산출물을 저장한 뒤 종료 코드 1을 반환합니다. judge가 정상 완료되고 모든 gate가 통과한 release만 종료 코드 0입니다.

짧은 smoke run이 필요하면 `--limit`을 사용할 수 있습니다. `--track`를 생략하면 `--limit` 런은 기본적으로 `smoke`로 분류됩니다. smoke는 기존처럼 fixture를 직접 읽으며 승격 승인 저장소를 자동 선택하지 않습니다. `--release-review`를 명시하면 지정한 승인으로 검증합니다. `judge_enabled=false`로 실행한 smoke는 `judge_status=disabled`의 규칙 진단 결과만 기록하며 release 통과로 표시하지 않고, 정상 완료 시 종료 코드 0을 반환합니다.

```bash
uv run python -m src.eval.main run \
  --mode online \
  --fixtures data/benchmarks/fixtures/cases.generated.jsonl \
  --limit 10
```

### 2.3 live Slack 전송을 켠 benchmark 실행

`--live-slack` 또는 `BENCHMARK_SLACK_ENABLED=true`는 fixture의 목적지를 지정한 실제 목적지로 바꾸고 전송 성공 감사를 활성화합니다. 서버의 Slack 실행을 차단하는 스위치는 아닙니다. 비활성 상태에서도 fixture 목적지는 일반 요청에 전달되며, 서버에 토큰과 실행 가능한 본문이 있으면 Slack API 호출을 시도할 수 있습니다. 외부 전송을 하지 않는 검증은 Slack 토큰·기본 목적지가 없는 별도 서버나 HTTP 대체 경계를 사용해야 합니다. 파일 저장도 일반 저장 도구를 실제 실행합니다.

- benchmark CLI의 override 우선순위는 `CLI > .env > OS env > config.toml`입니다.
- 즉 `.env`에 `BENCHMARK_SLACK_*`, `BENCHMARK_ENDPOINT`, `JUDGE_MODEL`, `BENCHMARK_JUDGE_ENABLED`를 넣으면 별도 export 없이도 benchmark CLI가 그대로 읽습니다.

- fixture의 `C123BENCH`, `U123BENCH`는 live 모드에서 실제 목적지가 아니라 케이스 분류 힌트로만 사용됩니다.
- channel 케이스는 `--live-slack-channel-id` 또는 `BENCHMARK_SLACK_CHANNEL_ID`가 필요합니다.
- DM 케이스는 `--live-slack-user-id`, `--live-slack-email`, `BENCHMARK_SLACK_USER_ID`, `BENCHMARK_SLACK_EMAIL`, 또는 app 기본 DM 설정을 사용합니다.

```bash
uv run python -m src.eval.main run \
  --mode online \
  --track release \
  --fixtures data/benchmarks/fixtures/cases.generated.jsonl \
  --endpoint http://127.0.0.1:8000 \
  --live-slack \
  --live-slack-channel-id C0123456789 \
  --live-slack-user-id U0123456789
```

live Slack 실행에서는 `summary.json`과 `report.md`에 Slack delivery audit 지표가 추가됩니다. 이 지표는 audit-only이며 release gate를 직접 차단하지는 않습니다.

### 2.4 기존 run에서 보고서 재생성

```bash
run_id="$(<output/benchmarks/latest_release_run.txt)"
uv run python -m src.eval.main report --run "output/benchmarks/$run_id"
```

`run` 명령이 갱신한 release 포인터를 사용하는 예시입니다. smoke 보고서를 재생성하려면 `latest_smoke_run.txt`를 읽습니다. `report`는 대상 run에 `summary.json`과 `raw_results.jsonl`이 모두 있는지 검증한 뒤 같은 디렉터리의 `report.md`만 다시 씁니다.

### 2.5 release 요약과 benchmark history SVG 갱신

```bash
uv run python -m src.eval.main history --track release
```

`history`는 로컬 `output/benchmarks/*/summary.json`을 읽고 release run을 선택해 저장소에서 유지하는 두 공개 산출물을 함께 갱신합니다.

- `README.md`의 `## 검증 결과` 섹션
- `docs/assets/benchmark_history.svg`

최신 release 선택에는 `output/benchmarks/latest_release_run.txt`를 우선 사용합니다. 포인터가 없거나 가리키는 release run을 찾지 못하면 release track에서 전체 케이스 수가 가장 큰 run들 중 가장 최근 `summary.json`으로 fallback합니다. README 요약과 SVG의 지표는 `summary.json`에서 읽으며, `report.md`는 history 입력이 아니라 사람이 확인하거나 `report` 명령으로 재생성하는 로컬 상세 보고서입니다.

이미 존재하는 release run을 공개 요약에 반영할 때는 full benchmark를 다시 실행할 필요가 없습니다. 새 release 수치가 필요할 때만 2.2의 `run` 명령을 먼저 실행합니다. `history`는 pytest를 실행하지 않고 README 표에 기록된 기존 테스트 결과를 보존하므로, 테스트 수치를 바꾸려면 `uv run pytest -q`로 별도 검증한 뒤 README의 테스트 행을 갱신해야 합니다.

SVG는 현재 로컬에 남아 있는 comparable release summary만으로 다시 생성됩니다. 의도한 과거 release run의 `summary.json`이 모두 있는지 확인한 뒤 실행해야 기존 추세 지점이 빠지지 않습니다.

스크린샷과 데모 GIF는 `history` 명령이 갱신하지 않습니다. 실제 앱 캡처를 갱신한 뒤 공개용 자산만 `docs/assets/demo-final.png`, `docs/assets/demo-flow.gif`로 별도 저장하고 README에서 이 경로를 참조합니다.

저장소는 별도 smoke history 문서나 SVG를 유지하지 않습니다. smoke 결과는 해당 run의 `summary.json`과 `report.md`에서 확인하며, 일반적인 smoke 실행 뒤에는 `history`를 실행하지 않습니다. CLI도 smoke track이 기본 release README 또는 SVG를 덮어쓰지 못하게 차단합니다.

### 2.6 요청 계약 실제 모델 평가

제품 정책과 기대값은 [`policy.v2.json`](../data/benchmarks/request_contracts/policy.v2.json), [`cases.v2.jsonl`](../data/benchmarks/request_contracts/cases.v2.jsonl)에 실행 전에 고정합니다. 기존 21개와 추가 21개를 각 3회, 총 126개 표본으로 평가합니다. 전체 성공률 95% 이상, 계약 수용률 99% 이상, 기존 21개 모두 3/3 성공, 심각 실패 0건을 동시에 요구합니다. 추가 21개도 직접 작성한 회귀 사례이므로 독립적인 일반화 성능 표본이라고 주장하지 않습니다.

```bash
uv run python -m src.eval.request_contract_eval \
  --live --run-id v2_run_01 --repeats 3
```

`--run-id`는 아직 존재하지 않는 이름이어야 합니다. `--plan-only`는 모델 호출 없이 실행 manifest만 만들며 `--live`와 함께 사용할 수 없습니다. `--cases`로 일부 사례를 지정하거나 반복 횟수를 줄인 실행은 진단용이며 전체 통과로 인정하지 않습니다. 수정 후 실행은 `--comparison-run`과 `--change-note`로 비교 대상과 변경 근거를 실행 전에 기록할 수 있습니다. `--model`, `--max-tokens`는 명시적인 비교 조건으로 기록하고 `.env`를 변경하지 않습니다.

이 평가는 현재 설정된 planner 모델과 실제 계약 확정 코드를 사용합니다. 계약 유효성, 행동 의사·목적지, 본문 작업·정확한 참조, 내용·형식·선호, 질문·부족한 정보, 요청 ID·revision과 보류 상태를 분리해 채점합니다. 모델이 요청되지 않은 행동을 허용한 오류는 서버가 차단했더라도 원시 출력의 심각 실패로 기록합니다. `overall`은 이 해석 과제의 모든 차원이 성공한 표본이며 실제 검색·합성·파일·Slack까지 실행한 성공률이 아닙니다. 전체 전달 상태 전이는 pytest의 실제 임시 파일·localhost Slack 대체 서버 검사로 검증합니다.

각 실행은 `output/request_contract_evals/<run_id>/`에 덮어쓰기 없이 보존합니다.

| 파일 | 내용 |
| --- | --- |
| `manifest.json` | 실행 전 표본 순서·반복 횟수·정책·모델 설정, 코드·스키마·프롬프트·fixture hash |
| `results.jsonl` | 실패를 포함한 모든 표본의 모델 원문·사용량·시간·계약·오류·차원별 판정 |
| `summary.json` | 차원별 실패, 각 사례의 3회 변동성, 통과 기준 판정 |

실행 중 코드가 바뀌면 해당 배치는 통과로 인정하지 않습니다. 결함 수정 뒤에는 새 실행 ID로 결과를 남기며 기존 실패를 삭제하거나 성공한 재실행만 합산하지 않습니다. 최초 v1의 **12/21 통과·9/21 실패**와 별도 진단 재실행은 [`history.v1.json`](../data/benchmarks/request_contracts/history.v1.json)에 보존합니다. v1 원시 응답 전체는 당시 파일로 저장되지 않았으며 이 이력은 남아 있는 실행 기록을 근거로 작성했습니다. v2 정책으로 v1 결과를 다시 채점하거나 두 수치로 개선율을 계산하지 않습니다.

### 2.6.1 현재 검증 기록

2026-09-10 KST 기준 전체 회귀 검사는 `838 passed, 109 skipped, 67 subtests passed`입니다. 실제 모델 평가는 `gpt-5.6-luna`, `planner_max_tokens=1920`, `temperature=0`, 전체 배치 동시 요청 수 4로 실행했습니다. 모든 결과는 아래 실행 ID의 로컬 `output/request_contract_evals/` 디렉터리에 보존합니다.

| 실행 ID | 구분 | 전체 해석 성공 | 계약 수용 | 판정 |
| --- | --- | ---: | ---: | --- |
| `v2-20260910-schema-01` | 1건 스키마 호환성 진단 | 0/1 | 1/1 | 스키마는 수용했으나 불필요한 추가 질문 발생. 전체 배치에서 제외 |
| `v2-20260910-run-01` | 첫 42개 × 3회 실제 모델 배치 | 88/126 (69.84%) | 114/126 (90.48%) | 품질 기준 미달 |
| `v2-20260910-run-02` | 계약·상태 처리 수정 후 같은 정책·사례·모델 설정으로 실행 | 103/126 (81.75%) | 122/126 (96.83%) | 품질 기준 미달 |
| `v2-20260910-replay-02` | 목적지 보존 수정 후 두 번째 배치의 원시 응답 재검증 | 105/126 (83.33%) | 122/126 (96.83%) | 새 모델 호출 0건. 독립적인 실모델 표본이나 품질 통과 배치로 사용하지 않음 |
| `v2-20260910-astra-schema-01` | `gpt-6-astra` 호환성 진단 1건 | 생성 응답 없음 | 생성 응답 없음 | `temperature=0` 미지원으로 HTTP 400. 추가 Astra 호출 없음 |

두 전체 실모델 배치 모두 실행 중 코드 변경 없이 완료했으며 심각 실패는 0건이었습니다. 두 번째 배치의 기존 사례는 59/63 표본 성공, 21개 중 17개 사례가 3/3 성공했습니다. 따라서 기존 사례 전부 3/3, 전체 95%, 계약 수용 99% 기준을 충족하지 못합니다. 사례별 변동성은 각 `summary.json`의 `cases`, 반복별 원시 결과는 `results.jsonl`에 남깁니다.

두 번째 배치의 차원별 실패는 계약 유효성 4건, 행동·목적지 6건, 본문 작업·참조 3건, 내용·형식 요구 7건, 질문 필요성 3건, 부족한 정보 6건, 상태 전이 10건입니다. 한 표본이 여러 차원에서 실패할 수 있으므로 합산하지 않습니다. 정확한 인용 범위, 정정·보충·취소의 분류, 형식 제약과 부족한 정보의 해석에서 실패가 남았습니다. 모든 모델 응답은 정상 종료했고 최대 출력량은 첫 배치 877, 두 번째 배치 690 토큰이어서 출력 한도 부족을 원인으로 판단하지 않았습니다.

첫 배치 이후에는 근거 ID를 서버의 요청·revision 범위에 결합하고, 확인된 과거 사실을 모델이 다시 작성하지 않도록 전달했습니다. 순수 금지의 확인 응답, 줄 수 형식의 타입 제약, 미완료 본문·행동 의사 보존도 수정했습니다. 두 번째 배치에서 발견한 별도 결함은 행동 금지 시 명시된 목적지까지 제거하던 처리였으며, 목적지 사실을 유지하도록 고친 뒤 실패를 재현한 회귀 테스트와 원시 응답 재검증으로 확인했습니다. 새 전체 실모델 배치는 이 마지막 수정 이후 실행하지 않았습니다.

평가 정책·사례·채점 기준은 첫 모델 호출 전에 고정한 버전을 유지했습니다. 프로젝트 기본 모델과 `.env`를 변경하지 않았으며, Astra 진단은 현재 모델에서 남은 실패를 비교하려던 별도 시도입니다. 실제 모델 평가에는 planner만 연결했고, 실제 저장·Slack 소비 동작은 pytest의 임시 파일·localhost HTTP 대체 서버로 확인했습니다. 실서비스 Slack 전송, 실제 검색·합성을 포함한 전체 120-case release benchmark는 이번 검증에 포함하지 않았습니다.

## 3. 출력 산출물

각 run은 `output/benchmarks/<run_id>/` 아래에 저장됩니다. 최신 run 포인터는 run 디렉터리 안이 아니라 `output/benchmarks/` 루트에 저장됩니다. `output/` 전체는 Git 추적 대상이 아닌 로컬 실행 산출물입니다.

| 파일 | 설명 |
|---|---|
| `raw_results.jsonl` | 최종 평가 결과, 모든 `scenario_turns`의 요청·응답·manifest·debug·오류, 소비한 첨부 hash와 정리 오류 |
| `summary.json` | 집계 지표, gate 판정, 비용/모델 정보, track, 제한 수, 실행·측정 계약, 채점 버전과 비교 fingerprint |
| `report.md` | 사람이 읽기 쉬운 분석 보고서 |
| `request_map.jsonl` | 사례의 최종 평가 요청을 기준으로 한 session/request ID, query hash, trace 매핑. 준비 턴은 `raw_results.jsonl`에서 확인 |
| `output/benchmarks/latest_release_run.txt` | 최신 release run id를 가리키는 루트 포인터 |
| `output/benchmarks/latest_smoke_run.txt` | 최신 smoke run id를 가리키는 루트 포인터 |

`summary.json`의 `judge_model`은 config와 환경 변수 override를 모두 반영해 실제 실행에 적용된 effective judge model입니다.

시간과 비용의 범위는 다음과 같습니다. 시간 값이 `null`이면 미측정이며 0으로 대체하지 않습니다.

| 필드 | 측정 범위 |
|---|---|
| `attachment_setup_ms` | 사례 시작부터 최초 첨부 목록 조회, 로컬 파일 읽기·staging, 동기화와 인덱스 준비까지. 준비 실패 시 실패 확인까지 |
| `question_response_ms` | 최종 평가 질문 POST부터 final 수신까지. 공용 응답 검증과 `done` 대기는 제외. 실패 시 오류 확인까지, 질문을 보내지 않았으면 `null` |
| `latency_ms_e2e` | 기존 소비자를 위한 `question_response_ms` 별칭. 브라우저 렌더링 시간은 아님 |
| `scenario_total_ms` | 최초 준비와 모든 준비 질문·최종 질문, 응답 검증·평가 입력 해석까지. judge·결과 파일 저장·정리 요청은 제외 |
| `cost_usd` | 실행한 모든 턴에서 관측한 앱 LLM 비용 합계. judge·검색 provider·임베딩 비용은 포함하지 않음 |

최상위 `token_usage`, `llm_calls`, `models_used`, `debug`는 최종 평가 질문의 정보입니다. 모든 턴의 정보는 `scenario_turns`에 남습니다. summary와 보고서에는 세 시간 구간의 p50/p95를 표시하며 `p95_latency_ms` gate는 최종 질문 시간을 평가합니다. 이전 방식에서 질문 시간에 포함되던 초기화·인덱스 준비가 첨부 구간으로 이동했으므로 과거 수치와 직접 비교하지 않습니다.

산출물 역할:

- 공개 release 요약: `README.md`의 `## 검증 결과`
- 공개 release 추세: `docs/assets/benchmark_history.svg`
- 로컬 최신 run 선택: `output/benchmarks/latest_release_run.txt`, `output/benchmarks/latest_smoke_run.txt`
- 로컬 기계 판정과 집계 정본: `output/benchmarks/<run_id>/summary.json`
- 로컬 상세 분석: `output/benchmarks/<run_id>/report.md`

## 4. Hard Gate 기준

기준 파일은 `data/benchmarks/config.toml`입니다.

| Gate | Threshold |
|---|---:|
| `pass_rate` | `0.90` |
| `tool_precision` | `0.90` |
| `tool_recall` | `0.85` |
| `citation_compliance` | `0.95` |
| `p95_latency_ms` | `10000` |
| `avg_cost_per_case_usd` | `0.01` |
| `cost_gate_min_llm_call_coverage` | `0.80` |

judge minimum score와 pricing도 같은 파일에서 관리합니다. `cost_gate_min_llm_call_coverage`는 `src/eval/config_models.py::HardGates`의 기본값이며, config에 명시하지 않으면 `0.80`이 적용됩니다. 다중 턴 비용 관측률은 모든 실행 턴을 검사합니다. 준비 턴에 LLM 사용량이 있고 마지막 저장이 deterministic이면 인정하지만, 어느 턴의 사용량이 누락되면 최종 질문의 정상 진단만으로 비용 gate를 활성화하지 않습니다.

### 4.0.1 평가 상태와 release 판정

사례별 결과는 평가 실행 상태와 답변 품질 합격을 분리해 기록합니다.

- `judge_status`: `disabled`(설정으로 비활성화), `not_run`(제품 응답 부재·입력 불완전 등으로 미실행, 사유는 `judge_status_reason`), `failed`(호출 실패 또는 출력 계약 오류), `succeeded`(입력 완비·호출·출력 계약 통과. 품질 합격이 아님), `legacy_unknown`(상태 필드가 없는 과거 결과).
- `eval_validity`: `valid`(현재 정책의 평가가 완료되어 판정 가능, 품질 불합격도 유효한 평가), `incomplete`(필수 입력·단계 부족), `invalid`(judge 호출 실패·출력 계약 오류로 결과 신뢰 불가), `legacy_unknown`.
- `llm_judge_score`: 유효한 judge 0점은 `0`으로, 평가하지 못한 점수는 `null`로 보존합니다. 평가 오류를 점수로 대체하거나 범위 밖 값을 잘라내지 않습니다.
- `judge_pass`: `succeeded` 평가가 전체 점수와 필수 세부 점수 기준을 모두 충족할 때만 `true`. `disabled`·`not_run`·`failed`는 `null`입니다.
- `product_pass`: 완전한 composite 점수가 있을 때 composite 기준 판정. 제품 실행 실패 또는 필수 저장 계약 실패는 점수와 관계없이 `false`, judge 장애로 composite를 계산할 수 없으면 `null`입니다.
- `release_pass`: 평가 유효성(`valid`), 제품 판정(`product_pass`), judge 품질 판정(`judge_pass`), 응답 계약과 저장 계약을 모두 충족할 때만 `true`입니다. `incomplete`·`invalid` 평가는 항상 `false`입니다.

`composite_quality_score`는 judge 점수가 있을 때만 가중합으로 계산합니다. judge 점수가 없으면 남은 규칙 점수를 재정규화해 composite를 만들지 않고 `null`로 남기며, 규칙 원점수는 `rule_scores`·`rule_score_total`에 진단용으로 보존합니다. judge 가용성은 제품 규칙 점수를 바꾸지 않습니다.

모든 카테고리에 `judge_min_score`가 명시적으로 존재합니다(기본 `0.70`, case의 `judge_min_score` 필드가 우선). 세부 점수 기준은 `[judge_min_subscores]`의 `answer_quality`(전 카테고리)와 `groundedness`(근거 기반 답변)이며, 인용을 요구하지 않는 순수 `tool_action` 사례에는 groundedness를 적용하지 않습니다. 이 값들은 인간 평가로 보정된 최적값이 아니라 정책 출발점입니다.

run 집계의 `release_pass_rate` 분모는 실행 대상으로 확정한 전체 사례(`planned_cases`)입니다. composite가 `null`이거나 judge 오류·제품 실패인 사례를 빼지 않으며, 계획 대비 결과 누락·중복·예상외 case_id는 `evaluation_completeness` release gate가 차단합니다. judge 상태별 개수(`judge_succeeded/failed/not_run/disabled_cases`)와 `judge_execution_rate`(judge 필수 대상 중 succeeded 비율)를 별도로 기록하며, 대상이 0개인 비율은 `-`(N/A)로 표시합니다.

### 4.0.2 저장 산출물의 필수 조건

저장 사례는 `save_expectation`으로 결과와 대상을 고정합니다. `expected_tools=["save_text"]`는 호출 기대값이며 저장 성공 기대를 대신하지 않습니다. release 실행에서 저장을 기대하면서 `save_expectation`이 없으면 설정 검증이 거절합니다.

| `outcome` | 필수 확인 |
|---|---|
| `required_success` | 요청·세션·대상 답변과 연결된 verified receipt, 실제 다운로드의 manifest binding·artifact ID·bytes가 모두 일치해야 함 |
| `expected_failure` | `error_codes`에 명시한 실패가 올바르게 전달되고, 성공 receipt와 다운로드 가능한 성공 산출물이 없어야 함 |
| `must_not_execute` | 저장 도구 호출과 저장 receipt가 모두 없어야 함 |

성공·예상 실패 사례는 `target`을 지정합니다. `final_answer`는 실제 최종 답변, `setup_answer`는 `setup_turn_index`로 지정한 준비 턴의 답변이며 인덱스는 0부터 시작합니다. 이전 답변 저장은 준비 턴의 export와 비교하므로 현재 잘못된 답변을 저장하고 같은 hash를 반환해도 통과하지 않습니다. export는 출처·제한을 포함한 UTF-8 BOM bytes이며, 본문 `content_hash`와 별개로 bytes hash와 길이를 확인합니다.

평가기는 일반 `GET /download/{filename}`로 실제 bytes를 읽고 `X-Save-Binding-SHA256`, `X-Artifact-Id`, receipt의 바인딩과 비교합니다. 도구 호출 전에 캡처된 `answer_provenance.save_operation_binding_sha256`도 대조하므로 receipt와 파일이 서로 일치하더라도 다른 저장 작업의 결과로 바뀌었다면 실패합니다. 파일·manifest 부재, 내용 불일치, 만료는 제품 실패이며, timeout·읽기 불가 같은 검증 불가는 `eval_validity=incomplete`로 남깁니다. `CaseResult.save_assessment`에는 `status`(`verified`, `failed`, `unverifiable`, `not_applicable`), `passed`, `failure_codes`, 예상·관측 hash와 크기, artifact ID, 확인 시각을 기록합니다. 개별 사례 후 검사와 실행 끝의 재검사(`phase=case/run_end`)로 뒤의 저장이 앞선 파일을 덮거나 삭제했는지도 확인합니다.

준비 턴에서 성공 receipt를 반환한 저장도 보존 검사 대상입니다. 각 준비 턴의 실제 답변·provenance를 기준으로 확인하고 `CaseResult.setup_save_assessments`에 0부터 시작하는 턴 인덱스별 결과를 남깁니다. 최종 질문의 저장 기대값과는 별개로 사례 후·실행 종료 때 다시 확인하며, 확인 누락·불일치·검증 불가는 해당 사례와 `save_outcome_contract` gate를 실패시킵니다. 마지막 질문의 저장만 성공했다고 앞선 저장 손실을 가리지 않습니다.

실패·검증 불가는 composite나 judge 만점으로 상쇄되지 않습니다. 사례의 `release_pass`와 run의 `save_outcome_contract` gate가 같은 결과를 사용하며, `metrics.save_contract_failures`가 0이어야 이 gate를 통과합니다. 이 gate는 `required_success`·`expected_failure` 및 준비 턴의 성공 저장에 `phase=run_end` 확인 근거도 요구하므로, 파일을 다시 읽지 않는 오프라인 summary는 사례 직후의 성공 결과만으로 release를 통과시키지 않습니다. 구형 receipt와 과거 결과를 새 verified 계약으로 자동 승격하지 않습니다.

### 4.1 참조 연결과 의미적 지지의 구분

| 지표·입력 | 측정 범위 |
|---|---|
| `reference_coverage` | `interaction`을 제외한 실제 내용 단위 중 refs가 있고, 모든 참조가 현재 검색 또는 선택한 선행 답변의 검증된 인용에서 최종 packet으로 연결되는 비율. rule 가중치는 `0.20` |
| `citation_traceability` | 요청한 docs/upload 출처 범위와 최종 packet의 채택 참조를 확인. 현재 검색의 근거는 해당 턴의 도구 실행을 요구하며, 검증된 선행 인용은 재검색 없이 계승 가능 |
| `checks.reference_status` | 런타임의 참조 연결 결과. `resolved`는 의미적 정확성 판정이 아님 |
| `checks.support_status` | `not_evaluated`, `exact_match`, `unsupported` 개수를 별도 집계. `not_evaluated`를 자동으로 0점 처리하지 않음 |
| LLM judge의 groundedness | 실제 표시 본문과 연결된 근거가 설명·해석을 의미적으로 뒷받침하는지 평가 |
| `actions`와 도구 실행 기록 | 실행 의도·결과 진단. 저장 성공은 fixture 대상과 실제 HTTP 산출물 readback까지 일치해야 인정 |

현재 검색에서 packet으로 이어지는 근거는 snapshot과 원문 element가 같고, packet의 문자 범위 또는 표 cell ID가 `observed_hits`의 선택 범위에 포함되는지 확인합니다. synthesis 예산 때문에 범위를 줄이면 새 근거 ID가 생깁니다. 최종 citation은 이렇게 검증한 실제 packet의 ID와 일치해야 하므로, 검색 원문에 있었지만 모델에 제공하지 않은 범위는 인정하지 않습니다. 페이지 bbox가 없더라도 snapshot·요소·선택 범위가 유효하면 원문 연결을 인정합니다.

기존 답변 복사·변환은 debug `answer_provenance.source`의 `ref`, `response_hash`, `citation_ids`로 서버가 선택한 본문과 실제 채택 출처를 확인합니다. 같은 사례·세션의 앞선 턴에서 hash와 전체 citation ID 목록이 모두 일치하는 가장 최근 답변을 찾고, 그 답변의 응답 구조와 출처 연결이 검증되어야 상속을 허용합니다. 상속 가능한 근거도 그 답변의 실제 인용 범위 안에 있는 현재 최종 packet으로 제한합니다. 최종 citation은 이 packet의 정확한 ID와 일치해야 합니다. 단순히 conversation에 등장한 답변이나 이전 턴에서 검색만 한 근거는 합치지 않습니다. 현재 턴의 `observed_hits`, 도구 정밀도·재현율, 원문 복사 감점 입력은 유지하므로 과거 검색 호출이 현재 도구 실행으로 집계되지 않습니다.

`content_hash`는 본문 구조·basis·refs를 포함하지만 citation의 존재 여부나 전달 receipt 전체를 hash하지 않습니다. 따라서 선행 답변의 실제 citation 목록을 함께 비교합니다. 이 연결은 동일한 본문·채택 출처를 확인하며, 같은 내용이 반복된 대화의 정확한 발생 시점이나 설명의 의미적 지지를 증명하지 않습니다. 현재 online 실행에서 provenance 누락·불일치·연결되지 않은 선행 답변은 응답 계약 또는 필수 진단 오류이며, 이전 대화나 검색 원문을 합쳐 성공으로 승격하지 않습니다.

runtime의 exact excerpt 검사는 발췌와 원문의 일치만 보장합니다. 인용된 원문 자체의 진실성이나 답변 전체의 충분함까지 보장하지 않으므로, rule 지표와 judge 결과를 함께 해석해야 합니다.

## 5. 환경 변수 override

`src/eval/main.py`는 아래 환경 변수로 일부 설정을 덮어쓸 수 있습니다. 기본값은 `data/benchmarks/config.toml`, override 정의와 `.env.example` 생성 기준은 `src/infra/settings.py`입니다.

우선순위는 `CLI > .env > OS env > config.toml`입니다.

| 이름 | 기본값 | 설명 |
|---|---|---|
| `BENCHMARK_ENDPOINT` | `http://127.0.0.1:8000` | `/agent/stream`을 붙여 호출할 FastAPI 기본 주소 |
| `JUDGE_MODEL` | config 값 사용 | judge 모델 override |
| `BENCHMARK_JUDGE_ENABLED` | config 값 사용 | judge 사용 여부 override |
| `BENCHMARK_SLACK_ENABLED` | `false` | 실제 목적지 치환·전송 성공 감사 활성화. 서버 전송 차단 스위치는 아님 |
| `BENCHMARK_SLACK_CHANNEL_ID` | 없음 | live channel 케이스 전송용 Slack channel id |
| `BENCHMARK_SLACK_USER_ID` | 없음 | live DM 케이스 전송용 Slack user id |
| `BENCHMARK_SLACK_EMAIL` | 없음 | live DM 케이스 전송용 Slack email |

## 6. 비교 이력 규칙

history 리포터는 다음 조건이 모두 같은 run만 comparable run으로 묶습니다.

- `track`
- `fixtures_path`
- `total_cases`
- `execution_contract_version`: 현재 `shared-client-scenario-v1`
- `measurement_contract_version`: 현재 `attachment-question-scenario-v1`
- `suite_fingerprint`: 준비 턴·첨부 목록을 포함한 사례 내용과 staging에서 실제 읽은 첨부 bytes의 SHA256
- `evaluation_fingerprint`: 채점 계약 버전과 가중치·gate·pricing·judge·timeout 설정, Slack 실행 옵션

같은 경로의 fixture를 덮어써도 내용이나 첨부 bytes가 달라지면 자동 비교되지 않습니다. 새 계약 정보가 일부만 있는 run은 자기 자신만 표시합니다. 계약 정보가 없는 legacy끼리는 이전 경로·사례 수 비교를 유지하지만, 새 실행과 섞지 않습니다. 과거 JSON을 읽을 때 새 계약으로 자동 승격하지 않습니다.

현재 채점 계약은 `verified-save-contract-v3`이며 `summary.json`의 `audit_metrics.scoring_contract_version`에 기록합니다. 이 버전도 `evaluation_fingerprint`에 포함하므로 같은 fixture·설정이라도 이전 채점 결과와 자동 비교하지 않습니다. 실행·측정·채점 계약의 의미를 바꾸면 해당 버전도 갱신해야 합니다. 과거 summary는 당시 값 그대로 읽고 새 채점 버전을 채워 넣지 않습니다. 원격 서버의 모든 설정이나 모델 provider의 변동을 fingerprint가 자동 고정하지는 않습니다. 비교할 변경은 동일한 평가 계약·fixture와 확인된 서버 설정에서 다시 실행합니다. 현재 rule의 `reference_coverage`는 의미적 groundedness를 측정하지 않으며, 이전 스키마·rule의 기록을 새 계약의 품질 상승·하락 근거로 사용하지 않습니다.

## 7. 운영 메모

- benchmark는 현재 `online` 모드만 지원합니다.
- 확정된 첨부는 사례 종료 후 일반 sync API의 clear로 해제하고 staging 파일은 공용 정리 함수를 사용합니다. 질문 결과나 sync 처리 여부가 불확실하면 재전송·세션 폴더 직접 삭제를 하지 않습니다. 남은 세션·파일은 일반 TTL/LRU와 서버 파일 정리 주기에 따르며, 후속 요청이나 서버 재시작의 정리 실행까지 남을 수 있습니다. 정리 요청 실패는 `cleanup_errors`로 기록합니다.
- 공용 클라이언트 평가는 Streamlit 화면 렌더링·브라우저 업로드 시간·다운로드 클릭을 측정하지 않습니다. UI 회귀는 Streamlit AppTest와 첨부 통합 테스트로 별도 확인합니다.
- `history` 명령은 README 안의 자동 갱신 마커를 기준으로 동작하므로, `README.md`의 `## 검증 결과`와 `## 문서` 제목은 유지해야 합니다.
- 공개 release 결과의 정본은 README 요약이고, 비교 추세는 기존 benchmark history SVG에 유지합니다. 별도 결과 문서나 smoke history 파일은 만들지 않습니다.
