# 문서 변환과 OCR

DocuMate는 Docling을 선택적 로컬 변환 경계로 사용합니다. PDF·DOCX·스캔 PDF·단일 프레임 PNG/JPEG/TIFF/WebP/BMP의 본문과 표를 `ParsedDocument`로 매핑하고 기존 세션별 Chroma 검색·인용으로 연결합니다. Python·Notebook 파서와 AST 분석은 유지합니다.

## 실행

Python 3.12 기준으로 검증한 선택 의존성을 설치하고 모델을 준비합니다.

```bash
uv sync --extra docling
uv run --no-sync docling-tools models download \
  layout tableformer rapidocr \
  --output-dir output/docling/models \
  --rapidocr-backend-lang onnxruntime:korean
DOCLING_ENABLED=true uv run --no-sync python -m src.app.service_manager startweb
```

지속적으로 사용하려면 `.env`의 `DOCLING_ENABLED`를 `true`로 설정합니다. `DOCLING_ARTIFACTS_PATH` 기본값은 프로젝트 기준 `output/docling/models`입니다. 설정 전체와 기본 한도는 [런타임 참고](runtime_reference.md#21-애플리케이션-설정)에 있습니다. 선택 의존성을 설치한 뒤 `uv run --no-sync`를 사용하거나 매 실행에서 `--extra docling`을 지정해야 기본 sync가 선택 패키지를 제거하지 않습니다.

기본 엔진은 `rapidocr`, backend는 `onnxruntime`, 언어는 한국어 모델의 한영 문자 집합입니다. CPU 4스레드, 전체 페이지 OCR 강제 비활성화, 표 구조 인식 accurate를 사용합니다. 그림 설명·수식 enrichment·원격 모델 서비스는 사용하지 않습니다. worker는 사전 모델만 사용하고 네트워크로 모델을 내려받지 않습니다. 비교용 `easyocr`를 선택하려면 아래 모델도 준비합니다.

```bash
uv run --no-sync docling-tools models download easyocr \
  --output-dir output/docling/models --easyocr-lang ko --easyocr-lang en
```

문서 변환은 로컬에서 실행됩니다. 기존 검색 임베딩과 답변 생성은 설정된 OpenAI API를 사용하므로 앱 전체가 오프라인인 것은 아닙니다. API 키·원본 문서는 OCR 서비스로 전송하지 않습니다.

## 변환·첨부 계약

`UploadService`는 세션 lock과 epoch/revision/operation ID를 유지한 채 후보 관리 원본을 준비합니다. PDF·DOCX·이미지는 요청별 worker에서 순서대로 변환하고 converter를 재사용합니다. worker에는 활성 세션이나 색인 변경 권한이 없습니다. 모든 원본·변환 결과·청크·임베딩이 준비된 뒤 활성 첨부를 한 번에 교체합니다.

Docling `success` 결과만 수용하며, 오류가 동반된 성공·부분 성공·페이지 누락도 실패입니다. 유효한 문서가 전혀 검색 가능한 내용을 만들지 못하면 거부합니다. 파서의 처리 완료와 OCR 내용의 정확성은 구분합니다. 페이지 수가 맞아도 글자나 문단이 잘못 인식될 수 있어 PDF·이미지의 인용에는 실제 OCR 설정에 맞는 제한을 남깁니다.

기본 파일 한도는 10개·파일당 10 MiB·활성 합계 50 MiB입니다. 추가 한도는 PDF 30페이지, 이미지 4천만 픽셀, DOCX 압축 해제 50 MiB, 변환 결과 합계 16 MiB, 문서 청크 2천개입니다. 표의 한 행과 필수 헤더를 합친 청크가 8천 UTF-8 bytes를 넘으면 문서 한도 오류로 거부하며 셀을 잘라 넣지 않습니다. 애니메이션·다중 프레임 이미지는 페이지별 파일이나 PDF로 변환해야 합니다.

변환 프로세스는 기본 90초, RSS 감시 기준 4096 MiB이며 초과 시 자식 프로세스까지 종료합니다. RSS 감시는 OS의 절대 메모리 상한은 아닙니다. 후보 처리의 기본 기한은 150초이며 임베딩 배치마다 남은 시간으로 provider timeout을 정합니다. 기한이 지나면 후보를 커밋하지 않습니다. 현재 클라이언트의 180초 read timeout에는 lock 대기·통신도 영향을 주므로 전체 요청이 항상 180초 안에 끝난다는 보장은 없습니다. 네트워크 단절 시 manifest 또는 같은 operation ID로 결과를 확인합니다.

실패 후보는 새 원본과 색인만 폐기하고 기존 첨부를 유지합니다. 관리 원본은 `uploads/<session>/objects`, 변환 임시 파일은 `uploads/<session>/conversions`에 둡니다. 종료·시간 초과 때 worker와 임시 파일을 회수하며, 다음 세션 요청에서도 남은 변환 작업 공간을 정리합니다. worker는 모델을 읽기 전부터 소유 API 프로세스의 PID와 생성 시각을 감시하고, 부모 강제 종료나 PID 재사용을 감지하면 자신과 자식 프로세스를 종료합니다. Windows 가상환경의 중간 실행 프로세스가 있어도 실제 소유 프로세스를 기준으로 감시합니다. 새 형식 접수 기능을 꺼도 저장된 원본의 소유권 판정·정리는 유지합니다.

## 문단·표와 페이지 인용

DOCX 본문은 굵게·기울임·하이퍼링크 등 서식이 바뀌어도 원래 문단 단위로 추출합니다. 원본 run과 링크 표시문을 공백 제거 전에 연결하므로 단어 내부의 서식 경계, 연속 공백, 탭, 문단 안 줄바꿈을 보존하며 인접한 별도 문단은 합치지 않습니다. 표의 rich cell도 Markdown 장식이나 HTML escape를 넣지 않은 표시문으로 저장합니다. 이 동작은 고정된 Docling Word 백엔드를 확장하므로 의존성 업그레이드 시 실제 DOCX 회귀 테스트를 실행해야 합니다. 추출 정책 변경은 어댑터 버전에 반영해 이전 변환·임베딩 캐시와 구분합니다.

네이티브 PDF는 `PdfBackendOptions(enforce_same_font=False)`로 읽습니다. 글꼴 변경만으로 붙어 있는 글자를 나눴다가 조립 과정에서 공백을 삽입하는 문제를 방지하며, 실제 글자 위치와 레이아웃에 따른 셀 구분은 유지합니다. PDF의 탭 위치·줄바꿈·문단 구분은 DOCX처럼 명시적인 텍스트 구조가 아니므로 같은 문서처럼 보여도 요소 경계가 달라질 수 있습니다. DOCX의 `Code` 스타일도 별도 Docling 코드 처리 규칙을 따르며 행 끝 공백은 제거됩니다.

표의 셀·행/열 병합·열/행/구역 헤더를 저장합니다. 청크 metadata에는 snapshot/element ID와 `cell_ids_json`만 넣고 표 전체는 source registry에 한 번 보존합니다. `page_content`는 선택한 셀의 원문으로 생성하고 검색 후 복원 시 다시 일치 여부를 검사합니다. 생성 단계에서는 검색된 셀 집합 안에서 필요한 행과 헤더를 선택합니다. 최소 단위가 근거 문자 예산에 들어가지 않으면 그 요구를 충족됐다고 표시하지 않습니다.

페이지 번호·bbox·좌표 원점·페이지 크기는 제공된 값만 보존합니다. 셀의 페이지가 불명확한 다중 페이지 표는 표 전체 페이지 위치로 표시하며, DOCX의 가상 페이지 번호를 만들지 않습니다. bbox는 요소 수준의 위치이며 선택한 글자 수에 비례해 잘라내지 않습니다. 생성·UI·텍스트 내보내기는 같은 위치 선택 함수를 사용합니다.

`exact_match`는 보관된 추출문과 답변 발췌의 일치를 검사합니다. OCR이 시각적 원본을 정확하게 읽었는지를 증명하지 않습니다. PDF에 코드처럼 보이는 내용이 있어도 Python 구현 전체를 AST로 분석했다고 판단하지 않습니다.

## 변환·임베딩 캐시

캐시는 세션별 `uploads/<session>/cache` 아래에만 저장합니다. 기본 64 MiB를 변환과 임베딩에 절반씩 할당하며 생성 시점 기준 TTL은 1800초입니다. 조회는 LRU 순서를 갱신하지만 TTL을 연장하지 않습니다. 손상·기한 만료·용량 초과 항목은 제거하거나 miss로 처리하며, 캐시 쓰기 실패는 정상 변환·임베딩 결과를 버리지 않습니다.

변환 키에는 원본 SHA-256·형식·어댑터 버전·Docling/core/parser/OCR/runtime 버전·로컬 모델 파일 내용 해시·OCR 엔진/언어/모드·표 옵션을 반영합니다. 운영 시간 한도는 추출 결과 키에 넣지 않고 캐시 조회 때 현재 페이지·입력 확장·출력 크기 한도를 다시 검사합니다. 모델 디렉터리의 파일 추가·변경도 무효화를 일으킵니다.

캐시에는 재사용할 문서 요소와 추출 정보를 보관하며 이전 파일의 snapshot ID·file ID를 재사용하지 않습니다. 조회할 때 현재 파일명·source URI와 원본 bytes로 snapshot을 다시 만들고 새 문서 객체를 반환합니다. 따라서 동일 원본을 다른 이름으로 첨부해도 별도 인용이며 이전 색인의 `release()`가 현재 문서를 지우지 않습니다.

임베딩 키에는 정확한 청크 텍스트 해시, 원본 해시, 파서 버전·설정, 청킹 버전·크기·중첩, 임베딩 모델·요청 차원·API base·SDK 버전을 포함합니다. 디스크에는 텍스트·파일 식별자·API 키를 저장하지 않고 namespace 해시·벡터·유효 기간·checksum만 저장합니다. query 임베딩은 캐시하지 않습니다. 호스팅 서비스가 동일 모델 이름 아래의 내부 모델을 변경하는 것은 감지할 수 없으므로 TTL과 캐시 해제를 사용합니다.

개별 삭제·교체 시 캐시는 세션 안에서 TTL·용량 한도까지 남아 재첨부에 사용할 수 있습니다. 전체 첨부 해제와 세션 종료·reset은 캐시도 지웁니다. 첫 첨부 실패로 활성 파일이 없어도 명시적 clear는 남은 캐시를 지웁니다. 이전 인덱스 정리는 캐시를 삭제하지 않으며, 이미 응답에 보존한 인용은 캐시·원본 삭제와 독립적입니다.

## OCR 선택 근거

2026-09-13 Windows/Python 3.12, RAM 15.9 GiB 환경에서 Docling 2.126.0/core 2.96.0, EasyOCR 1.7.2와 RapidOCR 3.9.2를 CPU 4스레드로 비교했습니다. 모델은 사전 다운로드하고 OCR은 오프라인으로 실행했습니다. 각 형식·엔진은 2회 실행했으며 아래 시간은 두 번째 변환 시간입니다. 동일 GPU 성능을 측정한 결과가 아닙니다.

| 직접 작성한 150 DPI 한영 표본 | EasyOCR 전체/한글 CER | RapidOCR 전체/한글 CER | EasyOCR/RapidOCR 시간 |
|---|---|---|---|
| 스캔 PNG | 18.42% / 51.85% | 0.33% / 0% | 15.07 / 8.23초 |
| 스캔 PDF | 11.18% / 24.69% | 2.63% / 2.47% | 17.33 / 9.64초 |
| 네이티브 PDF 2페이지 | 0% / 0% | 0% / 0% | 10.80 / 6.65초 |
| DOCX | 0% / 0% | 0% / 0% | 0.153 / 0.126초 |

CER은 정규화한 정답 문자 대비 편집 거리이며 누락도 계산합니다. 구조화 표 셀의 완전 일치 회수율은 스캔 PNG/PDF에서 EasyOCR 100%/100%, RapidOCR 88.9%/77.8%였습니다. RapidOCR는 한글 본문 누락이 적지만 일부 표제어를 잘못 읽었습니다. 엔진 프로세스 최대 RSS는 약 994/1472 MiB, 첫 PNG 변환은 모델 초기화 포함 55.9/35.2초였습니다. 한 종류의 합성 표본에 대한 수치이므로 일반 문서 정확도로 일반화하지 않습니다.

추가로 [보관 팀 문서](../archive/team_docs/Langchain_Project_Team_3.pdf)의 물리적 7페이지(인쇄 번호 6)를 150 DPI 이미지로 렌더링했습니다. 검은 배경·도형·두 단의 실문서에서 화면과 대조한 한글 문장 5개 중 EasyOCR는 0개, RapidOCR는 3개를 완전히 회수했습니다. 날짜·기술명 5개는 둘 다 회수했습니다. 전체 페이지 강제 OCR도 문장 회수율을 개선하지 못했고 RapidOCR의 날짜·기술명 회수는 3/5로 떨어졌습니다. 따라서 RapidOCR의 기본 영역 OCR을 선택합니다. 이 실자료는 일부 문장만 수동 확인했으므로 전체 CER·표 정확도를 산출하지 않았습니다.

이번 실행의 원시 결과는 `output/docling/benchmark/benchmark.json`, `output/docling/real_sample/reference.json`, `output/docling/real_sample/benchmark/benchmark.json`, `output/docling/real_sample/full_page/benchmark.json`에 있습니다. 출력 디렉터리는 Git에 포함하지 않습니다. 실제 슬라이드의 문단 누락이 남아 있으므로 중요한 수치와 인용은 원본을 함께 확인해야 합니다.

## 검증 재현

```bash
LIVE_TEST=false uv run --no-sync pytest -q
LIVE_TEST=false uv run --no-sync pytest tests/tools/test_docx_conversion.py -q
RUN_DOCLING_TESTS=1 LIVE_TEST=false HF_HUB_OFFLINE=1 \
  uv run --no-sync pytest tests/tools/test_docling_adapter.py tests/web/test_document_pipeline_live.py -q
uv run --no-sync python script/benchmark_document_ocr.py
uv run --no-sync python script/sync_env_example.py --check
uv run --no-sync python script/check_encoding.py
```

일반 테스트는 실제 임시 파일·Chroma와 결정적인 외부 임베딩/LLM 경계를 사용합니다. `test_docx_conversion.py`는 합성 Docling 객체를 매핑하는 단위 테스트와 달리 임시 DOCX bytes를 실제 Word 백엔드로 변환합니다. Docling 선택 의존성은 필요하지만 PDF/OCR 추론 모델이나 외부 API는 사용하지 않습니다.

`RUN_DOCLING_TESTS=1` 검증은 실제 문서 변환 worker와 사전 모델을 사용하며 유료 API를 호출하지 않습니다. PDF·DOCX·스캔 PDF·PNG의 실제 변환에서 시작해 HTTP 첨부·검색·표/페이지 인용·다른 이름으로 캐시 재사용·원본 삭제 후 이전 인용까지 확인합니다. 혼합 서식 fixture는 문단별 검색과 전체·부분 문자 범위 인용도 확인합니다. 모델이 필요한 테스트는 명시적으로 활성화한 환경에서만 실행합니다.

범용 문서 전체 요약, PDF 페이지 뷰어, 다중 프레임 이미지, 손글씨 정확도 보장, PPTX/XLSX, 비동기 작업 API는 현재 지원 범위가 아닙니다. 출처 보존과 처리 성공은 OCR 내용의 무오류 보장과 구분합니다.
