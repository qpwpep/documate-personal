"""Convert local files into the application's source-preserving document model.

Docling imports stay inside the optional conversion boundary. The worker owns
process isolation and deadlines; this module accepts only verified local bytes.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from typing import Any

from src.core.documents import DocumentElement, ParsedDocument, SourceAnchor, TableCell, TableData, build_snapshot
from src.core.table_selection import table_excerpt
from src.core.uploads import UploadRecord
from src.core.upload_formats import DOCUMENT_MEDIA_TYPES
from src.infra.document_ingestion import ConversionPolicy, IngestionError, read_verified_upload, validate_document_input


def _value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _fail(code: str, message: str, file: UploadRecord, *, retryable: bool = False) -> IngestionError:
    return IngestionError(code, message, file_name=file.name, retryable=retryable)


@lru_cache(maxsize=2)
def _converter(policy: ConversionPolicy):
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import EasyOcrOptions, OcrMode, PdfPipelineOptions, RapidOcrOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption, WordFormatOption

    artifacts = Path(policy.artifacts_path).resolve()
    if not policy.artifacts_path or not artifacts.is_dir():
        raise IngestionError("DOCUMENT_CONVERTER_UNAVAILABLE", "문서 변환 모델을 먼저 준비해 주세요.")
    if policy.ocr_engine == "easyocr":
        ocr = EasyOcrOptions(lang=list(policy.ocr_languages), download_enabled=False,
                            mode=OcrMode.FULL_PAGE if policy.force_full_page_ocr else OcrMode.DEFAULT)
    else:
        # The Korean recognizer includes Latin characters. Passing ['ko', 'en']
        # directly would silently discard the second entry in RapidOCR.
        languages = set(policy.ocr_languages)
        if languages.issubset({"ko", "en", "korean"}) and languages.intersection({"ko", "korean"}):
            lang = "korean"
        elif languages == {"en"}:
            lang = "en"
        else:
            raise IngestionError("DOCUMENT_CONVERTER_UNAVAILABLE", "RapidOCR 언어 조합을 지원하지 않습니다.")
        ocr = RapidOcrOptions(lang=[lang], backend="onnxruntime",
                             mode=OcrMode.FULL_PAGE if policy.force_full_page_ocr else OcrMode.DEFAULT)
    options = PdfPipelineOptions(
        artifacts_path=artifacts, do_ocr=policy.do_ocr, ocr_options=ocr,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=4),
        document_timeout=policy.timeout_seconds, do_table_structure=True,
        do_picture_description=False, do_picture_classification=False,
        do_code_enrichment=False, do_formula_enrichment=False,
        generate_page_images=False, generate_picture_images=False,
        enable_remote_services=False, ocr_batch_size=1, layout_batch_size=1, table_batch_size=1,
        queue_max_size=2,
    )
    options.table_structure_options.mode = TableFormerMode(policy.table_mode)
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.IMAGE],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options),
                        InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
                        InputFormat.DOCX: WordFormatOption()},
    )


def convert_file(file: UploadRecord, policy: ConversionPolicy, *, parser_config: dict | None = None) -> ParsedDocument:
    """Convert one verified upload; incomplete output is never a successful snapshot."""
    raw = read_verified_upload(file)
    validate_document_input(file, raw, policy)
    if Path(file.name).suffix.lower() not in DOCUMENT_MEDIA_TYPES:
        raise _fail("DOCUMENT_INVALID", "지원하지 않는 문서 형식입니다.", file)
    try:
        from docling.datamodel.base_models import DocumentStream

        converter = _converter(policy)
        result = converter.convert(DocumentStream(name=file.name, stream=BytesIO(raw)), raises_on_error=False,
                                   max_num_pages=policy.max_pdf_pages, max_file_size=file.size_bytes)
        config = parser_config if parser_config is not None else policy.extraction_options()
        return _map_result(result, file, config, raw, ocr_enabled=policy.do_ocr)
    except IngestionError as exc:
        if not exc.file_name:
            exc.file_name = file.name
        raise
    except (ImportError, PackageNotFoundError) as exc:
        raise _fail("DOCUMENT_CONVERTER_UNAVAILABLE", "문서 변환용 선택 의존성을 설치해 주세요.", file) from exc
    except (FileNotFoundError, OSError) as exc:
        raise _fail("DOCUMENT_CONVERTER_UNAVAILABLE", "문서 변환 모델 또는 실행 환경을 확인해 주세요.", file) from exc
    except Exception as exc:
        raise _fail("DOCUMENT_ADAPTER_ERROR", "문서를 안전하게 변환하지 못했습니다.", file) from exc


def map_docling_document(result: Any, file: UploadRecord, parser_config: dict) -> ParsedDocument:
    """Map an actual Docling result without losing provenance or table roles."""
    return _map_result(result, file, parser_config, read_verified_upload(file))


def _page_anchor(doc: Any, *, page_no: int, bbox: Any = None, kind: str = "page") -> SourceAnchor:
    page = doc.pages.get(page_no)
    size = getattr(page, "size", None)
    values: dict[str, Any] = dict(kind=kind, page_no=page_no, precision="page",
                                  coordinate_space="points")
    if size is not None:
        values.update(page_width=float(size.width), page_height=float(size.height))
    if bbox is not None:
        values.update(bbox=(float(bbox.l), min(float(bbox.t), float(bbox.b)),
                            float(bbox.r), max(float(bbox.t), float(bbox.b))), precision="element",
                      coordinate_origin="bottom_left" if _value(bbox.coord_origin) == "BOTTOMLEFT" else "top_left")
    return SourceAnchor(**values)


def _anchors(doc: Any, item: Any, *, kind: str = "page") -> list[SourceAnchor]:
    return [_page_anchor(doc, page_no=location.page_no, bbox=location.bbox, kind=kind)
            for location in getattr(item, "prov", [])]


def _ancestors(doc: Any, item: Any) -> list[Any]:
    ancestors, seen = [], {item.self_ref}
    parent = getattr(item, "parent", None)
    while parent is not None:
        node = parent.resolve(doc)
        if node.self_ref in seen:
            raise ValueError("Docling hierarchy contains a cycle")
        seen.add(node.self_ref)
        ancestors.append(node)
        parent = getattr(node, "parent", None)
    return ancestors


def _table(doc: Any, item: Any) -> tuple[TableData, dict[str, list[str]]]:
    cells, roles = [], {"column": [], "row": [], "section": []}
    known_pages = {prov.page_no for prov in item.prov}
    for index, cell in enumerate(item.data.table_cells):
        cell_id = f"{item.self_ref}/cells/{index}"
        for name, attribute in (("column", "column_header"), ("row", "row_header"), ("section", "row_section")):
            if getattr(cell, attribute):
                roles[name].append(cell_id)
        text = cell.text
        if getattr(cell, "ref", None) is not None:
            from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

            text = MarkdownDocSerializer(doc=doc).serialize(item=cell.ref.resolve(doc)).text
        # A Docling cell rectangle has no page field. It identifies a page only
        # when the owning table has exactly one known page.
        locations = []
        if len(known_pages) == 1:
            locations = [_page_anchor(doc, page_no=next(iter(known_pages)), bbox=cell.bbox, kind="table")]
        cells.append(TableCell(cell_id=cell_id, row=cell.start_row_offset_idx, col=cell.start_col_offset_idx,
                               row_span=cell.row_span, col_span=cell.col_span, text=text,
                               is_header=cell.column_header or cell.row_header or cell.row_section, anchors=locations))
    return TableData(cells=cells), roles


def _map_result(result: Any, file: UploadRecord, parser_config: dict, raw: bytes,
                *, ocr_enabled: bool | None = None) -> ParsedDocument:
    from docling_core.types.doc import ContentLayer, GroupItem, PictureItem, TableItem

    status = _value(result.status)
    errors = list(result.errors)
    source_pages = getattr(result.input, "page_count", 0)
    page_limit = getattr(getattr(result.input, "limits", None), "max_num_pages", None)
    if page_limit is not None and source_pages > page_limit:
        raise _fail("DOCUMENT_LIMIT_EXCEEDED", "문서 페이지 수가 변환 한도를 초과했습니다.", file)
    if any(_value(getattr(error, "category", "")) == "timeout" for error in errors):
        raise _fail("DOCUMENT_PROCESSING_TIMEOUT", "문서 변환 시간이 제한을 초과했습니다.", file, retryable=True)
    if status == "partial_success" or (status == "success" and errors):
        raise _fail("DOCUMENT_PARTIAL_CONVERSION", "문서 일부를 변환하지 못해 검색에 반영하지 않았습니다.", file)
    if status != "success":
        raise _fail("DOCUMENT_INVALID", "문서를 읽지 못했습니다. 파일 형식과 손상 여부를 확인해 주세요.", file)
    doc = result.document
    if source_pages and set(doc.pages) != set(range(1, source_pages + 1)):
        raise _fail("DOCUMENT_PARTIAL_CONVERSION", "문서 페이지 일부가 누락되어 검색에 반영하지 않았습니다.", file)
    elements: list[DocumentElement] = []
    by_id: dict[str, DocumentElement] = {}
    heading_stack: list[tuple[int, DocumentElement]] = []
    quality: list[str] = []
    if ocr_enabled is None:
        options = parser_config.get("extraction_options", parser_config)
        ocr_enabled = options.get("do_ocr")
    if Path(file.name).suffix.lower() != ".docx":
        if ocr_enabled is True:
            quality.append("OCR을 포함한 자동 변환 결과에는 글자 인식 오류나 내용 누락이 있을 수 있습니다.")
        elif ocr_enabled is False:
            quality.append("OCR을 사용하지 않아 이미지 안의 문자는 검색 대상에 포함되지 않을 수 있습니다.")
    for item, _depth in doc.iterate_items(with_groups=True, included_content_layers={ContentLayer.BODY}):
        if isinstance(item, GroupItem):
            continue
        ancestors = _ancestors(doc, item)
        # Rich table cell descendants are represented by their owning cell,
        # never indexed again as unrelated standalone paragraphs.
        if any(isinstance(parent, TableItem) for parent in ancestors):
            continue
        label = _value(item.label)
        parent = next((by_id[node.self_ref] for node in ancestors if node.self_ref in by_id), None)
        text = str(getattr(item, "text", ""))
        metadata: dict[str, Any] = {"docling_label": label}
        table = None
        level = None
        if label in {"title", "section_header"}:
            kind = "heading"
            level = int(getattr(item, "level", 1))
            stack_level = 0 if label == "title" else level
            while heading_stack and heading_stack[-1][0] >= stack_level:
                heading_stack.pop()
        elif isinstance(item, TableItem):
            kind = "table"
            table, roles = _table(doc, item)
            metadata["table_header_cell_ids"] = roles
            text = table_excerpt(table, [cell.cell_id for cell in table.cells])
        elif isinstance(item, PictureItem):
            kind = "image"
            text = ""
            if "그림의 시각적 내용은 추출하지 않았습니다." not in quality:
                quality.append("그림의 시각적 내용은 추출하지 않았습니다.")
        elif label == "code":
            kind = "code"
        elif label == "list_item":
            kind = "list"
        else:
            kind = "paragraph"
            if label == "formula" and "수식 전용 인식을 수행하지 않았습니다." not in quality:
                quality.append("수식 전용 인식을 수행하지 않았습니다.")
        if parent is None and heading_stack:
            parent = heading_stack[-1][1]
        heading_path = list(parent.heading_path) if parent else []
        if parent is not None and parent.kind == "heading":
            heading_path.append(parent.text)
        element = DocumentElement(element_id=item.self_ref, kind=kind, text=text, parent_id=parent.element_id if parent else None,
                                  order=len(elements), heading_level=level, heading_path=heading_path,
                                  language=_value(item.code_language) if kind == "code" and getattr(item, "code_language", None) else None,
                                  table=table, anchors=_anchors(doc, item, kind="table" if kind == "table" else "page"), metadata=metadata)
        elements.append(element)
        by_id[element.element_id] = element
        if kind == "heading":
            heading_stack.append((0 if label == "title" else level, element))
    if not any(element.text.strip() for element in elements if element.kind != "image"):
        raise _fail("DOCUMENT_NO_SEARCHABLE_CONTENT", "검색할 수 있는 문서 본문을 찾지 못했습니다.", file)
    if source_pages:
        elements[0].metadata["document_page_count"] = source_pages
    snapshot = build_snapshot(source_uri=file.source_uri, title=file.name,
                              media_type=DOCUMENT_MEDIA_TYPES[Path(file.name).suffix.lower()], source_type="upload",
                              content=raw, parser="docling", parser_version=version("docling"),
                              parser_config=parser_config, quality_issues=quality)
    return ParsedDocument(snapshot=snapshot, elements=elements)
