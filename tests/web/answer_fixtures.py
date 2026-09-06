from src.core.answer_schema import ActionReceipt, AnswerDocument, CodeBlock, ContentUnit, ParagraphBlock, ResponseIssue, finalize_answer, text_document
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import build_evidence


def answer_response(text: str = "응답"):
    return finalize_answer(text_document(text), [])


def response_payload(text: str = "응답"):
    return answer_response(text).model_dump(mode="json")


def cited_response():
    source = "def value():\n    return 3\n"
    snapshot = build_snapshot(source_uri="upload:///example.py", title="example.py", media_type="text/x-python", source_type="upload", content=source, parser="python", parser_version="1")
    element = DocumentElement(element_id="function", kind="code", text=source, language="python", anchors=[SourceAnchor(kind="code", line_start=7, line_end=8)])
    evidence = build_evidence(snapshot=snapshot, element=element)
    document = AnswerDocument(blocks=[
        ParagraphBlock(content=[ContentUnit(text="함수는 3을 반환합니다.", basis="source", refs=[evidence.id])]),
        CodeBlock(language="python", content=ContentUnit(text=source, basis="excerpt", refs=[evidence.id])),
    ])
    return finalize_answer(document, [evidence], actions=[ActionReceipt(kind="save_text", status="success", file_path="output/result.txt")], issues=[ResponseIssue(code="scope", message="이 함수의 범위만 확인했습니다.")])
