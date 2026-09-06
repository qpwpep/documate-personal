from src.core.answer_schema import AnswerDocument, finalize_answer, text_document
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence


def plain_response(text: str | list[str]) -> dict:
    paragraphs = [text] if isinstance(text, str) else text
    document = AnswerDocument.model_validate({
        "blocks": [
            {"type": "paragraph", "content": [{"text": paragraph, "basis": "interaction", "refs": []}]}
            for paragraph in paragraphs
        ],
    })
    return finalize_answer(document, []).model_dump(mode="json")


def source_evidence(*, official: bool = True, text: str = "공식 설명", source_uri: str | None = None):
    snapshot = build_snapshot(
        source_uri=source_uri or ("https://numpy.org/doc/stable/" if official else "uploads/demo/sample.ipynb"),
        title="NumPy Docs" if official else "Notebook",
        media_type="text/plain",
        source_type="official" if official else "upload",
        content=text,
        parser="eval-fixture",
        parser_version="1",
    )
    element = DocumentElement(
        element_id="body",
        kind="paragraph" if official else "code",
        text=text,
        anchors=[SourceAnchor(kind="web" if official else "notebook", start=0, end=len(text))],
    )
    return build_evidence(snapshot=snapshot, element=element)


def source_hit(*, official: bool = True, text: str = "공식 설명", source_uri: str | None = None) -> SearchHit:
    return SearchHit(evidence=source_evidence(official=official, text=text, source_uri=source_uri), score=RetrievalScore(metric="rank", raw=1, direction="lower"), rank=1)


def comparison_response() -> dict:
    official = source_evidence()
    upload = source_evidence(official=False, text="업로드 비교")
    document = AnswerDocument.model_validate({
        "blocks": [
            {"type": "paragraph", "content": [{"text": "공식 설명", "basis": "source", "refs": [official.id]}]},
            {"type": "paragraph", "content": [{"text": "업로드 비교", "basis": "source", "refs": [upload.id]}]},
        ],
    })
    return finalize_answer(document, [official, upload]).model_dump(mode="json")
