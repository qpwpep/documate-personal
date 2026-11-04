"""Short source pointers for model views, without changing owned answer snapshots."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from src.core.answer_schema.models import AnswerDocument, iter_content_units


def source_reference_aliases(source_ids: Iterable[str]) -> dict[str, str]:
    """Assign one alias to each exact source identity in stable packet order."""
    return {source_id: f"e{index}" for index, source_id in enumerate(dict.fromkeys(source_ids), 1)}


def remap_document_source_references(
    document: AnswerDocument, references: Mapping[str, str], *, mark_unavailable: bool = False,
) -> AnswerDocument:
    """Copy and remap only typed citation pointers; quoted IDs remain literal text."""
    projected = document.model_copy(deep=True)
    for _path, unit in iter_content_units(projected):
        unit.refs = [references.get(reference, f"unavailable:{reference}" if mark_unavailable else reference)
                     for reference in unit.refs]
        if unit.source_location is not None:
            reference = unit.source_location.ref
            unit.source_location = unit.source_location.model_copy(update={
                "ref": references.get(reference, f"unavailable:{reference}" if mark_unavailable else reference),
            })
    return projected
