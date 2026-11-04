"""Approved complementary source ranges must survive the packet budget."""

import json
from pathlib import Path

from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.runtime.nodes.synthesis.budgets import RetrievedEvidenceBudget
from src.runtime.nodes.synthesis.evidence_selection import prepare_evidence_candidates, select_evidence_packet


def document(text, *, uri="https://scikit-learn.org/stable/precision.html", title="Precision"):
    snapshot = build_snapshot(source_uri=uri, title=title, media_type="text/plain",
        source_type="official", content=text, parser="official_html", parser_version="1")
    return build_evidence(snapshot=snapshot, element=DocumentElement(element_id="body", kind="paragraph", text=text))


def test_captured_csv_windows_preserve_dtype_and_column_selection_descriptions():
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/evidence/read_csv_candidate_windows.json").read_text(
        encoding="utf-8"))
    task = RetrievalTask.model_validate(fixture["task"])
    originals = [document(value["text"], uri=value["source_uri"], title=value["title"])
                 for value in fixture["documents"]]
    hits = []
    for window in fixture["windows"]:
        original = originals[window["document"]]
        hits.append(SearchHit(evidence=build_evidence(snapshot=original.snapshot, element=original.element,
                start=window["start"], end=window["end"]), rank=window["rank"], requirement_id=task.requirement_id,
            score={"metric": "retrieval_rank", "direction": "lower", "raw": float(window["rank"])}))

    candidates = prepare_evidence_candidates(user_input=task.query, hits=hits,
        planner_output=PlannerOutput(use_retrieval=True, tasks=[task]),
        budget=RetrievedEvidenceBudget(1800, 6000, 8))

    assert len(candidates) <= 12
    assert all(len(hit.evidence.excerpt) <= 1800 for hit in candidates)
    for candidate in candidates:
        item = candidate.evidence
        assert item.excerpt == item.element.text[item.selection.start:item.selection.end]
        assert any(hit.evidence.snapshot.snapshot_id == item.snapshot.snapshot_id
                   and hit.evidence.selection.start <= item.selection.start
                   and item.selection.end <= hit.evidence.selection.end for hit in hits)
    assert any("Use str or object together with suitable na_values settings" in hit.evidence.excerpt
               for hit in candidates)
    assert any("Subset of columns to select, denoted either by column labels or column indices." in hit.evidence.excerpt
               for hit in candidates)


def test_requested_definition_is_not_displaced_by_shorter_related_note_and_upload_fragments():
    definition = (
        "Precision is the number of true positives divided by the sum of true positives and false positives.\n\n"
        "High precision means that few returned positive predictions are false positives.\n\n"
    )
    recall = "Recall divides true positives by the sum of true positives and false negatives.\n"
    original = document(definition + recall)
    formula = build_evidence(snapshot=original.snapshot, element=original.element, end=len(definition))
    note = build_evidence(snapshot=original.snapshot, element=original.element, start=len(definition))
    docs_task = RetrievalTask(requirement_id="definition", route="docs", query="Explain the official definition of precision", k=5,
        requirement={"library": "scikit-learn", "evidence_needs": ["explanation"]})
    upload_task = RetrievalTask(requirement_id="card", route="upload", query="Read the supplied model card", k=5,
        requirement={"file_ids": ["card"]})
    snapshot = build_snapshot(source_uri="upload:///card", title="Model card", media_type="text/plain",
        source_type="upload", content="model card", parser="test", parser_version="1")
    uploads = [build_evidence(snapshot=snapshot, element=DocumentElement(element_id=f"label-{index}", kind="paragraph",
        text=text, metadata={"file_id": "card"})) for index, text in enumerate(
            ["Model card", "True positives", "18", "False positives", "6", "Sample", "Synthetic"])]
    budget = RetrievedEvidenceBudget(400, 1200, 8)

    packet, associations = select_evidence_packet([note, formula, *uploads], budget=budget,
        requirements_by_evidence={note.id: [docs_task], formula.id: [docs_task],
            **{item.id: [upload_task] for item in uploads}},
        semantic_support={item.id: {"definition": {"relevance", "need:explanation"}} for item in [note, formula]})

    assert formula.id in associations
    assert any(definition in item.excerpt for item in packet)
    assert any(item.route == "upload" for item in packet)
    assert len(packet) <= budget.max_items


def test_identical_captures_at_alias_urls_do_not_displace_complementary_description():
    introduction = "Fields can have default values and metadata.\n\n"
    behavior = "Defaults are not validated unless validate_default is enabled.\n"
    text = introduction + behavior
    original = document(text, uri="https://docs.pydantic.dev/2.11/concepts/fields", title="Fields")
    alias = document(text, uri="https://pydantic.dev/docs/validation/2.11/concepts/fields", title="Fields")
    first = build_evidence(snapshot=original.snapshot, element=original.element, end=len(introduction))
    duplicate = build_evidence(snapshot=alias.snapshot, element=alias.element, end=len(introduction))
    detail = document(behavior, uri="https://docs.pydantic.dev/2.11/concepts/validation", title="Validation")
    entry = RetrievalTask(requirement_id="defaults", route="docs", query="Describe default validation", k=5,
        requirement={"library": "pydantic", "version": "2.11", "evidence_needs": ["explanation"]})
    candidates = [first, duplicate, detail]

    packet, associations = select_evidence_packet(candidates, budget=RetrievedEvidenceBudget(200, 300, 2),
        requirements_by_evidence={item.id: [entry] for item in candidates},
        semantic_support={item.id: {entry.requirement_id: {"relevance", "need:explanation"}} for item in candidates})

    assert detail.id in associations
    assert sum(item.excerpt == introduction for item in packet) == 1
    assert len(packet) == 2


def test_compact_packet_preserves_separate_shape_and_copy_descriptions():
    shape = "shape: tuple\n\nOne dimension may be -1; its value is inferred from the array size.\n\n"
    order = "order: str\n\nThe order determines how elements are read and written.\n\n"
    copy = "copy: bool or None\n\nA view is returned when possible, otherwise a copy is made.\n"
    original = document(shape + order + copy, uri="https://numpy.org/doc/stable/reference/generated/numpy.reshape.html",
        title="numpy.reshape")
    entry = RetrievalTask(requirement_id="reshape", route="docs", query="Explain shape=-1 and whether reshape guarantees no copy", k=5,
        requirement={"library": "numpy", "symbols": ["numpy.reshape"], "aspects": ["-1"],
            "evidence_needs": ["parameters", "explanation"]})
    budget = RetrievedEvidenceBudget(100, 300, 4)

    packet, associations = select_evidence_packet([original], budget=budget,
        requirements_by_evidence={original.id: [entry]},
        semantic_support={original.id: {entry.requirement_id: {"relevance", "aspect:-1", "need:parameters", "need:explanation"}}})

    assert any("One dimension may be -1" in item.excerpt for item in packet)
    assert any("A view is returned when possible, otherwise a copy is made." in item.excerpt for item in packet)
    assert original.id not in associations
    assert all(len(item.excerpt) <= budget.max_excerpt_chars for item in packet)
    assert sum(len(item.excerpt) for item in packet) <= budget.max_total_excerpt_chars


def test_partly_overlapping_parameter_description_precedes_later_optional_examples():
    """The live merge packet lost its approved indicator range to nonoverlapping examples."""
    introduction = "merge(left, right, how='left', indicator=False)\nPreserve the left table's keys.\n\n"
    keys = "on: Column names found in both tables.\n\nleft_on: Column names from the left table.\n\n"
    indicator = (
        "indicator: bool or str\n\nTrue adds a _merge column. Its values are left_only for rows found only "
        "on the left, right_only for rows found only on the right, and both for matched rows.\n\n"
    )
    examples = "Examples of suffixes for overlapping columns.\n" * 4
    usage = "merged = merge(orders, customers, how='left')\n"
    original = document(introduction + keys + indicator + examples + usage,
        uri="https://pandas.pydata.org/docs/reference/api/pandas.merge.html", title="pandas.merge")
    key_start = len(introduction)
    indicator_start = key_start + keys.index("left_on:")
    indicator_end = key_start + len(keys + indicator)
    intro = build_evidence(snapshot=original.snapshot, element=original.element, end=key_start)
    key_options = build_evidence(snapshot=original.snapshot, element=original.element,
        start=key_start, end=key_start + len(keys))
    indicator_options = build_evidence(snapshot=original.snapshot, element=original.element,
        start=indicator_start, end=indicator_end)
    redundant_example = build_evidence(snapshot=original.snapshot, element=original.element,
        start=indicator_end, end=indicator_end + len(examples))
    left_usage = build_evidence(snapshot=original.snapshot, element=original.element,
        start=indicator_end + len(examples))
    candidates = [intro, key_options, indicator_options, redundant_example, left_usage]
    entry = RetrievalTask(requirement_id="join", route="docs", query="Keep match status and separately filter unmatched rows", k=5,
        requirement={"library": "pandas", "symbols": ["pandas.merge"], "evidence_needs": ["parameters", "explanation"]})
    budget = RetrievedEvidenceBudget(400, len(intro.excerpt + key_options.excerpt + indicator_options.excerpt + left_usage.excerpt), 8)

    packet, associations = select_evidence_packet(candidates, budget=budget,
        requirements_by_evidence={item.id: [entry] for item in candidates},
        semantic_support={
            key_options.id: {"join": {"relevance", "need:explanation", "need:parameters"}},
            intro.id: {"join": {"relevance", "need:explanation"}},
            indicator_options.id: {"join": {"relevance", "need:parameters"}},
            redundant_example.id: {"join": {"relevance", "need:explanation"}},
            left_usage.id: {"join": {"relevance"}},
        })

    assert indicator_options.id in associations
    assert any("left_only" in item.excerpt for item in packet)
    assert sum(len(item.excerpt) for item in packet) <= budget.max_total_excerpt_chars
