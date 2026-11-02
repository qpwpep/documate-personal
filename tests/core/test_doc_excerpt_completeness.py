"""Fresh documentation windows keep complete source lines within their budget."""

from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalRequirement, RetrievalTask
from src.infra.tools.docs_search.ranking import select_docs_candidates
from src.runtime.nodes.synthesis.budgets import RetrievedEvidenceBudget
from src.runtime.nodes.synthesis.evidence_selection import prepare_evidence_candidates


def test_fresh_windows_do_not_cut_exception_conditions_mid_line():
    condition = "ParseError is raised when the input does not contain a valid document."
    text = (
        "# codec.decode\ncodec.decode(data)\n\n"
        + "The decoder accepts text and optional configuration.\n\n" * 58
        + "Raises:\n\n" + condition + "\n\n"
        + "# codec.encode\ncodec.encode(value)\n\n"
        + "Encoding writes the supplied value as text.\n\n" * 100
    )
    snapshot = build_snapshot(source_uri="https://docs.python.org/3/library/codec.html",
        title="codec", media_type="text/markdown", source_type="official", content=text,
        parser="official_html", parser_version="1")
    element = DocumentElement(element_id="body", kind="paragraph", text=text)
    original = SearchHit(evidence=build_evidence(snapshot=snapshot, element=element, end=3200),
        rank=1, score={"metric": "provider", "direction": "higher", "normalized": 0.9})

    candidates = select_docs_candidates("codec.decode invalid document", [original],
        RetrievalRequirement(library="Python", symbols=["codec.decode"], evidence_needs=["explanation"]),
        k=1, allowed_domains=["docs.python.org"])

    assert any(condition in hit.evidence.excerpt for hit in candidates)
    assert all(condition in hit.evidence.excerpt for hit in candidates if "ParseError" in hit.evidence.excerpt)
    assert all(0 < len(hit.evidence.excerpt) <= 3200 for hit in candidates)
    assert all(hit.evidence.selection.end == len(text) or hit.evidence.excerpt.endswith("\n")
               for hit in candidates)
    assert all(hit.evidence.excerpt == text[hit.evidence.selection.start:hit.evidence.selection.end]
               for hit in candidates)


def test_previously_selected_partial_window_keeps_its_exact_identity():
    text = "codec.decode(data)\n\n" + "Document details and examples.\n" * 180
    snapshot = build_snapshot(source_uri="https://docs.python.org/3/library/codec.html",
        title="codec", media_type="text/markdown", source_type="official", content=text,
        parser="official_html", parser_version="1")
    original = SearchHit(evidence=build_evidence(snapshot=snapshot,
        element=DocumentElement(element_id="body", kind="paragraph", text=text), end=3200),
        rank=1, score={"metric": "provider", "direction": "higher", "normalized": 0.9})

    candidates = select_docs_candidates("codec.decode", [original],
        RetrievalRequirement(library="Python", symbols=["codec.decode"]), k=1,
        allowed_domains=["docs.python.org"], preserved_evidence_ids={original.evidence.id})

    assert candidates[0].evidence == original.evidence


def test_overlong_source_line_still_has_a_bounded_candidate():
    text = "codec.decode(data) " + "Details without source line breaks. " * 200
    snapshot = build_snapshot(source_uri="https://docs.python.org/3/library/codec.html",
        title="codec", media_type="text/plain", source_type="official", content=text,
        parser="official_html", parser_version="1")
    original = SearchHit(evidence=build_evidence(snapshot=snapshot,
        element=DocumentElement(element_id="body", kind="paragraph", text=text), end=3200),
        rank=1, score={"metric": "provider", "direction": "higher", "normalized": 0.9})

    candidates = select_docs_candidates("codec.decode", [original],
        RetrievalRequirement(library="Python", symbols=["codec.decode"]), k=1,
        allowed_domains=["docs.python.org"])

    assert candidates
    assert all(0 < len(hit.evidence.excerpt) <= 3200 for hit in candidates)
    assert all(hit.evidence.excerpt == text[hit.evidence.selection.start:hit.evidence.selection.end]
               for hit in candidates)


def test_long_trailing_line_does_not_remove_the_requested_anchor():
    text = "a" * 1000 + "\n" + "b" * 100 + "\n" + "c" * 200 + "axis means the selected dimension. " + "d" * 5000 + "\n"
    snapshot = build_snapshot(source_uri="https://numpy.org/doc/stable/reference/generated/numpy.mean.html",
        title="numpy.mean", media_type="text/plain", source_type="official", content=text,
        parser="official_html", parser_version="1")
    original = SearchHit(evidence=build_evidence(snapshot=snapshot,
        element=DocumentElement(element_id="body", kind="paragraph", text=text), end=3200),
        rank=1, score={"metric": "provider", "direction": "higher", "normalized": 0.9})

    candidates = select_docs_candidates("numpy.mean axis", [original],
        RetrievalRequirement(library="numpy", symbols=["numpy.mean"], aspects=["axis"]), k=1,
        allowed_domains=["numpy.org"])

    assert any("axis means the selected dimension." in hit.evidence.excerpt for hit in candidates)
    assert all(0 < len(hit.evidence.excerpt) <= 3200 for hit in candidates)


def test_fresh_target_module_keeps_later_conditions_ahead_of_unrelated_stable_pages():
    condition = "ParseError is raised when the input does not contain a valid document."
    body = "codec.decode(data)\n\n" + "Parameter configuration details.\n\n" * 140 + "Raises:\n\n" + condition + "\n"
    hits = []
    for name, text, score in [("codec", body, 0.1), *[(f"other{index}", "Other module details.\n", 0.99) for index in range(8)]]:
        snapshot = build_snapshot(source_uri=f"https://docs.python.org/3/library/{name}.html",
            title=name, media_type="text/plain", source_type="official", content=text,
            parser="official_html", parser_version="1")
        hits.append(SearchHit(evidence=build_evidence(snapshot=snapshot,
            element=DocumentElement(element_id="body", kind="paragraph", text=text), end=min(len(text), 3200)),
            rank=len(hits) + 1, score={"metric": "provider", "direction": "higher", "normalized": score}))

    candidates = select_docs_candidates("codec.decode invalid input", hits,
        RetrievalRequirement(library="Python", symbols=["codec.decode"], evidence_needs=["explanation"]),
        k=1, allowed_domains=["docs.python.org"])

    assert candidates[0].evidence.snapshot.source_uri.endswith("/library/codec.html")
    assert any(condition in hit.evidence.excerpt for hit in candidates)
    assert all(0 < len(hit.evidence.excerpt) <= 3200 for hit in candidates)


def test_candidate_cap_keeps_error_condition_before_unrelated_bare_declarations():
    condition = "ParseError is raised when the decoded input is not a valid document."
    body = (
        "codec.decode(data)¶\n\nIdentical to codec.read(), but accepts a string.\n\n"
        "class codec.Decoder()¶\n\n"
        + "Parameters:\n\n" + "Decoder configuration details. " * 30 + "\n\n"
        + "\n\n".join(f"value_{index}(int)" for index in range(12)) + "\n\n"
        + condition + "\n\n" + "More decoder configuration details. " * 25 + "\n\n"
        + "codec.other()¶\n\nOther decoding helper.\n"
    )
    snapshot = build_snapshot(source_uri="https://docs.python.org/3/library/codec.html",
        title="codec", media_type="text/plain", source_type="official", content=body,
        parser="official_html", parser_version="1")
    entry = RetrievalTask(requirement_id="decode-error", route="docs", query="codec.decode exception for invalid data",
        k=3, requirement={"library": "Python", "symbols": ["codec.decode"], "evidence_needs": ["explanation"]})
    original = SearchHit(evidence=build_evidence(snapshot=snapshot,
        element=DocumentElement(element_id="body", kind="paragraph", text=body)), requirement_id=entry.requirement_id,
        rank=1, score={"metric": "provider", "direction": "higher", "normalized": 0.9})

    candidates = prepare_evidence_candidates(user_input=entry.query, hits=[original],
        planner_output=PlannerOutput(use_retrieval=True, tasks=[entry]),
        budget=RetrievedEvidenceBudget(1000, 6000, 8))

    assert any("Identical to codec.read()" in hit.evidence.excerpt for hit in candidates)
    assert any(condition in hit.evidence.excerpt for hit in candidates)


def test_multiple_windows_do_not_spend_a_documents_complementary_passage_slots():
    condition = "ParseError is raised when the decoded input is not a valid document."
    primary = ("codec.decode(data)¶\n\nIdentical to codec.read(), but accepts a string.\n\n"
        "class codec.Decoder()¶\n\nParameters:\n\n" + "Decoder configuration details. " * 35
        + "\n\n" + condition + "\n\n" + "Additional decoder behavior. " * 20 + "\n\n")
    secondary = "codec.other()¶\n\n" + "Other reference information. " * 45 + "\n\n"
    body = primary + secondary * 5
    entry = RetrievalTask(requirement_id="decode-error", route="docs", query="codec.decode exception for invalid data",
        k=3, requirement={"library": "Python", "symbols": ["codec.decode"], "evidence_needs": ["explanation"]})
    hits = []
    for locale in ("", "ko/", "uk/"):
        snapshot = build_snapshot(source_uri=f"https://docs.python.org/{locale}3/library/codec.html",
            title="codec", media_type="text/plain", source_type="official", content=body,
            parser="official_html", parser_version="1")
        element = DocumentElement(element_id="body", kind="paragraph", text=body)
        spans = [(0, len(primary)), *[(len(primary) + index * len(secondary), len(primary) + (index + 1) * len(secondary))
                                      for index in range(5)]]
        hits.extend(SearchHit(evidence=build_evidence(snapshot=snapshot, element=element, start=start, end=end),
            requirement_id=entry.requirement_id, rank=index + 1,
            score={"metric": "provider", "direction": "higher", "normalized": 0.9})
            for index, (start, end) in enumerate(spans))

    candidates = prepare_evidence_candidates(user_input=entry.query, hits=hits,
        planner_output=PlannerOutput(use_retrieval=True, tasks=[entry]),
        budget=RetrievedEvidenceBudget(1000, 6000, 8))

    assert len(candidates) <= 12
    assert any("Identical to codec.read()" in hit.evidence.excerpt for hit in candidates)
    assert any(condition in hit.evidence.excerpt for hit in candidates)
