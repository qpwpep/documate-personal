"""Packet selection consumes semantic verdicts without interpreting document prose."""

import pytest

from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.runtime.nodes.synthesis.budgets import RetrievedEvidenceBudget
from src.runtime.nodes.synthesis.evidence_selection import (
    contains_evidence_range, select_evidence_packet,
)


def source(text, name="memoize", *, uri=None, quality_issues=None, parser="test"):
    snapshot = build_snapshot(source_uri=uri or f"https://docs.python.org/3/{name}.html", title=name,
        media_type="text/markdown", source_type="official", content=text, parser=parser, parser_version="1",
        quality_issues=quality_issues)
    return build_evidence(snapshot=snapshot, element=DocumentElement(element_id="body", kind="paragraph", text=text))


def task(name="limit", *, needs=None):
    return RetrievalTask(requirement_id=name, route="docs", query=f"memoize {name} 동작",
        requirement={"library": "Python", "symbols": ["memoize"], "aspects": [name],
                     "evidence_needs": needs or ["parameters", "explanation"]}, k=4)


def prepare(originals, tasks, budget):
    from src.runtime.nodes.synthesis.evidence_selection import prepare_evidence_candidates
    hits = [SearchHit(evidence=item, rank=index + 1, requirement_id=entry.requirement_id,
            score={"metric": "test", "direction": "higher", "raw": 1.0})
            for index, (item, entry) in enumerate(zip(originals, tasks))]
    return prepare_evidence_candidates(user_input="매개변수 동작 설명", hits=hits,
        planner_output=PlannerOutput(use_retrieval=True, tasks=list({entry.requirement_id: entry for entry in tasks}.values())),
        budget=budget)


def assert_ranges(packet, originals, budget):
    assert len(packet) <= budget.max_items
    assert sum(len(item.excerpt) for item in packet) <= budget.max_total_excerpt_chars
    for item in packet:
        assert len(item.excerpt) <= budget.max_excerpt_chars
        assert any(contains_evidence_range(original, item) for original in originals)
        assert item.excerpt == item.element.text[item.selection.start:item.selection.end]


def test_semantic_support_selects_complementary_sources_without_lexical_gate():
    entry = task("limit", needs=["parameters", "explanation", "examples"])
    definition = source("이 값으로 저장하는 호출 수의 상한을 정합니다.", "definition")
    example = source("@memoize(None)\ndef f(n): return n * n", "example")
    irrelevant = source("limit limit limit: 다른 API에 관한 설명입니다.", "other")
    packet, ids = select_evidence_packet([irrelevant, definition, example],
        budget=RetrievedEvidenceBudget(200, 200, 2), requirements_by_evidence={item.id: [entry] for item in [irrelevant, definition, example]},
        semantic_support={definition.id: {entry.requirement_id: {"relevance", "aspect:limit", "need:parameters", "need:explanation"}},
                          example.id: {entry.requirement_id: {"relevance", "need:examples"}}})
    assert {item.id for item in packet} == {definition.id, example.id}
    assert ids == {item.id: [entry.requirement_id] for item in packet}


def test_extra_context_preserves_ranked_example_over_shorter_redundant_signatures():
    entry = RetrievalTask(requirement_id="cache-usage", route="docs", query="memoize usage", k=4,
        requirement={"library": "Python", "symbols": ["memoize"], "match": "symbol",
                     "evidence_needs": ["signature", "parameters", "examples", "explanation"]})
    example = source(
        "Decorate a function to reuse its result for the same arguments.\n"
        "@memoize(limit=2)\ndef square(n):\n    return n * n\n"
        "square(3)  # computes 9\nsquare(3)  # reuses 9\n", "usage-guide")
    broad = source("memoize(limit=128)\nlimit controls how many calls are stored. See the usage guide for examples.", "reference")
    explanation = source("Repeated arguments reuse their previously computed result.", "behavior")
    redundant = source("memoize(limit=128)", "older-reference")
    originals = [example, broad, explanation, redundant]
    budget = RetrievedEvidenceBudget(200, len(example.excerpt + broad.excerpt + explanation.excerpt), 3)

    # External semantic labels are allocation hints and can overlap. Retaining
    # only the shortest sources after coverage would lose the actual example.
    packet, ids = select_evidence_packet(originals, budget=budget,
        requirements_by_evidence={item.id: [entry] for item in originals}, semantic_support={
            broad.id: {entry.requirement_id: {"relevance", "need:signature", "need:parameters", "need:examples"}},
            explanation.id: {entry.requirement_id: {"relevance", "need:explanation"}},
            example.id: {entry.requirement_id: {"relevance", "need:examples", "need:explanation"}},
            redundant.id: {entry.requirement_id: {"relevance", "need:signature"}},
        })

    assert [item.id for item in packet] == [broad.id, explanation.id, example.id]
    assert ids == {item.id: [entry.requirement_id] for item in packet}
    assert_ranges(packet, originals, budget)


def test_same_document_usage_flow_precedes_overlapping_late_notes():
    entry = RetrievalTask(requirement_id="cache-usage", route="docs", query="memoize usage", k=4,
        requirement={"library": "Python", "symbols": ["memoize"], "match": "symbol",
                     "evidence_needs": ["explanation"]})
    introduction = ("memoize stores function results so repeated calls with the same arguments can reuse "
                    "the previously computed value.\n\n")
    workflow = ("Apply @memoize(limit=128) to the function before calling it. Call square(3) to compute "
                "the result, then call square(3) again to reuse the cached value. Choose limit=None "
                "only when the cache may grow without a fixed size limit.\n\n")
    controls = "Cache controls: clear() removes saved results; info() reports cache statistics.\n\n"
    note = "Statistics count cache hits and misses, and clearing stored results also resets these counters.\n"
    original = source(introduction + workflow + controls + note)
    notes_start = len(introduction + workflow)
    intro = build_evidence(snapshot=original.snapshot, element=original.element, start=0, end=len(introduction))
    usage = build_evidence(snapshot=original.snapshot, element=original.element,
                           start=len(introduction), end=notes_start)
    summary = build_evidence(snapshot=original.snapshot, element=original.element,
                             start=notes_start, end=notes_start + len(controls))
    repeated_notes = build_evidence(snapshot=original.snapshot, element=original.element,
                                    start=notes_start + len("Cache controls: "), end=len(original.excerpt))
    candidates = [intro, repeated_notes, usage, summary]
    budget = RetrievedEvidenceBudget(300, len(introduction + workflow + controls), 3)

    packet, ids = select_evidence_packet(candidates, budget=budget,
        requirements_by_evidence={item.id: [entry] for item in candidates},
        semantic_support={item.id: {entry.requirement_id: {"relevance", "need:explanation"}} for item in candidates})

    assert [item.id for item in packet] == [summary.id, intro.id, usage.id]
    assert any(workflow in item.excerpt for item in packet)
    assert ids == {item.id: [entry.requirement_id] for item in packet}
    assert_ranges(packet, [original], budget)


@pytest.mark.parametrize("excerpt_limit", [300, 70])
def test_unversioned_packet_uses_approved_stable_sources_without_mixing_archives(excerpt_limit):
    entry = RetrievalTask(requirement_id="merge", route="docs", query="pandas merge usage", k=4,
        requirement={"library": "pandas", "symbols": ["pandas.merge"], "evidence_needs": ["explanation"]})
    stable = source("Merge tables on common keys.\nUse on to name a shared key column.\nChoose how to retain matching or unmatched rows.\n",
                    uri="https://pandas.pydata.org/docs/reference/api/pandas.merge.html")
    old = source("Old defaults.", uri="https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.merge.html")
    dev = source("Future defaults.", uri="https://pandas.pydata.org/docs/dev/reference/api/pandas.merge.html")
    originals = [old, dev, stable]
    packet, ids = select_evidence_packet(originals, budget=RetrievedEvidenceBudget(excerpt_limit, 500, 4),
        requirements_by_evidence={item.id: [entry] for item in originals},
        semantic_support={item.id: {entry.requirement_id: {"relevance", "need:explanation"}} for item in originals})

    assert packet
    assert {item.snapshot.source_uri for item in packet} == {stable.snapshot.source_uri}
    assert ids == {item.id: [entry.requirement_id] for item in packet}
    assert_ranges(packet, originals, RetrievedEvidenceBudget(excerpt_limit, 500, 4))


@pytest.mark.parametrize("version, stable_approved", [("1.5", True), (None, False)])
def test_explicit_version_or_rejected_stable_source_preserves_approved_archive(version, stable_approved):
    entry = RetrievalTask(requirement_id="merge", route="docs", query="pandas merge usage", k=4,
        requirement={"library": "pandas", "version": version, "evidence_needs": ["explanation"]})
    old = source("Requested old behavior.", uri="https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.merge.html")
    stable = source("A longer current document about a different behavior.",
                    uri="https://pandas.pydata.org/docs/reference/api/pandas.merge.html")
    support = {old.id: {entry.requirement_id: {"relevance", "need:explanation"}}}
    if stable_approved:
        support[stable.id] = {entry.requirement_id: {"relevance", "need:explanation"}}
    packet, _ = select_evidence_packet([old, stable], budget=RetrievedEvidenceBudget(100, 100, 1),
        requirements_by_evidence={item.id: [entry] for item in [old, stable]}, semantic_support=support)

    assert [item.id for item in packet] == [old.id]


def test_current_capture_is_not_mixed_with_stale_search_excerpt_on_stable_url():
    entry = RetrievalTask(requirement_id="merge", route="docs", query="pandas merge usage", k=4,
        requirement={"library": "pandas", "evidence_needs": ["explanation"]})
    cached = source("Cached defaults.", uri="https://pandas.pydata.org/docs/user_guide/merging.html",
                    quality_issues=["freshness_unverified"])
    current = source("The currently retrieved official reference describes the merge operation.",
                     uri="https://pandas.pydata.org/docs/reference/api/pandas.merge.html")
    packet, _ = select_evidence_packet([cached, current], budget=RetrievedEvidenceBudget(100, 200, 3),
        requirements_by_evidence={item.id: [entry] for item in [cached, current]},
        semantic_support={item.id: {entry.requirement_id: {"relevance", "need:explanation"}} for item in [cached, current]})

    assert [item.id for item in packet] == [current.id]


@pytest.mark.parametrize("excerpt_limit", [300, 70])
def test_comparison_keeps_archive_associated_only_with_its_explicit_version(excerpt_limit):
    current_task = RetrievalTask(requirement_id="current", route="docs", query="pandas merge current usage", k=4,
        requirement={"library": "pandas", "evidence_needs": ["explanation"]})
    archived_task = RetrievalTask(requirement_id="archived", route="docs", query="pandas 1.5 merge usage", k=4,
        requirement={"library": "pandas", "version": "1.5", "evidence_needs": ["explanation"]})
    current = source("Merge current tables on a common key.\nThe current default controls matching rows.\nAdditional current behavior is documented here.\n",
                     uri="https://pandas.pydata.org/docs/reference/api/pandas.merge.html")
    archived = source("The archived default behaves differently.",
                      uri="https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.merge.html")
    budget = RetrievedEvidenceBudget(excerpt_limit, 500, 4)

    packet, ids = select_evidence_packet([archived, current], budget=budget,
        requirements_by_evidence={archived.id: [current_task, archived_task], current.id: [current_task]},
        semantic_support={
            archived.id: {entry.requirement_id: {"relevance", "need:explanation"}
                          for entry in [current_task, archived_task]},
            current.id: {current_task.requirement_id: {"relevance", "need:explanation"}},
        })

    assert {item.snapshot.source_uri for item in packet} == {current.snapshot.source_uri, archived.snapshot.source_uri}
    assert ids[archived.id] == ["archived"]
    assert all(ids[item.id] == ["current"] for item in packet if item.id != archived.id)
    assert_ranges(packet, [archived, current], budget)


def test_compact_packet_gets_new_identity_instead_of_reusing_full_source_verdict():
    entry, original = task(), source("limit: " + "조건과 설명을 함께 읽으세요. " * 20)
    packet, _ = select_evidence_packet([original], budget=RetrievedEvidenceBudget(70, 100, 2),
        requirements_by_evidence={original.id: [entry]},
        semantic_support={original.id: {entry.requirement_id: {"relevance", "aspect:limit", "need:parameters"}}})
    assert packet
    assert original.id not in {item.id for item in packet}
    assert_ranges(packet, [original], RetrievedEvidenceBudget(70, 100, 2))


def test_candidate_pool_retains_complete_wrapped_parameter_and_condition():
    lines = 5
    description = "Controls the parser used to read records and convert the input values. " * 20
    words = description.split()
    wrapped = [" ".join(words[index * len(words) // lines:(index + 1) * len(words) // lines])
               for index in range(lines)]
    wrapped[-1] += " The python engine supports additional parsing features for complex input files."
    required = "engine: str\n" + "\n".join(wrapped) + "\n"
    original = source("encoding: str\n" + "The character encoding decodes bytes. " * 20 + "\n"
                      + required + "other: str\n" + "Other behavior. " * 50)
    candidates = prepare([original], [task("engine")], RetrievedEvidenceBudget(1800, 6000, 8))
    assert any(required in hit.evidence.excerpt for hit in candidates)
    assert len(candidates) <= 8
    assert all(contains_evidence_range(original, hit.evidence) for hit in candidates)


@pytest.mark.parametrize("version, uri, parser, prefer_current", [
    (None, "https://pandas.pydata.org/docs/reference/api/pandas.merge.html", "official_html", True),
    ("1.5", "https://pandas.pydata.org/docs/reference/api/pandas.merge.html", "official_html", False),
    (None, "https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.merge.html", "official_html", False),
    (None, "https://pandas.pydata.org/docs/reference/api/pandas.merge.html", "test", False),
])
def test_candidate_cap_keeps_current_page_middle_before_cached_source_breadth(version, uri, parser, prefer_current):
    entry = RetrievalTask(requirement_id="merge-usage", route="docs", query="pandas.merge usage", k=5,
        requirement={"library": "pandas", "symbols": ["pandas.merge"], "version": version,
                     "evidence_needs": ["explanation"], "match": "symbol"})
    on = ("on: Hashable or a sequence of the previous\n\n"
          "Column or index level names to join on. These must be found in both DataFrames.")
    left_on = ("left_on: Hashable or a sequence of the previous, or array-like\n\n"
               "Column or index level names to join on in the left DataFrame.")
    current = source("pandas.merge(left, right, how='inner', on=None)\n\n"
        + "Merge DataFrame objects with a database-style join. " * 8 + "\n\n"
        + on + "\n\n" + left_on + "\n\n"
        + "indicator: bool or str, default False\n\n"
        + "Adds a column to the output DataFrame with information on the source of each row.",
        "pandas.merge", uri=uri, parser=parser)
    cached = [source("pandas.merge overview from the search provider.", "pandas.merge",
        uri=f"https://pandas.pydata.org/docs/reference/cached-{index}.html",
        quality_issues=["freshness_unverified"]) for index in range(12)]
    originals = [current, *cached]

    candidates = prepare(originals, [entry] * len(originals), RetrievedEvidenceBudget(450, 6000, 8))

    assert len(candidates) == 12
    current_candidates = [hit.evidence for hit in candidates
                          if hit.evidence.snapshot.snapshot_id == current.snapshot.snapshot_id]
    if prefer_current:
        assert any(on in item.excerpt for item in current_candidates)
        assert any(left_on in item.excerpt for item in current_candidates)
        assert any(hit.evidence.snapshot.quality_issues == ["freshness_unverified"] for hit in candidates)
    else:
        assert len(current_candidates) == 1
    assert all(len(hit.evidence.excerpt) <= 450 for hit in candidates)
    assert all(any(contains_evidence_range(original, hit.evidence) for original in originals) for hit in candidates)


def test_normal_packet_preserves_identity_and_charges_shared_range_once():
    original, first, second = source("두 동작을 함께 설명한 원문입니다."), task("first"), task("second")
    packet, ids = select_evidence_packet([original], budget=RetrievedEvidenceBudget(200, 200, 1),
        requirements_by_evidence={original.id: [first, second]}, semantic_support={original.id: {
            first.requirement_id: {"relevance", "aspect:first"}, second.requirement_id: {"relevance", "aspect:second"}}})
    assert packet == [original]
    assert set(ids[original.id]) == {first.requirement_id, second.requirement_id}


def test_small_generic_range_does_not_displace_deferred_complete_range():
    first, second = task("first"), task("second")
    complete = source("Complete first behavior. " * 15, "complete")
    generic = source("First context.", "generic")
    small = source("Second behavior.", "small")
    budget = RetrievedEvidenceBudget(400, 400, 2)
    packet, _ = select_evidence_packet([generic, complete, small], budget=budget,
        requirements_by_evidence={generic.id: [first], complete.id: [first], small.id: [second]},
        semantic_support={generic.id: {first.requirement_id: {"relevance"}},
                          complete.id: {first.requirement_id: {"relevance", "aspect:first", "need:explanation"}},
                          small.id: {second.requirement_id: {"relevance", "aspect:second", "need:explanation"}}})
    assert {item.id for item in packet} == {complete.id, small.id}
    assert_ranges(packet, [generic, complete, small], budget)


def test_saturated_source_pool_keeps_behavior_context_before_short_signatures():
    entry = RetrievalTask(requirement_id="lru-settings", route="docs",
        query="functools.lru_cache maxsize None typed True", k=6,
        requirement={"library": "Python", "symbols": ["functools.lru_cache"],
                     "aspects": ["maxsize", "None", "typed", "True"],
                     "evidence_needs": ["explanation", "parameters"], "match": "symbol"})
    signature = "@functools.lru_cache(maxsize=128, typed=False)"
    maxsize = "maxsize를 None으로 설정하면 캐시 크기에 제한이 없습니다."
    typed = "typed를 True로 설정하면 3과 3.0은 서로 다른 캐시 항목에 저장됩니다."
    body = signature + "\n\n" + "Shared background about cached calls. " * 40 + "\n\n"
    body += maxsize + "\n\n" + typed + "\n\n" + "Further implementation details. " * 35
    originals = [source(body, f"functools-version-{index}") for index in range(14)]
    candidates = prepare(originals, [entry] * 14, RetrievedEvidenceBudget(1800, 6000, 8))

    # The requirement cap is exhausted by one candidate from each source.
    # Those first candidates must provide context, not just a 50-character API declaration.
    assert len(candidates) == 14
    assert {hit.evidence.snapshot.snapshot_id for hit in candidates} == {item.snapshot.snapshot_id for item in originals}
    assert all(maxsize in hit.evidence.excerpt and typed in hit.evidence.excerpt for hit in candidates)
    assert all(len(hit.evidence.excerpt) <= 1800 for hit in candidates)


def test_mixed_api_window_prefers_requested_section_over_more_repeated_name_tokens():
    entry = RetrievalTask(requirement_id="lru-settings", route="docs",
        query="Python functools lru_cache maxsize typed cache_clear explanation example", k=6,
        requirement={"library": "Python", "symbols": ["functools.lru_cache"],
                     "aspects": ["maxsize", "typed"],
                     "evidence_needs": ["explanation", "parameters", "examples"], "match": "symbol"})
    unrelated = ("functools.cmp_to_key(func)\n\n"
        + "Python functools.lru_cache is also listed in the Python functools module index. " * 10
        + "\n\nExamples\n>>> sorted(items, key=cmp_to_key(compare))\n\n")
    signature = "functools.lru_cache(maxsize=128, typed=False)"
    explanation = "The wrapper stores recent results and reuses matching function calls. " * 8
    maxsize = "If maxsize is None, the cache can grow without bound."
    typed = "If typed is true, function arguments of different types are cached separately."
    clear = "The decorator provides a cache_clear() method to clear all stored results."
    target = "# functools.lru_cache\n" + signature + "\n\n" + explanation + "\n\n" + maxsize + "\n\n" + typed + "\n\n" + clear
    original = source(unrelated + target + "\n\n" + "Additional target details. " * 40)
    candidates = prepare([original], [entry], RetrievedEvidenceBudget(1800, 6000, 8))

    first = candidates[0].evidence
    assert first.selection.start >= len(unrelated)
    assert maxsize in first.excerpt and typed in first.excerpt and clear in first.excerpt
    assert all(contains_evidence_range(original, hit.evidence) for hit in candidates)


def test_same_owner_context_after_example_remains_in_the_first_candidate():
    entry = RetrievalTask(requirement_id="read-text", route="docs",
        query="Python pathlib.Path.read_text encoding errors", k=6,
        requirement={"library": "Python", "symbols": ["pathlib.Path.read_text"],
                     "aspects": ["encoding", "errors"],
                     "evidence_needs": ["explanation", "parameters"], "match": "symbol"})
    explanation = "The file is opened and then closed. The optional parameters have the same meaning as in open()."
    target = (
        'Path.read_text(encoding=None, errors=None)[\u00b6](#pathlib.Path.read_text "Link to this definition")\n'
        ':   Return the decoded contents of the pointed-to file as a string:\n\n'
        '    ```\n    >>> p = Path("file.txt")\n    >>> p.read_text()\n    "contents"\n    ```\n\n'
        '    ' + explanation + '\n\n    Added in version 3.5.\n\n'
    )
    other = ('Path.write_text(data, encoding=None, errors=None)[\u00b6](#pathlib.Path.write_text "Link to this definition")\n'
             ':   Write the given data.\n\n' + 'Other method implementation details. ' * 60)
    original = source(target + other)
    candidates = prepare([original], [entry], RetrievedEvidenceBudget(1800, 6000, 8))

    assert explanation in candidates[0].evidence.excerpt
    assert "Path.write_text" not in candidates[0].evidence.excerpt
    assert candidates[0].evidence.selection.end <= len(target)
    assert all(contains_evidence_range(original, hit.evidence) for hit in candidates)
