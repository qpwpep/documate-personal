"""Packet selection consumes semantic verdicts without interpreting document prose."""

from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.runtime.nodes.synthesis.budgets import RetrievedEvidenceBudget
from src.runtime.nodes.synthesis.evidence_selection import (
    contains_evidence_range, select_evidence_packet,
)


def source(text, name="memoize"):
    snapshot = build_snapshot(source_uri=f"https://docs.python.org/3/{name}.html", title=name,
        media_type="text/markdown", source_type="official", content=text, parser="test", parser_version="1")
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
