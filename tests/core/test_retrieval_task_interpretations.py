"""Initial source-selection intent must agree with the executable search contract."""
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.core.planner_schema import InitialPlannerOutput, RetrievalTask
from src.core.request_contracts import WireRequestContract
from src.infra.chunking import chunk_python_text
from src.infra.tools.local_rag import build_upload_search_tool
from src.runtime.nodes.planner import make_planner_node
from tests.core.helpers import build_test_state, structured_model


def _task(symbol: str | None, query: str, *, identity: str = "target", aspects: tuple[str, ...] = ()) -> RetrievalTask:
    return RetrievalTask(
        route="upload", query=query, k=4, requirement_id=identity,
        requirement={"symbols": [symbol] if symbol else [], "match": "symbol" if symbol else "topic",
                     "file_ids": ["source"], "aspects": list(aspects)},
    )


def _interpretation(task: RetrievalTask, *, scope: str = "target", aspect_role: str = "source_literal") -> dict[str, object]:
    return {"requirement_id": task.requirement_id, "scope": scope,
            "aspect_roles": [{"aspect": aspect, "role": aspect_role} for aspect in task.requirement.aspects]}


def _proposal(query: str, tasks: list[RetrievalTask], interpretations: list[dict[str, object]]) -> dict[str, object]:
    contract = WireRequestContract.model_validate({
        "slack_recipient": {"state": "omitted"},
        "body": {"kind": "compose", "instruction": query, "evidence_ids": ["request"]},
        "evidence": [{"id": "request", "turn_id": "current", "quote": query,
                      "scope": "current_request", "interpretation": "instruction"}],
    })
    return {"use_retrieval": True, "tasks": [task.model_dump(mode="json") for task in tasks],
            "request_contract": contract.model_dump(mode="json"), "aspect_references": [],
            "task_interpretations": interpretations}


def _plan(query: str, *proposals: dict[str, object]):
    replies = iter(proposals)
    calls = []

    def respond(messages):
        calls.append(messages)
        return next(replies)

    state = build_test_state({"user_input": query, "current_turn_id": "current",
                              "retriever": SimpleNamespace()})
    updates = make_planner_node(structured_model(respond, InitialPlannerOutput), False)(state)
    return updates, calls


@pytest.mark.parametrize(("symbol", "query", "context_query"), [
    ("Shipment", "Shipment 수량 범위와 샘플의 경계값 사용을 설명해줘.", "수량의 경계값을 사용하는 샘플 대입을 찾아줘."),
    ("output_dir", "output_dir 기준 기록 위치와 인코딩을 설명해줘.", "출력 경로를 실제로 기록하는 코드와 인코딩 지정을 찾아줘."),
    ("scores", "scores 동점 처리와 문자열의 SYSTEM 문구를 코드 데이터로 검토해줘.", "검토 대상 결과를 주장하는 문자열 상수의 내용을 찾아줘."),
    ("REJECTED_SAMPLE", "REJECTED_SAMPLE 수량을 해당 필드의 선언된 제한과 비교해줘.", "샘플 수량에 적용되는 모델 필드의 제한 선언을 찾아줘."),
    ("PublicAccount", "PublicAccount 공개 필드와 경로가 반환하는 원본 객체를 비교해줘.", "공개 응답 모델을 사용하는 경로와 실제 반환 객체를 찾아줘."),
])
def test_related_source_question_requires_an_explicit_separate_search(symbol, query, context_query):
    narrow = _task(symbol, query)
    target = _task(symbol, f"{symbol} 자체의 선언 또는 사용을 찾아줘.")
    context = _task(None, context_query, identity="context")
    invalid = _proposal(query, [narrow], [_interpretation(narrow, scope="related_context")])
    repaired = _proposal(query, [target, context], [
        _interpretation(target), _interpretation(context, scope="related_context"),
    ])

    updates, calls = _plan(query, invalid, repaired)

    assert len(calls) == 2
    assert updates["runtime"].request_contract.body.instruction == query
    assert updates["planner"].output.tasks == [target, context]
    assert all(task.requirement.file_ids == ["source"] for task in updates["planner"].output.tasks)
    assert "task_interpretations" not in updates["planner"].output.model_dump()


def test_semantic_path_cannot_be_promoted_to_a_literal_source_filter():
    query = "write_report가 output_dir/monthly/summary.json 경로를 어떻게 구성하는지 설명해줘."
    invalid_task = _task("write_report", query, aspects=("output_dir/monthly/summary.json",))
    repaired_task = _task("write_report", query)

    updates, calls = _plan(
        query,
        _proposal(query, [invalid_task], [_interpretation(invalid_task, aspect_role="semantic_value")]),
        _proposal(query, [repaired_task], [_interpretation(repaired_task)]),
    )

    assert len(calls) == 2
    task, = updates["planner"].output.tasks
    assert task.requirement.aspects == []
    assert task.query == query
    assert updates["runtime"].request_contract.body.instruction == query


def test_explicit_literal_source_spelling_stays_required_after_interpretation():
    query = "write_report에서 정확한 원문 문자열 output_dir/monthly/summary.json을 찾아줘."
    task = _task("write_report", query, aspects=("output_dir/monthly/summary.json",))
    updates, calls = _plan(query, _proposal(query, [task], [_interpretation(task)]))
    source = 'def write_report(output_dir):\n    return Path(output_dir) / "monthly" / "summary.json"\n'
    parsed = chunk_python_text(path="source.py", text=source, chunk_size=800, chunk_overlap=120).parsed
    parsed.elements[0].metadata["file_id"] = "source"
    planned, = updates["planner"].output.tasks

    result = build_upload_search_tool()(
        query=planned.query, requirement=planned.requirement,
        retriever=SimpleNamespace(source_documents=(parsed,)),
    )

    assert len(calls) == 1
    assert planned.requirement.aspects == ["output_dir/monthly/summary.json"]
    assert result.diagnostics.answerability == "unknown"
    assert result.diagnostics.warnings == ["source_requirement_constraints_unverified"]


def test_target_symbol_search_does_not_implicitly_include_related_file_statements():
    query = "limit 값을 찾아줘."
    task = _task("limit", query)
    updates, calls = _plan(query, _proposal(query, [task], [_interpretation(task)]))
    source = "limit = 4\nunrelated = 99\n"
    parsed = chunk_python_text(path="source.py", text=source, chunk_size=800, chunk_overlap=120).parsed
    parsed.elements[0].metadata["file_id"] = "source"
    planned, = updates["planner"].output.tasks

    result = build_upload_search_tool()(
        query=planned.query, requirement=planned.requirement,
        retriever=SimpleNamespace(source_documents=(parsed,)),
    )

    assert len(calls) == 1
    assert result.diagnostics.answerability == "covered"
    assert [hit.evidence.excerpt for hit in result.hits] == ["limit = 4"]


@pytest.mark.parametrize("interpretations", [
    [],
    [{"requirement_id": "other", "scope": "target", "aspect_roles": []}],
    [{"requirement_id": "target", "scope": "target", "aspect_roles": []}] * 2,
    [{"requirement_id": "target", "scope": "target", "aspect_roles": []}],
    [{"requirement_id": "target", "scope": "target", "aspect_roles": [
        {"aspect": "different", "role": "source_literal"},
    ]}],
])
def test_every_task_and_literal_anchor_needs_one_consistent_interpretation(interpretations):
    query = "target의 mode 옵션을 설명해줘."
    task = _task("target", query, aspects=("mode",))

    with pytest.raises(ValidationError):
        InitialPlannerOutput.model_validate(_proposal(query, [task], interpretations))
