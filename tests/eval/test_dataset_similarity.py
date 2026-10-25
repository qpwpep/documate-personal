from __future__ import annotations

import json

import pytest

from src.eval.dataset_similarity import load_case_records


@pytest.mark.parametrize(("field", "value"), [("query", {"text": "hello"}), ("setup_turns", "AB")])
def test_similarity_input_rejects_invalid_case_fields_with_source_location(tmp_path, field, value):
    path = tmp_path / "candidates.jsonl"
    case = {"case_id": "invalid-input", "category": "docs_only", "query": "Explain arrays."}
    case[field] = value
    path.write_text("\n\n" + json.dumps(case) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        load_case_records([path], group="candidate")

    assert f"{path}:3" in str(caught.value)
    assert field in str(caught.value)


@pytest.mark.parametrize(("group", "record_id"), [("candidate", "follow-up"), ("legacy", "legacy:follow-up")])
def test_similarity_input_preserves_dialogue_order_and_source_metadata(tmp_path, group, record_id):
    path = tmp_path / "cases.jsonl"
    case = {
        "case_id": "follow-up",
        "category": "docs_only",
        "query": "  이제 비교해줘.  ",
        "setup_turns": ["배열을 설명해줘.", "목록도 설명해줘."],
        "capability": "comparison",
        "evaluation_role": "new_evaluation",
        "oracle": {"required_facts": ["Only evaluation sees this fact."]},
    }
    path.write_text("\n" + json.dumps(case) + "\n", encoding="utf-8-sig")

    records = load_case_records([path], group=group)

    assert records == [{
        "id": record_id,
        "case_id": "follow-up",
        "group": group,
        "query": "  이제 비교해줘.  ",
        "text": "  이제 비교해줘.  \nPrior dialogue:\n배열을 설명해줘.\n목록도 설명해줘.\nCapability: comparison",
        "source": str(path),
        "line": 2,
        "evaluation_role": "new_evaluation",
    }]
