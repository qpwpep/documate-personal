from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from .schemas import BenchmarkCase, dump_jsonl, load_cases_jsonl


_CATEGORY_ORDER = ["docs_only", "rag_only", "hybrid", "tool_action"]
_QUERY_VARIANTS = [
    "핵심 개념을 먼저 요약한 뒤 실무 관점으로 정리해줘.",
    "초보자 관점에서 단계별로 풀어줘.",
    "한글로 간결하게 설명하고 필요한 경우 예시 코드를 포함해줘.",
    "실수하기 쉬운 포인트를 함께 알려줘.",
    "요약 + 체크리스트 형태로 답변해줘.",
]


def _build_variant_query(category: str, base_query: str, variant_index: int) -> str:
    suffix = _QUERY_VARIANTS[(variant_index - 1) % len(_QUERY_VARIANTS)]
    if category == "tool_action":
        return f"{base_query}\n추가로 최종 답변은 저장/공유 요청 조건을 충족해줘. {suffix}"
    return f"{base_query}\n추가 조건: {suffix}"


def _clone_case_with_variant(
    *,
    base_case: BenchmarkCase,
    category: str,
    index: int,
    make_variant: bool,
) -> BenchmarkCase:
    query = base_case.query if not make_variant else _build_variant_query(category, base_case.query, index)
    payload = base_case.model_dump()
    payload["case_id"] = f"{category}_{index:03d}"
    payload["category"] = category
    payload["query"] = query
    return BenchmarkCase(**payload)


def build_generated_cases(
    seed_cases: list[BenchmarkCase],
    target: int,
    random_seed: int = 42,
) -> list[BenchmarkCase]:
    if target <= 0:
        raise ValueError("target must be greater than 0")

    grouped: dict[str, list[BenchmarkCase]] = defaultdict(list)
    for case in seed_cases:
        grouped[case.category].append(case)

    for category in _CATEGORY_ORDER:
        if not grouped.get(category):
            raise ValueError(f"Missing seed case for category: {category}")

    rng = random.Random(random_seed)
    target_per_category = target // len(_CATEGORY_ORDER)
    remainder = target % len(_CATEGORY_ORDER)

    generated: list[BenchmarkCase] = []
    for category_index, category in enumerate(_CATEGORY_ORDER):
        category_target = target_per_category + (1 if category_index < remainder else 0)
        base_cases = list(grouped[category])
        rng.shuffle(base_cases)

        category_cases: list[BenchmarkCase] = []
        index = 1

        # Keep original seed semantics first.
        for base_case in base_cases:
            if len(category_cases) >= category_target:
                break
            category_cases.append(
                _clone_case_with_variant(
                    base_case=base_case,
                    category=category,
                    index=index,
                    make_variant=False,
                )
            )
            index += 1

        # Fill the rest with deterministic variants.
        while len(category_cases) < category_target:
            template = base_cases[(index - 1) % len(base_cases)]
            category_cases.append(
                _clone_case_with_variant(
                    base_case=template,
                    category=category,
                    index=index,
                    make_variant=True,
                )
            )
            index += 1

        generated.extend(category_cases)

    return generated


def generate_cases_file(
    *,
    seed_path: Path,
    out_path: Path,
    target: int,
    random_seed: int = 42,
) -> list[BenchmarkCase]:
    seed_cases = load_cases_jsonl(seed_path)
    generated_cases = build_generated_cases(seed_cases=seed_cases, target=target, random_seed=random_seed)
    dump_jsonl(out_path, generated_cases)
    return generated_cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate benchmark cases from seed fixtures.")
    parser.add_argument("--seed", type=Path, required=True, help="Seed JSONL path")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--target", type=int, required=True, help="Target number of total cases")
    parser.add_argument("--random-seed", type=int, default=42, help="Deterministic random seed")
    args = parser.parse_args()

    generated_cases = generate_cases_file(
        seed_path=args.seed,
        out_path=args.out,
        target=args.target,
        random_seed=args.random_seed,
    )
    print(f"Generated {len(generated_cases)} cases -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
