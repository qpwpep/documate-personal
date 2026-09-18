"""Screen benchmark intents against each other, legacy cases and prompt examples.

Cosine similarity selects pairs for review; it never proves duplication or holdout
independence and never automatically removes a case. Only static source strings
are extracted from prompt modules. No runtime imports, environment values, oracle
answers or generator rationales are sent to the embedding provider.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "nvidia/nemotron-3-embed-1b"
DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1/embeddings"
DEFAULT_LEGACY = [Path("data/benchmarks/fixtures/cases.seed.jsonl"),
                  Path("data/benchmarks/fixtures/cases.regression.seed.jsonl")]
DEFAULT_PROMPTS = [Path("src/runtime/nodes/planner/prompt_builder.py"),
                   Path("src/runtime/nodes/synthesis/prompt_builder.py"),
                   Path("src/core/prompts.py")]


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def load_case_records(paths: list[Path], *, group: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = str(row["case_id"])
            if identifier in seen:
                raise ValueError(f"Repeated {group} case_id: {identifier}")
            seen.add(identifier)
            query = str(row["query"])
            intent = query
            setup = row.get("setup_turns", [])
            if setup:
                intent += "\nPrior dialogue:\n" + "\n".join(str(turn) for turn in setup)
            if row.get("capability"):
                intent += "\nCapability: " + str(row["capability"])
            records.append({"id": identifier if group == "candidate" else f"{group}:{identifier}",
                            "case_id": identifier, "group": group, "query": query, "text": intent,
                            "source": str(path), "line": line_number,
                            "evaluation_role": row.get("evaluation_role")})
    return records


def extract_prompt_records(paths: list[Path]) -> list[dict[str, Any]]:
    """Extract quoted examples and bounded rule segments without importing code.

    Literal concatenation is already represented as a single Constant by ast.
    Rule segments cover unquoted examples too. Dynamic interpolated values cannot
    be recovered by this static method and are an explicit coverage limitation.
    """
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    quoted = re.compile(r"(?<!\w)'([^'\n]{6,300})'(?!\w)|[“‘]([^”’\n]{6,300})[”’]")
    for path in paths:
        path = Path(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        docstrings: set[int] = set()
        for owner in ast.walk(tree):
            if isinstance(owner, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if owner.body and isinstance(owner.body[0], ast.Expr):
                    first = owner.body[0].value
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        docstrings.add(id(first))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str) or id(node) in docstrings:
                continue
            value = node.value.strip()
            if len(value) < 60:
                continue
            pieces = [("quoted_example", match.group(1) or match.group(2)) for match in quoted.finditer(value)]
            for line in value.splitlines():
                line = line.strip()
                if len(line) >= 60:
                    # Prompt lines here are short enough for the hosted API's
                    # 4096-token limit. Longer static literals are bounded by
                    # explicit character chunks with their positions retained.
                    pieces.extend(("rule_segment", line[start:start + 1400]) for start in range(0, len(line), 1400))
            for kind, text in pieces:
                normalized = _normalized(text)
                if normalized in seen:
                    continue
                seen.add(normalized)
                index = len(records) + 1
                records.append({"id": f"prompt:{path.as_posix()}:{node.lineno}:{index}",
                                "group": "prompt", "kind": kind, "source": str(path),
                                "line": node.lineno, "query": text, "text": text})
    return records


def embed_records(records: list[dict[str, Any]], *, model: str = DEFAULT_MODEL,
                  endpoint: str = DEFAULT_ENDPOINT, api_key_env: str = "NVIDIA_API_KEY",
                  batch_size: int = 32, timeout: float = 90) -> tuple[list[list[float]], dict[str, Any]]:
    if not os.environ.get(api_key_env):
        raise ValueError(f"Missing embedding credential: set {api_key_env}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    vectors: list[list[float]] = []
    usage: Counter[str] = Counter()
    response_models: set[str] = set()
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        body = {"model": model, "input": [record["text"] for record in batch],
                "input_type": "query", "encoding_format": "float", "truncate": "NONE"}
        request = Request(endpoint, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                          headers={"Authorization": "Bearer " + os.environ[api_key_env],
                                   "Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            # Never include request headers or a provider's credential-echoing body.
            raise RuntimeError(f"Embedding API returned HTTP {exc.code} for batch starting at {start}") from None
        data = sorted(payload.get("data", []), key=lambda item: item["index"])
        if [item["index"] for item in data] != list(range(len(batch))):
            raise ValueError(f"Embedding API returned missing or repeated indices in batch {start}")
        vectors.extend([[float(value) for value in item["embedding"]] for item in data])
        response_models.add(str(payload.get("model", model)))
        for key, value in payload.get("usage", {}).items():
            if isinstance(value, (int, float)):
                usage[key] += value
    return vectors, {"model": model, "response_models": sorted(response_models), "endpoint": endpoint,
                     "api_key_env": api_key_env, "input_type": "query", "encoding_format": "float",
                     "truncate": "NONE", "batch_size": batch_size, "usage": dict(usage),
                     "dimension": len(vectors[0]) if vectors else 0,
                     "model_version_limit": "The hosted model alias was recorded; the provider did not expose an immutable model revision."}


def build_report(records: list[dict[str, Any]], vectors: list[list[float]], *,
                 threshold: float = 0.75, neighbors: int = 3) -> dict[str, Any]:
    if not -1 <= threshold <= 1:
        raise ValueError("threshold must be between -1 and 1")
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    if len(records) != len(vectors):
        raise ValueError("Record and vector counts differ")
    identifiers = [record["id"] for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Similarity record identifiers must be unique")
    unit: list[list[float]] = []
    dimension = len(vectors[0]) if vectors else 0
    for vector in vectors:
        if len(vector) != dimension or not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding vectors must have one dimension and finite values")
        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            raise ValueError("Embedding vectors must be nonzero")
        unit.append([value / norm for value in vector])
    candidate_indices = [index for index, record in enumerate(records) if record["group"] == "candidate"]
    per_case, screened, exact = [], [], []
    for index in candidate_indices:
        row = records[index]
        by_group: dict[str, list[dict[str, Any]]] = {"candidate": [], "legacy": [], "prompt": []}
        for other_index, other in enumerate(records):
            if index == other_index:
                continue
            score = min(1.0, max(-1.0, sum(a * b for a, b in zip(unit[index], unit[other_index]))))
            by_group.setdefault(other["group"], []).append({"id": other["id"], "score": round(score, 6)})
            if other["group"] == "candidate" and other_index < index:
                continue
            pair = {"left": row["id"], "right": other["id"], "right_group": other["group"],
                    "score": round(score, 6)}
            if score >= threshold:
                screened.append(pair)
            if _normalized(row["query"]) == _normalized(other["query"]):
                exact.append(pair)
        for values in by_group.values():
            values.sort(key=lambda match: (-match["score"], match["id"]))
        per_case.append({"case_id": row["id"],
                         "nearest_within": by_group["candidate"][:neighbors],
                         "nearest_legacy": by_group["legacy"][:neighbors],
                         "nearest_prompt": by_group["prompt"][:neighbors]})
    screened.sort(key=lambda pair: (-pair["score"], pair["left"], pair["right"]))
    return {"threshold": threshold, "neighbors_per_group": neighbors,
            "interpretation": "Screening only: scores select pairs for semantic review, not automatic deletion. Similarity is sensitive to topic, wording and embedding model. Different intents may share high scores; low scores do not establish holdout independence.",
            "embedding_text_scope": "Candidate and legacy query + prior setup dialogue + capability; no answer facts, oracle, rubric or rationale.",
            "prompt_extraction_limit": "Static AST string constants in listed source files, quoted examples and bounded rule segments; excludes dynamic interpolated content, imported text and docstrings.",
            "normalization": "Exact query comparison uses Unicode NFKC, casefold and collapsed whitespace.",
            "counts": dict(Counter(record["group"] for record in records)),
            "records": records, "per_case": per_case, "screened_pairs": screened,
            "exact_query_duplicates": exact}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, nargs="+", required=True)
    parser.add_argument("--legacy", type=Path, nargs="*", default=DEFAULT_LEGACY)
    parser.add_argument("--prompts", type=Path, nargs="*", default=DEFAULT_PROMPTS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, help="Optional raw vector output; use an ignored output/ path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--neighbors", type=int, default=3)
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env")
    source_hashes = [{"path": str(path), "sha256": _hash_file(path)}
                     for path in args.cases + args.legacy + args.prompts]
    records = load_case_records(args.cases, group="candidate")
    if not records:
        raise ValueError("No candidate cases to compare")
    records.extend(load_case_records(args.legacy, group="legacy"))
    records.extend(extract_prompt_records(args.prompts))
    vectors, metadata = embed_records(records, model=args.model, endpoint=args.endpoint,
                                      api_key_env=args.api_key_env, batch_size=args.batch_size)
    if any(_hash_file(Path(item["path"])) != item["sha256"] for item in source_hashes):
        raise ValueError("A compared source changed during embedding; rerun against a stable snapshot")
    report = build_report(records, vectors, threshold=args.threshold, neighbors=args.neighbors)
    report["metadata"] = {**metadata, "created_at": datetime.now(timezone.utc).isoformat(),
                          "python": platform.python_version(), "client": "Python urllib.request",
                          "sources": source_hashes, "implementation_sha256": _hash_file(Path(__file__)),
                          "embedding_inputs_sha256": hashlib.sha256(json.dumps(
                              [record["text"] for record in records], ensure_ascii=False).encode("utf-8")).hexdigest()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.vectors:
        args.vectors.parent.mkdir(parents=True, exist_ok=True)
        args.vectors.write_text(json.dumps({"ids": [row["id"] for row in records], "vectors": vectors,
                                           "metadata": metadata}, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"counts": report["counts"], "screened_pairs": len(report["screened_pairs"]),
                      "exact_query_duplicates": len(report["exact_query_duplicates"]), "report": str(args.out)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
