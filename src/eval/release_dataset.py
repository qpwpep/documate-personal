"""Validate and promote the reviewed 120-case release; never synthesize padding."""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
from types import MappingProxyType

from src.infra.runtime_paths import get_benchmark_data_dir, get_generated_cases_fixture_path

from .config_models import BenchmarkCase
from .approved_uploads import ApprovedUpload
from .dataset_validation import validate_dataset
from .nemo_generate import _sha256 as specification_sha256


DEFAULT_DESIGN = get_benchmark_data_dir() / "design"
DEFAULT_RELEASE = get_generated_cases_fixture_path()


@dataclass(frozen=True)
class ReviewedRelease:
    """The approved input snapshot consumed by an evaluation run."""

    cases: list[BenchmarkCase]
    candidate_bytes: bytes
    uploads: Mapping[str, ApprovedUpload]
    inspection: dict
    review_sha256: str
    review_bytes: bytes
    design_files: dict[str, bytes]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _absolute(path: Path) -> Path:
    """Normalize a location without changing symlink replacement/upload semantics."""
    return Path(os.path.abspath(path))


def _relative_file(root: Path, name: str) -> Path:
    relative = Path(name.replace("\\", "/"))
    if relative.is_absolute() or PureWindowsPath(name).drive or relative.root or ".." in relative.parts:
        raise ValueError(f"unsafe artifact path: {name}")
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"artifact outside root: {name}")
    return path


def _artifact_snapshot(cases: list[BenchmarkCase], *, design: Path, fixtures_path: Path,
                       plan_bytes: bytes, read_file: Callable[[Path], bytes]
                       ) -> tuple[dict[str, bytes], dict[str, bytes], list[str]]:
    plan = json.loads(plan_bytes)
    artifacts = {"design/plan.json": plan_bytes}
    errors = []
    for name in [*plan["source_manifests"], *plan["specifications"]]:
        artifacts[f"design/{name}"] = read_file(_relative_file(design, name))
    uploads = {}
    for name in sorted({name for case in cases for name in case.resolved_upload_fixtures}):
        try:
            content = read_file(_relative_file(fixtures_path.parent / "uploads", name))
        except (OSError, ValueError) as exc:
            errors.append(f"cannot bind upload {name}: {exc}")
            continue
        uploads[name] = content
        artifacts[f"uploads/{name}"] = content
    return artifacts, uploads, errors


def _snapshot_reader() -> Callable[[Path], bytes]:
    contents: dict[Path, bytes] = {}
    locations: dict[Path, Path] = {}

    def read_file(path: Path) -> bytes:
        location = _absolute(path)
        if location not in locations:
            locations[location] = location.resolve()
        resolved = locations[location]
        if resolved not in contents:
            contents[resolved] = resolved.read_bytes()
        return contents[resolved]

    return read_file


def release_artifact_hashes(cases: list[BenchmarkCase], *, design: Path = DEFAULT_DESIGN,
                            fixtures_path: Path) -> dict[str, str]:
    """Bind review to the plan, source text, authored oracles and actual uploads.

    Call this when saving the review, after inspecting those exact artifacts.
    Hash keys are logical paths; relocation alone does not invalidate a review.
    """
    read_file = _snapshot_reader()
    artifacts, _, errors = _artifact_snapshot(
        cases, design=design, fixtures_path=fixtures_path,
        plan_bytes=read_file(design / "plan.json"), read_file=read_file,
    )
    if errors:
        raise ValueError("\n".join(errors))
    return {name: _digest(content) for name, content in sorted(artifacts.items())}


def _authored_fields(case: BenchmarkCase) -> dict:
    payload = case.model_dump(mode="json", exclude={"query"})
    payload["provenance"].pop("nemo_generation", None)
    return payload


def _lineage_errors(case: BenchmarkCase, authored: dict | None) -> list[str]:
    generation = case.provenance.get("nemo_generation")
    required_text = ("library_version", "model", "reviewer_model", "source_spec_sha256",
                     "effective_query_sha256", "reference_answer", "rationale")
    if not isinstance(generation, dict) or any(
        not isinstance(generation.get(key), str) or not generation[key].strip() for key in required_text
    ) or type(generation.get("seed")) is not int or type(generation.get("query_preserved")) is not bool:
        return [f"{case.case_id}: complete generation lineage is required"]
    errors = []
    review = generation.get("review")
    if not isinstance(review, dict) or review.get("consistent") is not True or review.get("issues") != []:
        errors.append(f"{case.case_id}: generation lineage must contain an accepted, issue-free review")
    if authored is None:
        errors.append(f"{case.case_id}: authored specification is missing")
        return errors
    if generation["source_spec_sha256"] != specification_sha256(authored):
        errors.append(f"{case.case_id}: source_spec_sha256 differs from the current authored specification; regenerate")
    if generation["effective_query_sha256"] != hashlib.sha256(case.query.encode("utf-8")).hexdigest():
        errors.append(f"{case.case_id}: effective_query_sha256 differs from the query reviewed during generation")
    original = BenchmarkCase.model_validate(authored)
    if _authored_fields(case) != _authored_fields(original):
        errors.append(f"{case.case_id}: immutable fields differ from the authored specification; regenerate")
    preserve_query = original.evaluation_role == "public_regression"
    if generation["query_preserved"] != preserve_query:
        errors.append(f"{case.case_id}: query_preserved conflicts with the authored evaluation role")
    if preserve_query and case.query != original.query:
        errors.append(f"{case.case_id}: public regression query must match the authored specification exactly")
    return errors


def _inspect_snapshot(path: Path, *, execution_path: Path, design: Path,
                      read_file: Callable[[Path], bytes]
                      ) -> tuple[list[BenchmarkCase], bytes, dict[str, bytes], dict, dict, dict[str, bytes]]:
    candidate_bytes = read_file(path)
    cases = [BenchmarkCase.model_validate_json(line)
             for line in candidate_bytes.decode("utf-8-sig").splitlines() if line.strip()]
    plan_bytes = read_file(design / "plan.json")
    plan = json.loads(plan_bytes)
    artifacts, uploads, snapshot_errors = _artifact_snapshot(
        cases, design=design, fixtures_path=execution_path, plan_bytes=plan_bytes, read_file=read_file,
    )
    sources = {}
    for filename in plan["source_manifests"]:
        sources.update(json.loads(artifacts[f"design/{filename}"]))
    authored_specs = {}
    spec_errors = []
    for filename in plan["specifications"]:
        for line in artifacts[f"design/{filename}"].decode("utf-8-sig").splitlines():
            if not line.strip():
                continue
            spec = json.loads(line)
            if spec["case_id"] in authored_specs:
                spec_errors.append(f"duplicate authored specification: {spec['case_id']}")
            authored_specs[spec["case_id"]] = spec
    scenarios = Counter()
    for row in plan["category_scenarios"].values():
        scenarios.update(row)
    distributions = {
        "category": {category: sum(row.values()) for category, row in plan["category_scenarios"].items()},
        "scenario": dict(scenarios),
        "difficulty": {key: value * 4 for key, value in plan["difficulty_per_category"].items()},
        "evaluation_role": plan["evaluation_roles"],
    }
    errors = validate_dataset(
        cases, fixtures_path=execution_path, expected_count=120, attachment_bytes=uploads,
        source_manifest=sources, expected_distributions=distributions,
    )
    errors.extend(spec_errors)
    errors.extend(snapshot_errors)
    text_format = plan.get("text_format")
    if text_format == "utf8-lf-v1":
        from .dataset_bytes import validate_text_bytes

        for name, content in {"candidate.jsonl": candidate_bytes, **artifacts}.items():
            errors.extend(validate_text_bytes(name, content))
    elif text_format is not None:
        errors.append(f"unsupported release text_format: {text_format}")
    if plan["release_count"] != 120:
        errors.append("release_count must remain exactly 120")
    for category, expected in plan["category_scenarios"].items():
        subset = [case for case in cases if case.category == category]
        if dict(Counter(case.scenario for case in subset)) != expected:
            errors.append(f"{category}: scenario cross-distribution differs from plan")
        if dict(Counter(case.difficulty for case in subset)) != plan["difficulty_per_category"]:
            errors.append(f"{category}: difficulty cross-distribution differs from plan")
    # IDs, audit metadata and reference answers cannot disguise identical tasks.
    seen_tasks = {}
    for case in cases:
        task = json.dumps({"query": " ".join(case.query.split()),
                           "setup": case.setup_turns, "files": case.resolved_upload_fixtures},
                          ensure_ascii=False, sort_keys=True)
        if task in seen_tasks:
            errors.append(f"duplicate task: {case.case_id} and {seen_tasks[task]}")
        seen_tasks[task] = case.case_id
        errors.extend(_lineage_errors(case, authored_specs.get(case.case_id)))
    inspection = {
        "case_count": len(cases), "sha256": _digest(candidate_bytes), "errors": errors,
        "artifact_hashes": {name: _digest(content) for name, content in sorted(artifacts.items())},
        "distributions": {field: dict(Counter(str(getattr(case, field)) for case in cases))
                          for field in distributions},
        "attachment_count": len({name for case in cases for name in case.resolved_upload_fixtures}),
        "attachment_cases": sum(bool(case.resolved_upload_fixtures) for case in cases),
        "multi_attachment_cases": sum(len(case.resolved_upload_fixtures) > 1 for case in cases),
        "setup_turns": sum(len(case.setup_turns) for case in cases),
        "no_tool_cases": sum(not case.expected_tools for case in cases),
    }
    design_files = {name.removeprefix("design/"): content for name, content in artifacts.items()
                    if name.startswith("design/")}
    return cases, candidate_bytes, uploads, inspection, plan, design_files


def inspect_release(path: Path, *, execution_path: Path | None = None,
                    design: Path = DEFAULT_DESIGN) -> dict:
    """Inspect candidate bytes against the files at their intended execution path."""
    return _inspect_snapshot(path, execution_path=execution_path or path, design=design,
                             read_file=_snapshot_reader())[3]


def load_reviewed_release(path: Path, *, review: Path, design: Path = DEFAULT_DESIGN,
                          execution_path: Path | None = None) -> ReviewedRelease:
    """Verify and return one snapshot; consumers must use its cases and uploads."""
    return _load_reviewed_snapshot(path, review=review, design=design,
                                   execution_path=execution_path or path, read_file=_snapshot_reader())


def _load_reviewed_snapshot(path: Path, *, review: Path, design: Path,
                            execution_path: Path, read_file: Callable[[Path], bytes]) -> ReviewedRelease:
    review_bytes = read_file(review)
    reviewed = json.loads(review_bytes)
    cases, candidate_bytes, uploads, result, plan, design_files = _inspect_snapshot(
        path, execution_path=execution_path, design=design, read_file=read_file,
    )
    strict_text = plan.get("text_format") == "utf8-lf-v1"
    if strict_text:
        from .dataset_bytes import validate_text_bytes

        result["errors"].extend(validate_text_bytes("review.json", review_bytes))
    if reviewed.get("candidate_sha256") != result["sha256"]:
        result["errors"].append("review is not bound to these exact candidate bytes")
    if reviewed.get("artifact_hashes") != result["artifact_hashes"]:
        result["errors"].append("review artifact hashes do not match the plan, sources, specifications and uploads")
    approved = reviewed.get("approved_case_ids")
    if (not isinstance(approved, list) or not all(isinstance(item, str) for item in approved)
            or len(approved) != len(cases) or set(approved) != {case.case_id for case in cases}):
        result["errors"].append("review must approve every release case exactly")
    if reviewed.get("unresolved_issues") != []:
        result["errors"].append("review contains unresolved issues or lacks an explicit issue list")
    for key in ("semantic_similarity_review", "prompt_overlap_review", "oracle_review"):
        if not reviewed.get(key):
            result["errors"].append(f"missing {key}")
    audit_artifacts = reviewed.get("audit_artifacts", {})
    if not isinstance(audit_artifacts, dict):
        result["errors"].append("review audit_artifacts must map design filenames to SHA-256")
    else:
        audit_hashes = {}
        for logical_name, expected in audit_artifacts.items():
            try:
                name = logical_name.removeprefix("design/")
                content = read_file(_relative_file(design, name))
                design_files[name] = content
                if strict_text:
                    result["errors"].extend(validate_text_bytes(name, content))
                actual = _digest(content)
                audit_hashes[logical_name] = actual
                if actual != expected:
                    result["errors"].append(f"review audit artifact hash differs: {logical_name}")
            except (OSError, ValueError) as exc:
                result["errors"].append(f"cannot bind audit artifact {logical_name}: {exc}")
        if audit_artifacts:
            result["audit_artifact_hashes"] = audit_hashes
    if result["errors"]:
        raise ValueError("Release not approved:\n" + "\n".join(result["errors"]))
    approved_uploads = MappingProxyType({name: ApprovedUpload(name, content) for name, content in uploads.items()})
    return ReviewedRelease(cases=cases, candidate_bytes=candidate_bytes, uploads=approved_uploads,
                           inspection=result, review_sha256=_digest(review_bytes),
                           review_bytes=review_bytes, design_files=design_files)
