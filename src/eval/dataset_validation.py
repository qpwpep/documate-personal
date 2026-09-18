"""Offline release-fixture integrity checks, independent of product scoring.

Evidence checks prove that a declared excerpt occurs in its fixed source. They do
not prove semantic entailment, live retrieval quality, or successful document OCR.
Binary document excerpts and official web excerpts require an explicit text
manifest captured during dataset authoring; this module never fetches the web.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from src.core.upload_formats import ALL_UPLOAD_SUFFIXES
from .approved_uploads import logical_upload_name, validate_upload_names
from .config_models import BenchmarkCase
from .scenario_contracts import validate_execution_prerequisites


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _safe_snapshot_name(name: str) -> bool:
    """Snapshot capture checks symlinks; use only logical paths once captured."""
    try:
        logical_upload_name(name)
    except ValueError:
        return False
    return True


def _source_text(case: BenchmarkCase, source: str, uploads: Path,
                 source_manifest: dict[str, str],
                 attachment_bytes: dict[str, bytes] | None = None) -> tuple[str | None, str | None]:
    if source == "user:query":
        return case.query, None
    if source.startswith("user:setup:"):
        index = source.removeprefix("user:setup:")
        if index.isdigit() and int(index) < len(case.setup_turns):
            return case.setup_turns[int(index)], None
        return None, "invalid setup evidence reference"
    if source.startswith("user:"):
        return None, "invalid user evidence reference"
    if urlparse(source).scheme in {"http", "https", "design"}:
        return source_manifest.get(source), None
    if source not in case.resolved_upload_fixtures:
        return None, "undeclared upload evidence"
    if attachment_bytes is not None:
        if not _safe_snapshot_name(source) or source not in attachment_bytes:
            return None, "missing or unsafe upload evidence"
        suffix = PurePosixPath(logical_upload_name(source)).suffix.lower()
        contents = attachment_bytes[source]
    else:
        path = (uploads / source).resolve()
        if not path.is_relative_to(uploads) or not path.is_file():
            return None, "missing or unsafe upload evidence"
        suffix = path.suffix.lower()
        contents = path.read_bytes() if suffix in {".py", ".ipynb"} else b""
    if suffix == ".py":
        return contents.decode("utf-8-sig"), None
    if suffix == ".ipynb":
        notebook = json.loads(contents.decode("utf-8-sig"))
        return "\n".join(
            "".join(cell.get("source", [])) if isinstance(cell.get("source"), list)
            else str(cell.get("source", "")) for cell in notebook.get("cells", [])
        ), None
    return source_manifest.get(source), None


def validate_dataset(
    cases: list[BenchmarkCase], *, fixtures_path: Path, expected_count: int = 120,
    source_manifest: dict[str, str] | None = None,
    expected_distributions: dict[str, dict[str, int]] | None = None,
    attachment_bytes: dict[str, bytes] | None = None,
) -> list[str]:
    """Return all discovered integrity errors, without changing fixture records.

    ``source_manifest`` maps exact evidence source names to fixed source text.
    ``expected_distributions`` maps BenchmarkCase fields (for example category,
    difficulty, scenario, evaluation_role) to the planned counts for those fields.
    Legacy seeds remain loadable; this stricter contract applies when explicitly
    validating the new release dataset.

    When ``attachment_bytes`` is supplied, it is the already captured execution
    bundle. Attachment existence and content come exclusively from that snapshot;
    missing entries never fall back to the mutable filesystem.
    """
    errors: list[str] = []
    if len(cases) != expected_count:
        errors.append(f"expected {expected_count} cases, found {len(cases)}")
    uploads = fixtures_path.parent / "uploads"
    if attachment_bytes is None:
        uploads = uploads.resolve()
    manifest = source_manifest or {}
    seen_ids: set[str] = set()
    seen_content: dict[str, str] = {}
    for case in cases:
        label = case.case_id or "<empty case_id>"

        def add(message: str) -> None:
            errors.append(f"{label}: {message}")

        if not case.case_id.strip():
            add("case_id is blank")
        if case.case_id in seen_ids:
            add("duplicate case_id")
        seen_ids.add(case.case_id)
        payload = case.model_dump(mode="json", exclude={"case_id"})
        fingerprint = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if fingerprint in seen_content:
            add(f"duplicate content excluding ID (same as {seen_content[fingerprint]})")
        seen_content[fingerprint] = label
        if not case.query.strip():
            add("query is blank")
        if any(not turn.strip() for turn in case.setup_turns):
            add("setup turn is blank")
        for error in validate_execution_prerequisites(case):
            add(error)
        if set(case.expected_tools) & set(case.forbidden_tools):
            add("expected/forbidden tools overlap")
        if {_normalized(v).casefold() for v in case.must_include} & {
            _normalized(v).casefold() for v in case.must_not_include
        }:
            add("include/exclude overlap")
        if len(case.resolved_upload_fixtures) != len(set(case.resolved_upload_fixtures)):
            add("duplicate upload declaration")
        if attachment_bytes is not None:
            try:
                validate_upload_names(case.resolved_upload_fixtures)
            except ValueError as exc:
                add(str(exc))
        for filename in case.resolved_upload_fixtures:
            if attachment_bytes is not None:
                if not _safe_snapshot_name(filename):
                    add(f"attachment outside uploads: {filename}")
                elif filename not in attachment_bytes:
                    add(f"missing upload: {filename}")
                elif PurePosixPath(logical_upload_name(filename)).suffix.lower() not in ALL_UPLOAD_SUFFIXES:
                    add(f"unsupported upload format: {filename}")
                continue
            path = (uploads / filename).resolve()
            if not path.is_relative_to(uploads):
                add(f"attachment outside uploads: {filename}")
            elif not path.is_file():
                add(f"missing upload: {filename}")
            elif path.suffix.lower() not in ALL_UPLOAD_SUFFIXES:
                add(f"unsupported upload format: {filename}")
        expectation = case.save_expectation
        if "save_text" in case.expected_tools and expectation is None:
            add("save expectation is missing")
        if expectation is not None:
            if expectation.outcome == "must_not_execute":
                if "save_text" not in case.forbidden_tools:
                    add("must_not_execute must forbid save_text")
                if "save_text" in case.expected_tools:
                    add("must_not_execute cannot expect save_text")
            elif "save_text" not in case.expected_tools:
                add("save outcome requires expected save_text")
        if case.difficulty is None or case.evaluation_role is None or not case.capability:
            add("difficulty, evaluation_role and capability are required for release")
        oracle = case.oracle
        if oracle is None:
            add("oracle is missing")
            continue
        for field in ("required_facts", "expected_behaviors", "forbidden_behaviors"):
            values = getattr(oracle, field)
            if not values or any(not value.strip() for value in values):
                add(f"oracle.{field} must contain nonblank expectations")
        if {_normalized(v).casefold() for v in oracle.expected_behaviors} & {
            _normalized(v).casefold() for v in oracle.forbidden_behaviors
        }:
            add("oracle expected/forbidden behaviors overlap")
        if not oracle.evidence:
            add("oracle.evidence is empty")
        for evidence in oracle.evidence:
            if not evidence.source.strip() or not evidence.locator.strip() or not evidence.excerpt.strip():
                add("oracle evidence source, locator and excerpt must be nonblank")
                continue
            try:
                text, source_error = _source_text(case, evidence.source, uploads, manifest, attachment_bytes)
            except (OSError, ValueError, TypeError) as exc:
                add(f"unreadable evidence source {evidence.source}: {exc}")
                continue
            if source_error:
                add(f"{source_error}: {evidence.source}")
            elif text is None:
                add(f"source text unavailable: {evidence.source}")
            elif _normalized(evidence.excerpt) not in _normalized(text):
                add(f"excerpt not found in source: {evidence.source} ({evidence.locator})")
    for field, expected in (expected_distributions or {}).items():
        if field not in BenchmarkCase.model_fields:
            errors.append(f"unknown distribution field: {field}")
            continue
        observed = dict(Counter(str(getattr(case, field)) for case in cases))
        if observed != expected:
            errors.append(f"{field} distribution differs: expected {expected}, found {observed}")
    return errors


def assert_valid_dataset(cases: list[BenchmarkCase], **kwargs: Any) -> None:
    errors = validate_dataset(cases, **kwargs)
    if errors:
        raise ValueError("Invalid release dataset:\n" + "\n".join(errors))
