"""Reproducible planner-only evaluation of the frozen v2 request contract suite.

Live execution requires --live. A run directory is created exclusively and the
pre-execution manifest is never rewritten. This module never invokes retrieval,
save_text, Slack, or synthesis tools.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import re
import time
from typing import Any, Callable

from src.eval.request_contract_scoring import score_observation, summarize_results


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "data/benchmarks/request_contracts"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output/request_contract_evals"
SETTINGS_WHITELIST = ("planner_model", "planner_max_tokens")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def load_policy(path: Path | None = None) -> dict:
    return json.loads((path or FIXTURE_ROOT / "policy.v2.json").read_text(encoding="utf-8"))


def load_cases(path: Path | None = None) -> list[dict]:
    cases = [json.loads(line) for line in (path or FIXTURE_ROOT / "cases.v2.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("fixture case ids must be unique")
    return cases


def make_sample_plan(cases: list[dict], *, repeats: int) -> list[dict]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    return [{"sample_id": f"{case['id']}.r{repeat}", "case_id": case["id"], "repeat": repeat}
            for repeat in range(1, repeats + 1) for case in cases]


def build_case_state(case: dict) -> tuple[dict, dict]:
    """Build input facts only; no runtime interpreter constructs expected answers."""
    from langchain_core.messages import AIMessage, HumanMessage
    from src.core.answer_schema import finalize_answer, text_document
    from src.core.contracts.boundary.graph import build_graph_state_input
    from src.core.contracts.graph_state import PendingAction
    from src.core.request_contracts import RequestContract, UserTurnSnapshot

    context = case["context"]
    previous = context.get("previous_answer")
    previous_response = finalize_answer(text_document(previous["text"]), []) if previous else None
    history = list(context.get("history", []))
    pending_data = context.get("pending")
    if pending_data:
        history = [["user", pending_data["query"]], ["assistant", "어느 Slack 채널로 보낼까요?"], *history]
    messages, user_turns = [], []
    for index, (role, content) in enumerate(history):
        if role == "user":
            turn_id = f"u{len(user_turns) + 1}"
            messages.append(HumanMessage(content=content, id=turn_id))
            user_turns.append(UserTurnSnapshot(turn_id=turn_id, text=content))
        else:
            messages.append(AIMessage(content=content, id=f"a{index + 1}"))
    current_turn_id = f"u{len(user_turns) + 1}"
    messages.append(HumanMessage(content=case["query"], id=current_turn_id))
    user_turns.append(UserTurnSnapshot(turn_id=current_turn_id, text=case["query"]))
    pending = None
    if pending_data:
        pending_response = finalize_answer(text_document(pending_data["response_text"]), [])
        contract = RequestContract.model_validate({
            "request_id": pending_data["request_id"], "revision": 1,
            "body": {"kind": "copy_answer", "source": {"ref": "previous", "response_hash": pending_response.content_hash}},
            "actions": {name: {"intent": intent, "evidence_ids": ["original-instruction"]}
                        for name, intent in pending_data["actions"].items()},
            "evidence": [{"id": "original-instruction", "turn_id": "u1", "quote": pending_data["query"],
                          "scope": "current_request", "interpretation": "instruction"}],
            "slack_destination": pending_data.get("destination"),
            "missing_info": [{"slot": "slack_destination", "reason": "not_provided", "question": "어느 Slack 채널로 보낼까요?"}],
        })
        pending = PendingAction(contract=contract, response=pending_response, phase=pending_data["phase"],
                                completed_actions=tuple(pending_data["completed_actions"]), body_prepared=True)
    state = build_graph_state_input(
        user_input=case["query"], current_turn_id=current_turn_id, user_turns=tuple(user_turns),
        messages=messages, previous_response=previous_response, pending_action=pending,
    )
    evaluation_context = {
        "current_turn_id": current_turn_id, "user_turns": {turn.turn_id: turn.text for turn in user_turns},
        "answer_hashes": {"previous": previous_response.content_hash if previous_response else None,
                          "pending": pending.response.content_hash if pending and pending.response else None},
        "pending_request_id": pending.contract.request_id if pending else None,
        "pending_before": pending.model_dump(mode="json") if pending else None,
    }
    return state, evaluation_context


def _safe_model_message(message: Any) -> dict | None:
    if message is None:
        return None
    metadata = getattr(message, "response_metadata", None) or {}
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "content": getattr(message, "content", message if isinstance(message, (str, list)) else None),
        "tool_calls": getattr(message, "tool_calls", []) or [],
        "response_metadata": {key: metadata[key] for key in ("model_name", "model", "finish_reason", "system_fingerprint", "token_usage") if key in metadata},
        "usage_metadata": {key: usage[key] for key in ("input_tokens", "output_tokens", "total_tokens", "input_token_details", "output_token_details") if key in usage},
    }


def _prompt_messages(messages: list[Any]) -> list[dict]:
    return [{"role": getattr(message, "type", type(message).__name__), "id": getattr(message, "id", None),
             "content": getattr(message, "content", "")} for message in messages]


def _serializable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return str(value)


def _redact_secret(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, dict):
        return {key: _redact_secret(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_secret(item, secret) for item in value]
    return value


class _RecordingPlanner:
    def __init__(self, model: Any):
        self.model = model
        self.calls: list[dict] = []

    def invoke(self, messages: list[Any]):
        started = time.perf_counter()
        record = {"messages": _prompt_messages(messages)}
        try:
            output = self.model.invoke(messages)
            if isinstance(output, dict) and ("raw" in output or "parsed" in output):
                record.update(raw=_safe_model_message(output.get("raw")), parsed=_serializable(output.get("parsed")),
                              parsing_error=str(output["parsing_error"]) if output.get("parsing_error") else None)
            else:
                record.update(raw=_safe_model_message(output), parsed=_serializable(output), parsing_error=None)
            return output
        except Exception as exc:
            record["exception"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["latency_ms"] = round((time.perf_counter() - started) * 1000)
            self.calls.append(record)


def make_live_observer(settings: Any) -> Callable[[dict], dict]:
    """Create a factory boundary without calling any model until a sample runs."""
    def observe(case: dict) -> dict:
        from src.infra.llm import build_llm_registry
        from src.runtime.nodes.planner import make_planner_node

        state, context = build_case_state(case)
        started = time.perf_counter()
        errors = []
        result = None
        recorder = None
        try:
            recorder = _RecordingPlanner(build_llm_registry(settings).llm_planner)
            result = make_planner_node(recorder, verbose=False)(state)
            errors.extend(result.get("debug").planner_errors if result.get("debug") else [])
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        runtime = result.get("runtime") if result else None
        contract = runtime.request_contract if runtime else None
        planner = result.get("planner") if result else None
        calls = recorder.calls if recorder else []
        disposition = "clarify"
        if contract is not None:
            if contract.can_cancel_pending() or contract.can_acknowledge():
                disposition = "answer"
            elif (planner is not None and planner.guided_followup) or any(item.slot != "slack_destination" for item in contract.missing_info):
                disposition = "clarify"
            elif contract.can_prepare_body():
                disposition = "answer"
        raw_candidate = None
        if calls:
            parsed = calls[0].get("parsed")
            if isinstance(parsed, dict):
                raw_candidate = parsed.get("request_contract")
            if raw_candidate is None:
                raw_content = (calls[0].get("raw") or {}).get("content")
                if isinstance(raw_content, str):
                    try:
                        raw_candidate = json.loads(raw_content).get("request_contract")
                    except (ValueError, AttributeError):
                        pass
        usage = []
        for call in calls:
            raw = call.get("raw") or {}
            usage.append(raw.get("usage_metadata") or raw.get("response_metadata", {}).get("token_usage") or {})
        observation = {
            "canonical": contract.model_dump(mode="json") if contract else None,
            "raw_candidate": raw_candidate, "model_calls": calls,
            "contract_accepted": contract is not None and contract.failure is None,
            "errors": errors, "usage": usage,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "disposition": disposition, "context": context,
            "body_preparable": contract.can_prepare_body() if contract else False,
            "planner": planner.model_dump(mode="json") if planner else None,
            "pending_after": runtime.pending_action.model_dump(mode="json") if runtime and runtime.pending_action else None,
        }
        return _redact_secret(observation, str(settings.openai_api_key or ""))
    return observe


def code_fingerprint() -> dict:
    # Include all production and evaluator Python files: a changed indirect
    # dependency must not masquerade as the same implementation mid-batch.
    files = {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _hash(path.read_bytes())
             for path in sorted((PROJECT_ROOT / "src").rglob("*.py"))}
    return {"digest": _hash(_json(files)), "files": files}


def build_manifest(*, run_id: str, cases: list[dict], policy: dict, settings: Any,
                   repeats: int, max_workers: int, mode: str) -> dict:
    from src.infra.llm import _build_planner_response_schema
    from src.runtime.nodes.planner.prompt_builder import build_planner_messages

    prompt_hashes = {}
    for case in cases:
        state, _ = build_case_state(case)
        prompt_hashes[case["id"]] = _hash(_json(_prompt_messages(build_planner_messages(state))))
    schema = _build_planner_response_schema()
    code = code_fingerprint()
    whitelisted = {name: getattr(settings, name) for name in SETTINGS_WHITELIST}
    whitelisted.update(temperature=0, planner_timeout_seconds=30, planner_provider_max_retries=2,
                       planner_history_max_turns=6, max_workers=max_workers)
    dependencies = {}
    for name in ("openai", "langchain-openai", "langchain-core", "pydantic", "langgraph"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return {
        "run_id": run_id, "mode": mode, "policy_id": policy["policy_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": whitelisted, "python": platform.python_version(), "dependencies": dependencies,
        "hashes": {"code": code["digest"], "code_files": code["files"], "schema": _hash(_json(schema)),
                   "fixtures": _hash(_json(cases)), "policy": _hash(_json(policy)),
                   "settings": _hash(_json(whitelisted)), "prompts": prompt_hashes},
        "schema": schema, "sample_plan": make_sample_plan(cases, repeats=repeats),
        "repeats": repeats, "cases": [{"id": case["id"], "cohort": case["cohort"], "expected": case["expected"]} for case in cases],
        "scope": "planner model and canonicalization only; no retrieval, synthesis, save_text or Slack execution",
        "historical_runs_combined": False,
    }


def run_evaluation(*, cases: list[dict], policy: dict, observer: Callable[[dict], dict], manifest: dict,
                   output_root: Path = DEFAULT_OUTPUT_ROOT, repeats: int = 3, max_workers: int = 4,
                   code_hash: Callable[[], str] | None = None, progress: Callable[[dict], None] | None = None) -> dict:
    run_id = manifest["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}", run_id) or ".." in run_id:
        raise ValueError("run-id must be a simple unique identifier without path traversal")
    if not 1 <= max_workers <= 4:
        raise ValueError("max_workers must be between 1 and 4")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    by_id = {case["id"]: case for case in cases}
    plan = make_sample_plan(cases, repeats=repeats)
    initial_hash = manifest.get("hashes", {}).get("code")
    code_unchanged = True
    results = []

    def one(sample):
        case = by_id[sample["case_id"]]
        try:
            observation = observer(case)
        except Exception as exc:
            observation = {"canonical": None, "raw_candidate": None, "contract_accepted": False,
                           "errors": [f"{type(exc).__name__}: {exc}"], "disposition": None, "context": {}}
        return {**sample, "observation": observation, "score": score_observation(case, observation, policy)}

    with (run_dir / "results.jsonl").open("x", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="contract-eval") as pool:
            futures = [pool.submit(one, sample) for sample in plan]
            for future in as_completed(futures):
                row = future.result()
                try:
                    changed = code_hash is not None and code_hash() != initial_hash
                except Exception as exc:
                    changed = True
                    row["code_hash_error"] = f"{type(exc).__name__}: {exc}"
                row["code_changed_since_manifest"] = changed
                code_unchanged = code_unchanged and not changed
                results.append(row)
                handle.write(_json(row) + "\n")
                handle.flush()
                if progress:
                    progress({"sample_id": row["sample_id"], "completed": len(results), "total": len(plan),
                              "overall": row["score"]["overall"], "critical": row["score"]["critical_failures"]})
    if code_hash is not None:
        try:
            code_unchanged = code_unchanged and code_hash() == initial_hash
        except Exception:
            code_unchanged = False
    summary = summarize_results(results, cases, policy, repeats=repeats, code_unchanged=code_unchanged)
    summary.update(run_id=run_id, manifest_hash=_hash((run_dir / "manifest.json").read_bytes()))
    with (run_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly authorize the planner API calls in this run.")
    parser.add_argument("--plan-only", action="store_true", help="Write an exclusive manifest without invoking a model.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", nargs="+", help="Case ids or comma-separated ids; subsets are diagnostic and cannot qualify.")
    parser.add_argument("--model", help="Override the configured planner model and record the changed condition.")
    parser.add_argument("--max-tokens", type=int, help="Override the configured planner output cap and record the changed condition.")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--comparison-run", help="Earlier run id for an explicitly documented comparison; its results remain unchanged.")
    parser.add_argument("--change-note", help="Pre-execution reason and scope of the changes being evaluated.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.live == args.plan_only:
        parser.error("Select exactly one of --live or --plan-only. No model runs without --live.")
    if args.repeats < 1 or not 1 <= args.max_workers <= 4 or (args.max_tokens is not None and args.max_tokens < 1):
        parser.error("repeats and max-tokens must be positive; max-workers must be 1..4")
    cases, policy = load_cases(), load_policy()
    if args.cases:
        selected = {part for item in args.cases for part in item.split(",") if part}
        unknown = selected - {case["id"] for case in cases}
        if unknown:
            parser.error(f"Unknown case ids: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in selected]
    from src.infra.settings import get_settings
    overrides = {"verbose": False}
    if args.model:
        overrides["planner_model"] = args.model
    if args.max_tokens is not None:
        overrides["planner_max_tokens"] = args.max_tokens
    settings = get_settings().model_copy(update=overrides)
    if args.live and not settings.openai_api_key:
        parser.error("The configured planner API credential is unavailable.")
    manifest = build_manifest(run_id=args.run_id, cases=cases, policy=policy, settings=settings,
                              repeats=args.repeats, max_workers=args.max_workers,
                              mode="live" if args.live else "plan_only")
    manifest["comparison"] = {"previous_run_id": args.comparison_run, "change_note": args.change_note}
    manifest["settings_overrides"] = {key: value for key, value in {"planner_model": args.model, "planner_max_tokens": args.max_tokens}.items() if value is not None}
    if args.plan_only:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}", args.run_id) or ".." in args.run_id:
            parser.error("run-id must be a simple identifier")
        run_dir = args.output_root / args.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        with (run_dir / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Prepared {len(manifest['sample_plan'])} samples: {run_dir / 'manifest.json'}")
        return 0
    summary = run_evaluation(cases=cases, policy=policy, observer=make_live_observer(settings), manifest=manifest,
                             output_root=args.output_root, repeats=args.repeats, max_workers=args.max_workers,
                             code_hash=lambda: code_fingerprint()["digest"], progress=lambda event: print(_json(event), flush=True))
    print(_json({"run_id": args.run_id, "qualified": summary["qualified"], "counts": summary["counts"], "gates": summary["gates"]}))
    return 0 if summary["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
