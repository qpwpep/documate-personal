"""Opt-in, fixture-only prompt comparison; no search or delivery tools execute.

Run with LIVE_TEST=true and PROMPT_COMPARISON_RUN_ID set to a fresh identifier.
PROMPT_COMPARISON_PHASE=baseline/current allows separate before/after runs;
PROMPT_COMPARISON_BASE_REF selects the git revision supplying baseline prompts.
PROMPT_COMPARISON_STAGES=planner restricts a follow-up to the changed stage.
Results retain raw outputs and independent checks under ignored output/.
This small diagnostic sample does not qualify the complete v2 benchmark.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import pytest

from langchain_core.messages import HumanMessage, SystemMessage
from src.core.answer_schema import export_answer_text, iter_content_units
from src.core.contracts import PlannerState, RetrievalState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import (
    AnswerContract, ComposeBody, ContentRequirement, ContractEvidence, ExtractBody,
    RequestContract, check_answer_contract,
)
from src.eval.request_contract_eval import (
    _RecordingPlanner, _redact_secret, build_case_state, load_cases, load_policy,
)
from src.eval.request_contract_scoring import score_observation
from src.infra.llm import _build_planner_response_schema, build_llm_registry
from src.infra.settings import get_settings
from src.runtime.nodes.synthesis.budgets import resolve_synthesis_budget_profile
from src.runtime.nodes.synthesis.pipeline import _invoke_structured_attempt
from src.runtime.nodes.synthesis.schema_adapter import _build_synthesis_response_schema, build_structured_synthesizer


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol")
PHASES = (os.getenv("PROMPT_COMPARISON_PHASE"),) if os.getenv("PROMPT_COMPARISON_PHASE") else ("baseline", "current")
V2_IDS = ("negation_en", "quotation_en", "abbreviation_ko", "pending_destination")
pytestmark = pytest.mark.skipif(os.getenv("LIVE_TEST") != "true", reason="Live prompt comparison requires LIVE_TEST=true")


def _source_module(name: str, phase: str, revision: str) -> dict:
    module = importlib.import_module(name)
    namespace = dict(vars(module))
    source = (subprocess.check_output(
        ["git", "show", f"{revision}:{name.replace('.', '/')}.py"], cwd=ROOT,
    ).decode("utf-8") if phase == "baseline" else Path(module.__file__).read_text(encoding="utf-8"))
    exec(compile(source, f"{phase}:{name}", "exec"), namespace)
    return namespace


def _builders(phase: str, revision: str) -> dict:
    planner_prompt = _source_module("src.runtime.nodes.planner.prompt_builder", phase, revision)
    planner = _source_module("src.runtime.nodes.planner.node", phase, revision)
    planner["build_planner_messages"] = planner_prompt["build_planner_messages"]
    synthesis_prompt = _source_module("src.runtime.nodes.synthesis.prompt_builder", phase, revision)
    synthesis_prompt["SYS_POLICY"] = _source_module("src.core.prompts", phase, revision)["SYS_POLICY"]
    context = _source_module("src.runtime.nodes.synthesis.context", phase, revision)
    context["build_synthesis_messages"] = synthesis_prompt["build_synthesis_messages"]
    return {"planner": planner, "context": context,
            "session": _source_module("src.runtime.nodes.session", phase, revision)}


def _planner_cases() -> list[dict]:
    cases = [case for case in load_cases() if case["id"] in V2_IDS]
    assert {case["id"] for case in cases} == set(V2_IDS), "The selected v2 fixtures must all exist"
    context = {"previous_answer": None, "pending": None, "history": []}
    cases.extend([
        {"id": "alias_docs", "context": context,
         "query": "import numpy as np로 사용 중이야. 공식 문서에서 np.reshape의 order 매개변수를 확인해줘.",
         "routes": ["docs"], "symbol": "numpy.reshape", "aspect": "order"},
        {"id": "exclude_upload", "context": context,
         "query": "내 업로드 코드는 사용하지 말고 Python 공식 문서만 근거로 pathlib.Path.read_text를 설명해줘.",
         "routes": ["docs"], "symbol": "pathlib.Path.read_text"},
    ])
    return cases


def _observe_planner(case: dict, registry, builders: dict) -> dict:
    state, context = build_case_state(case)
    recorder = _RecordingPlanner(registry.llm_planner)
    result = builders["planner"]["make_planner_node"](recorder, False)(state)
    runtime, planner = result["runtime"], result["planner"]
    contract = runtime.request_contract
    accepted = contract is not None and contract.failure is None
    disposition = "clarify"
    if accepted:
        if contract.can_cancel_pending() or contract.can_acknowledge():
            disposition = "answer"
        elif not planner.guided_followup and not any(item.slot != "slack_destination" for item in contract.missing_info) and contract.can_prepare_body():
            disposition = "answer"
    parsed = recorder.calls[0].get("parsed") if recorder.calls else None
    observation = {
        "canonical": contract.model_dump(mode="json") if contract else None,
        "raw_candidate": parsed.get("request_contract") if isinstance(parsed, dict) else None,
        "model_calls": recorder.calls, "contract_accepted": accepted,
        "disposition": disposition, "context": context,
        "planner": planner.model_dump(mode="json"),
        "errors": result["debug"].planner_errors if result.get("debug") else [],
        "pending_after": runtime.pending_action.model_dump(mode="json") if runtime.pending_action else None,
    }
    if "expected" in case:
        score = score_observation(case, observation, load_policy())
    else:
        tasks = planner.output.tasks
        checks = {
            "contract_valid": accepted,
            "routes": sorted({task.route for task in tasks}) == case["routes"],
            "symbol": any(case["symbol"] in task.requirement.symbols for task in tasks),
            "no_actions": accepted and not any(contract.action_requested(action) for action in ("save_text", "slack_notify")),
        }
        if case.get("aspect"):
            checks["aspect"] = any(case["aspect"] in task.requirement.aspects for task in tasks)
        score = {"overall": all(checks.values()), "checks": checks}
    return {"case_id": case["id"], "stage": "planner", "score": score, "observation": observation}


def _synthesis_state(case_id: str):
    if case_id == "code_only":
        query = "Show only Python example code for numpy.reshape with order='F'; no explanation."
        entries = [("docs", "numpy.reshape(a, shape, order='C') changes the shape of an array. order='F' reads and writes elements in Fortran-like index order.")]
        constraints = (ContentRequirement(kind="code_example", mode="required", evidence_ids=("r1",)),
                       ContentRequirement(kind="explanation", mode="forbidden", evidence_ids=("r1",)))
        body = ComposeBody(instruction=query)
    elif case_id == "exact_excerpt":
        query = "Return the supplied original excerpt exactly, including its leading spaces and final newline."
        entries = [("docs", "  First setting is enabled.\nSecond setting uses three retries.\n")]
        constraints = ()
        body = ExtractBody(instruction=query)
    else:
        query = "Compare the documented retry default with my uploaded code and cite both sources."
        entries = [("docs", "The documented default is retries=3."), ("upload", "retries = 5\n")]
        constraints = (ContentRequirement(kind="comparison", mode="required", evidence_ids=("r1",)),)
        body = ComposeBody(instruction=query)
    contract = RequestContract(request_id=f"comparison-{case_id}", body=body,
        answer=AnswerContract(content=constraints),
        evidence=(ContractEvidence(id="r1", turn_id="u1", quote=query, scope="answer", interpretation="instruction"),))
    tasks, hits = [], []
    for index, (route, text) in enumerate(entries):
        task = RetrievalTask(route=route, query=query, k=1, requirement_id=f"r{index + 1}")
        snapshot = build_snapshot(source_uri=f"https://fixture.invalid/{index}" if route == "docs" else "uploads/fixture.py",
            title="Fixture source", media_type="text/plain", source_type="official" if route == "docs" else "upload",
            content=text, parser="fixture", parser_version="1")
        evidence = build_evidence(snapshot=snapshot,
            element=DocumentElement(element_id=f"source-{index}", kind="paragraph" if route == "docs" else "code", text=text))
        tasks.append(task)
        hits.append(SearchHit(evidence=evidence, rank=1, requirement_id=task.requirement_id,
            score=RetrievalScore(metric="fixture", raw=1, normalized=1, direction="higher")))
    return build_graph_state_input(user_input=query, messages=[HumanMessage(content=query)], request_contract=contract,
        planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=tasks)),
        retrieval=RetrievalState(hit_log=[hit.model_dump(mode="json") for hit in hits]))


def _observe_synthesis(case_id: str, registry, builders: dict, settings) -> dict:
    state = _synthesis_state(case_id)
    context = builders["context"]["build_synthesis_context"](state=state, has_default_slack_destination=False)
    profile = resolve_synthesis_budget_profile(user_input=context.user_input, planner_output=context.planner_output,
                                             snippet_char_limit=settings.synthesis_prompt_snippet_chars)
    prepared = builders["context"]["prepare_synthesis_inputs"](state=state, context=context, budget_profile=profile,
        max_turns=6, prompt_snippet_char_limit=profile.snippet_chars, prompt_evidence_char_budget=profile.evidence_chars)
    recorder = _RecordingPlanner(build_structured_synthesizer(registry.llm_synthesizer))
    try:
        result = _invoke_structured_attempt(structured_synthesizer=recorder, prepared=prepared, llm_calls=[], path="structured")
    except Exception as exc:
        return {"case_id": case_id, "stage": "synthesis", "score": {"overall": False},
                "error": f"{type(exc).__name__}: {exc}", "model_calls": recorder.calls}
    contract_check = check_answer_contract(context.request_contract.answer, result.content, evidence=prepared.evidence_packet)
    checks = {
        "schema": True, "references": all(check.reference_status != "missing" for check in result.checks),
        "excerpt_match": all(check.support_status != "unsupported" for check in result.checks),
        "contract": contract_check.valid,
    }
    if case_id == "code_only":
        checks["code_only"] = bool(result.content.blocks) and all(block.type == "code" for block in result.content.blocks)
    elif case_id == "exact_excerpt":
        units = [unit for _, unit in iter_content_units(result.content)]
        checks["exact_text"] = len(units) == 1 and units[0].text == prepared.evidence_packet[0].excerpt
        checks["exact_basis"] = len(units) == 1 and units[0].basis == "excerpt"
    else:
        checks["both_sources"] = {citation.evidence.route for citation in result.citations} == {"docs", "upload"}
    return {"case_id": case_id, "stage": "synthesis", "score": {"overall": all(checks.values()), "checks": checks},
            "model_calls": recorder.calls, "result": result.model_dump(mode="json"), "rendered": export_answer_text(result),
            "unchecked_semantic": contract_check.unchecked_semantic}


def _observe_summary(registry, builders: dict) -> dict:
    messages = [SystemMessage(content=builders["session"]["SUMMARY_SYS"]), HumanMessage(content=(
        "[Existing bounded memory]\nUser is reviewing NumPy 2.0 numpy.reshape order behavior.\n\n"
        "[Newly evicted conversation]\n"
        "user: Use only official docs, exclude uploaded code, and do not send anything to Slack.\n"
        "assistant: I have not verified order='A' yet. No message has been sent.\n"
        "user: Correction: use NumPy 2.1. Keep the same source restriction and do not send anything.\n"
    ))]
    recorder = _RecordingPlanner(registry.llm_summarizer)
    raw = recorder.invoke(messages)
    text = builders["session"]["extract_text_content"](raw.content)
    # Semantic preservation is reviewed from the recorded output, never inferred from keywords.
    return {"case_id": "summary_constraints", "stage": "summary", "score": {"overall": bool(text.strip())},
            "model_calls": recorder.calls, "text": text,
            "manual_checks": ["NumPy 2.1 supersedes 2.0", "official docs only; uploaded code excluded",
                              "Slack remains forbidden and unsent", "order='A' remains unverified"]}


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("model", MODELS)
def test_live_prompt_contract_compatibility(model: str, phase: str):
    """The requested models preserve request, reference, and output contracts on fixed inputs."""
    assert phase in {"baseline", "current"}
    settings = get_settings().model_copy(update={"planner_model": model, "chat_model": model, "summary_model": model, "verbose": False})
    assert settings.openai_api_key, "Configured API credential is unavailable"
    revision = subprocess.check_output(["git", "rev-parse", os.getenv("PROMPT_COMPARISON_BASE_REF", "HEAD")], cwd=ROOT).decode().strip()
    run_id = os.getenv("PROMPT_COMPARISON_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    assert re.fullmatch(r"[A-Za-z0-9_-]+", run_id), "Use a simple run ID"
    destination = ROOT / "output" / "prompt_comparison" / run_id / f"{phase}-{model}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists(), f"Comparison output already exists: {destination}"
    builders = _builders(phase, revision)
    registry = build_llm_registry(settings)
    stages = set(os.getenv("PROMPT_COMPARISON_STAGES", "planner,synthesis,summary").split(","))
    assert stages and stages <= {"planner", "synthesis", "summary"}
    jobs = []
    if "planner" in stages:
        jobs.extend((case["id"], lambda case=case: _observe_planner(case, registry, builders)) for case in _planner_cases())
    if "synthesis" in stages:
        jobs.extend((case_id, lambda case_id=case_id: _observe_synthesis(case_id, registry, builders, settings))
                    for case_id in ("code_only", "exact_excerpt", "hybrid_comparison"))
    if "summary" in stages:
        jobs.append(("summary_constraints", lambda: _observe_summary(registry, builders)))
    rows = []

    def run_job(case_id, operation):
        started = time.perf_counter()
        try:
            row = operation()
        except Exception as exc:
            row = {"case_id": case_id, "score": {"overall": False}, "error": f"{type(exc).__name__}: {exc}"}
        row["latency_ms"] = round((time.perf_counter() - started) * 1000)
        for call in row.get("model_calls", row.get("observation", {}).get("model_calls", [])):
            call["prompt_hash"] = hashlib.sha256(json.dumps(call["messages"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        return _redact_secret(row, str(settings.openai_api_key or ""))

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_job, case_id, operation) for case_id, operation in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps({"phase": phase, "model": model, "case": row["case_id"], "passed": row["score"]["overall"]}), flush=True)
    options = {key: getattr(settings, key) for key in (
        "planner_max_tokens", "synthesis_max_tokens", "synthesis_timeout_seconds", "synthesis_max_retries",
        "synthesis_use_responses_api", "synthesis_reasoning_effort", "summary_max_tokens", "synthesis_prompt_snippet_chars")}
    options.update(temperature=0, planner_timeout=30, planner_retries=2, summary_timeout=60, summary_retries=2)
    schema_hashes = {stage: hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
                     for stage, schema in (("planner", _build_planner_response_schema()), ("synthesis", _build_synthesis_response_schema()))}
    artifact = {"model": model, "phase": phase, "baseline_revision": revision, "settings": options,
                "schema_hashes": schema_hashes, "stages": sorted(stages), "repeats": 1, "results": rows,
                "scope": "Fixture-only model calls. No search, Slack, or file delivery. Semantic claims need manual review."}
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if phase == "current":
        assert all(row["score"]["overall"] for row in rows), f"Diagnostic failures preserved in {destination}"
