"""Compare local OCR engines on original bilingual document fixtures.

This is an opt-in live benchmark, not a deterministic test. Prefetch models into
--models before running. Each engine receives a separate process and identical
CPU options. Reports contain extraction quality, provenance, latency and RSS.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def normalized(text: str) -> str:
    return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", text).casefold(), flags=re.UNICODE)


def distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for index, lchar in enumerate(left, 1):
        current = [index]
        for other, rchar in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[other] + 1, previous[other - 1] + (lchar != rchar)))
        previous = current
    return previous[-1]


def language_scores(reference: str, text: str, lines: list[str]) -> dict[str, float]:
    """Separate Korean omissions from an otherwise correct English extraction."""
    scores = {}
    for name, pattern in (("korean", r"[^가-힣]"), ("latin", r"[^a-z]")):
        truth = re.sub(pattern, "", normalized(reference))
        observed = re.sub(pattern, "", normalized(text))
        scores[f"{name}_character_error_rate"] = distance(truth, observed) / max(1, len(truth))
    korean_lines = [line for line in lines if re.search(r"[가-힣]", line)]
    scores["korean_sentence_exact_recall"] = sum(normalized(line) in normalized(text) for line in korean_lines) / len(korean_lines)
    return scores


def worker(args) -> None:
    from src.core.uploads import UploadRecord
    from src.infra.docling_adapter import convert_file
    from src.infra.document_ingestion import ConversionPolicy, IngestionError

    expected = json.loads(args.reference.read_text(encoding="utf-8"))
    policy = ConversionPolicy(artifacts_path=str(args.models.resolve()), ocr_engine=args.engine,
                              ocr_languages=("ko", "en"), timeout_seconds=args.timeout,
                              max_pdf_pages=30, force_full_page_ocr=args.full_page)
    config = {"benchmark": True, "extraction_options": policy.extraction_options()}
    results = []
    for repetition in range(args.repeats):
        for filename in args.files:
            path = args.fixtures / filename
            raw = path.read_bytes()
            file = UploadRecord(file_id=filename, name=filename, path=str(path.resolve()), size_bytes=len(raw),
                                content_hash="sha256:" + hashlib.sha256(raw).hexdigest(),
                                source_uri=f"upload://benchmark/{filename}")
            started = time.perf_counter()
            entry = {"file": filename, "repetition": repetition + 1, "status": "failure", "input_sha256": file.content_hash}
            try:
                parsed = convert_file(file, policy, parser_config=config)
                text = "\n".join(element.text for element in parsed.elements if element.text)
                reference = expected["reference_text"]
                if filename == "bilingual.pdf":
                    reference += "\n" + "\n".join(expected["second_page"])
                observed, truth = normalized(text), normalized(reference)
                cells = [normalized(cell.text) for element in parsed.elements if element.table for cell in element.table.cells]
                expected_cells = [normalized(cell) for row in expected.get("table", []) for cell in row]
                facts = expected.get("facts", ["DOCUMATE-2026", "24", "16", "8", "Document retrieval preserves source pages and table structure"])
                if filename == "bilingual.pdf":
                    facts.append("두 번째 페이지의 정답은 42입니다")
                entry.update(status="success", character_error_rate=distance(truth, observed) / max(1, len(truth)),
                             table_cell_exact_recall=sum(cell in cells for cell in expected_cells) / len(expected_cells) if expected_cells else None,
                             required_fact_recall=sum(normalized(fact) in observed for fact in facts) / len(facts),
                             extracted_characters=len(text), element_count=len(parsed.elements),
                             table_count=sum(element.table is not None for element in parsed.elements),
                             pages=sorted({anchor.page_no for element in parsed.elements for anchor in element.anchors if anchor.page_no}),
                             page_count=parsed.elements[0].metadata.get("document_page_count"),
                             quality_issues=parsed.snapshot.quality_issues)
                entry.update(language_scores(reference, text, expected["lines"]))
                if expected.get("scoring") == "selected_passages":
                    for metric in ("character_error_rate", "korean_character_error_rate", "latin_character_error_rate", "table_cell_exact_recall"):
                        entry.pop(metric, None)
                    entry["passages"] = [{"reference": line, "exact_match": normalized(line) in observed} for line in expected["lines"]]
                    entry["source_page_no"] = expected["source_page_no"]
                target = args.output / f"{args.engine}-{Path(filename).stem}-{path.suffix[1:]}-{repetition + 1}"
                target.with_suffix(".txt").write_text(text, encoding="utf-8")
                target.with_suffix(".json").write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
            except IngestionError as exc:
                entry.update(error_code=exc.code, error=exc.message)
            except Exception as exc:
                entry.update(error_code=type(exc).__name__, error=str(exc))
            entry["conversion_seconds"] = round(time.perf_counter() - started, 3)
            results.append(entry)
            (args.output / f"{args.engine}-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(entry, ensure_ascii=False), flush=True)


def run(args) -> None:
    import psutil

    args.output.mkdir(parents=True, exist_ok=True)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    report = {"platform": platform.platform(), "python": platform.python_version(),
              "cpu": platform.processor(), "logical_cpu_count": os.cpu_count(),
              "memory_gib": round(psutil.virtual_memory().total / 1024 ** 3, 2),
              "versions": {name: version(name) for name in ("docling", "docling-core", "torch", "easyocr", "rapidocr", "onnxruntime")},
              "reference": reference, "full_page_ocr": args.full_page,
              "scope": "A selected local fixture; results do not establish universal OCR quality.",
              "metrics": {"character_error_rate": "Levenshtein distance after NFKC/casefold and punctuation/whitespace removal divided by reference character count",
                          "table_cell_exact_recall": "Exact normalized expected table cells recovered in structured output",
                          "conversion_seconds": "Adapter call including first model initialization, excluding parent process startup",
                          "wall_seconds": "Entire isolated engine process including Python/Docling imports",
                          "peak_rss_mib": "Peak sampled process-tree RSS at 0.1-second intervals"},
              "engines": {}}
    for engine in args.engines:
        argv = [sys.executable, str(Path(__file__).resolve()), "--engine", engine, "--models", str(args.models.resolve()),
                "--fixtures", str(args.fixtures.resolve()), "--output", str(args.output.resolve()),
                "--repeats", str(args.repeats), "--timeout", str(args.timeout), "--reference", str(args.reference.resolve()), "--files", *args.files]
        if args.full_page:
            argv.append("--full-page")
        environment = dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                           OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONUTF8="1")
        started, peak = time.perf_counter(), 0
        result_path = args.output / f"{engine}-results.json"
        result_path.unlink(missing_ok=True)
        with (args.output / f"{engine}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=environment)
            handle = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    peak = max(peak, sum(child.memory_info().rss for child in [handle, *handle.children(recursive=True)] if child.is_running()))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                if time.perf_counter() - started > args.timeout * len(args.files) * args.repeats + 180:
                    for child in reversed(handle.children(recursive=True)):
                        child.kill()
                    handle.kill()
                    process.wait()
                    break
                time.sleep(0.1)
        report["engines"][engine] = {"exit_code": process.returncode, "wall_seconds": round(time.perf_counter() - started, 3),
                                     "peak_rss_mib": round(peak / 1024 ** 2, 2),
                                     "results": json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else []}
        (args.output / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"engine": engine, **report["engines"][engine]}, ensure_ascii=False), flush=True)


def rescore(args) -> None:
    """Recompute language metrics from saved extracts without rerunning models."""
    report_path = args.output / "benchmark.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = json.loads(args.reference.read_text(encoding="utf-8"))
    if expected.get("scoring") == "selected_passages":
        raise ValueError("Full-text CER is not defined for selected-passage references")
    for engine, data in report["engines"].items():
        for entry in data["results"]:
            if entry["status"] != "success":
                continue
            source = Path(entry["file"])
            target = args.output / f"{engine}-{source.stem}-{source.suffix[1:]}-{entry['repetition']}.txt"
            reference = expected["reference_text"]
            if source.name == "bilingual.pdf":
                reference += "\n" + "\n".join(expected["second_page"])
            entry.update(language_scores(reference, target.read_text(encoding="utf-8"), expected["lines"]))
        (args.output / f"{engine}-results.json").write_text(json.dumps(data["results"], ensure_ascii=False, indent=2), encoding="utf-8")
    report["metrics"]["korean_character_error_rate"] = "Levenshtein error rate on Korean syllables only; includes omitted Korean text"
    report["metrics"]["latin_character_error_rate"] = "Levenshtein error rate on Latin letters only"
    report["metrics"]["korean_sentence_exact_recall"] = "Exact normalized recall of the three Korean or mixed-language body sentences"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "output/docling/models")
    parser.add_argument("--fixtures", type=Path, default=ROOT / "tests/fixtures/documents")
    parser.add_argument("--reference", type=Path, help="Ground truth JSON; defaults to the synthetic fixture reference")
    parser.add_argument("--output", type=Path, default=ROOT / "output/docling/benchmark")
    parser.add_argument("--engines", nargs="+", choices=("easyocr", "rapidocr"), default=["easyocr", "rapidocr"])
    parser.add_argument("--engine", choices=("easyocr", "rapidocr"), help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--full-page", action="store_true", help="Run full-page OCR instead of layout-region OCR")
    parser.add_argument("--rescore", action="store_true", help="Recompute language metrics from saved extracts")
    parser.add_argument("--files", nargs="+", default=["bilingual_scan.png", "bilingual_scan.pdf", "bilingual.pdf", "bilingual.docx"])
    arguments = parser.parse_args()
    if arguments.repeats < 1 or arguments.timeout <= 0:
        parser.error("repeats and timeout must be positive")
    if arguments.reference is None:
        arguments.reference = arguments.fixtures / "bilingual_expected.json"
    if arguments.rescore:
        rescore(arguments)
    elif arguments.engine:
        worker(arguments)
    else:
        run(arguments)
