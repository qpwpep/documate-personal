from __future__ import annotations

import hashlib
import json
import os
import sys
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import psutil

from src.core.uploads import UploadRecord
from src.infra.docling_runner import DoclingRunner
from src.infra import docling_runner as runner_module
from src.infra.document_ingestion import ConversionPolicy, IngestionError


WORKER = '''import argparse, json, sys
from pathlib import Path
from src.core.documents import ParsedDocument, DocumentElement, build_snapshot
p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--result'); a=p.parse_args()
request=json.loads(Path(a.request).read_text(encoding='utf-8'))
marker=Path(a.request).parent.parent/'worker_runs'
marker.write_text(marker.read_text()+'x' if marker.exists() else 'x')
if request['files'][0]['name']=='wait.pdf':
    import threading
    threading.Event().wait()
if request['files'][0]['name']=='treewait.pdf':
    import threading, subprocess
    child=subprocess.Popen([sys.executable, '-c', 'import threading; threading.Event().wait()'])
    (Path(a.request).parent.parent/'child_pid').write_text(str(child.pid))
    threading.Event().wait()
if request['files'][0]['name']=='partial.pdf':
    Path(a.result).write_text(json.dumps({'error':{'code':'DOCUMENT_PARTIAL_CONVERSION','message':'partial','file_name':'partial.pdf'}}))
    sys.exit(1)
if request['files'][0]['name']=='malformed.pdf':
    Path(a.result).write_text('[]')
    sys.exit(0)
documents=[]
for file in request['files']:
    snapshot=build_snapshot(source_uri=file['source_uri'], title=file['name'], media_type='application/pdf', source_type='upload', content=Path(file['path']).read_bytes(), parser='docling', parser_version='test', parser_config=request['parser_config'])
    doc=ParsedDocument(snapshot=snapshot,elements=[DocumentElement(element_id='t',kind='paragraph',text=('x'*700_000 if file['name'].startswith('large') else 'source 120'),metadata={'document_page_count': 5 if file['name']=='five.pdf' else 1})])
    documents.append(doc.model_dump(mode='json'))
Path(a.result).write_text(json.dumps({'documents':documents}),encoding='utf-8')
'''


@pytest.fixture
def conversion(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(WORKER, encoding="utf-8")
    runner = DoclingRunner(command=[sys.executable, str(script)],
                           identity_provider=lambda policy: policy.extraction_options())

    def source(name="native.pdf", identity="first"):
        raw = b"%PDF-1.7 fixture"
        path = tmp_path / name
        path.write_bytes(raw)
        return UploadRecord(file_id=identity, name=name, path=str(path), size_bytes=len(raw),
            content_hash="sha256:" + hashlib.sha256(raw).hexdigest(), source_uri=f"upload:///session/{identity}")

    yield runner, source
    runner.close()


def test_process_conversion_reuses_cache_without_reusing_source_identity(conversion, tmp_path):
    """A cache hit starts no second worker and binds the source to the new attachment."""
    runner, source = conversion
    args = dict(policy=ConversionPolicy(), cache_dir=tmp_path / "cache")
    first = runner.convert_many([source()], workspace=tmp_path / "run1", **args)[0]
    second = runner.convert_many([source("renamed.pdf", "second")], workspace=tmp_path / "run2", **args)[0]
    assert (tmp_path / "worker_runs").read_text() == "x"
    assert first.snapshot.snapshot_id != second.snapshot.snapshot_id
    assert second.snapshot.title == "renamed.pdf"
    assert not (tmp_path / "run1").exists()
    first.elements.clear()
    assert second.elements[0].text == "source 120"


def test_partial_conversion_is_not_cached_and_releases_workspace(conversion, tmp_path):
    """A partial document cannot become a successful source through the cache."""
    runner, source = conversion
    with pytest.raises(IngestionError) as failed:
        runner.convert_many([source("partial.pdf")], policy=ConversionPolicy(),
                            workspace=tmp_path / "run", cache_dir=tmp_path / "cache")
    assert failed.value.code == "DOCUMENT_PARTIAL_CONVERSION"
    assert not list((tmp_path / "cache").glob("*.json"))
    assert not (tmp_path / "run").exists()
    assert runner.active_process_count == 0


def test_timeout_terminates_worker_and_next_conversion_succeeds(conversion, tmp_path):
    """A hung conversion releases its process and concurrency slot for the next request."""
    runner, source = conversion
    with pytest.raises(IngestionError) as failed:
        runner.convert_many([source("wait.pdf")], policy=ConversionPolicy(timeout_seconds=1),
                            workspace=tmp_path / "run")
    assert failed.value.code == "DOCUMENT_PROCESSING_TIMEOUT"
    assert runner.active_process_count == 0
    assert not (tmp_path / "run").exists()
    result = runner.convert_many([source()], policy=ConversionPolicy(), workspace=tmp_path / "next")
    assert result[0].elements[0].text == "source 120"


def test_cleanup_failure_preserves_conversion_error_and_releases_slot(conversion, tmp_path, monkeypatch):
    """Failure to reap an already stopped worker cannot mask partial conversion or strand the slot."""
    runner, source = conversion
    stop = runner_module._stop_process

    def reap_then_fail(process):
        stop(process)
        raise OSError("simulated process bookkeeping failure")

    monkeypatch.setattr(runner_module, "_stop_process", reap_then_fail)
    with pytest.raises(IngestionError) as failed:
        runner.convert_many([source("partial.pdf")], policy=ConversionPolicy(), workspace=tmp_path / "run")
    assert failed.value.code == "DOCUMENT_PARTIAL_CONVERSION"
    assert not (tmp_path / "run").exists()
    result = runner.convert_many([source()], policy=ConversionPolicy(), workspace=tmp_path / "next")
    assert result[0].elements[0].text == "source 120"
    assert runner.active_process_count == 0


def test_close_during_preparation_does_not_launch_a_worker(conversion, tmp_path, monkeypatch):
    """Closing the runner before worker registration cancels the prepared request without a process."""
    runner, source = conversion
    entered, resume = threading.Event(), threading.Event()

    def blocked_identity(policy):
        entered.set()
        assert resume.wait(5)
        return policy.extraction_options()

    runner._identity_provider = blocked_identity
    launched = []
    popen = runner_module.subprocess.Popen

    def observe_launch(*args, **kwargs):
        launched.append(True)
        return popen(*args, **kwargs)

    monkeypatch.setattr(runner_module.subprocess, "Popen", observe_launch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(runner.convert_many, [source()], policy=ConversionPolicy(), workspace=tmp_path / "run")
        assert entered.wait(5)
        runner.close()
        resume.set()
        with pytest.raises(IngestionError):
            future.result(timeout=10)
    assert launched == []
    assert not (tmp_path / "worker_runs").exists()
    assert runner.active_process_count == 0


def test_expired_request_never_starts_conversion(conversion, tmp_path, monkeypatch):
    """An already expired request is rejected before allocating a worker process."""
    runner, source = conversion
    launched = []
    popen = runner_module.subprocess.Popen

    def observe_launch(*args, **kwargs):
        launched.append(True)
        return popen(*args, **kwargs)

    monkeypatch.setattr(runner_module.subprocess, "Popen", observe_launch)
    with pytest.raises(IngestionError) as failed:
        runner.convert_many([source()], policy=ConversionPolicy(), workspace=tmp_path / "run", deadline=0)
    assert failed.value.code == "DOCUMENT_PROCESSING_TIMEOUT"
    assert not launched


def test_cached_pdf_page_count_respects_a_lower_current_limit(conversion, tmp_path):
    """Unanchored or blank pages still count when reusing a conversion under stricter limits."""
    runner, source = conversion
    file = source("five.pdf")
    runner.convert_many([file], policy=ConversionPolicy(max_pdf_pages=10), workspace=tmp_path / "first", cache_dir=tmp_path / "cache")
    with pytest.raises(IngestionError) as failed:
        runner.convert_many([file], policy=ConversionPolicy(max_pdf_pages=2), workspace=tmp_path / "second", cache_dir=tmp_path / "cache")
    assert failed.value.code == "DOCUMENT_LIMIT_EXCEEDED"
    assert (tmp_path / "worker_runs").read_text() == "x"


def test_timeout_terminates_worker_descendants(conversion, tmp_path):
    """A timed-out conversion cannot leave a child process consuming resources."""
    runner, source = conversion
    child = None
    try:
        with pytest.raises(IngestionError) as failed:
            runner.convert_many([source("treewait.pdf")], policy=ConversionPolicy(timeout_seconds=3), workspace=tmp_path / "run")
        assert failed.value.code == "DOCUMENT_PROCESSING_TIMEOUT"
        child = int((tmp_path / "child_pid").read_text())
        assert not psutil.pid_exists(child)
        assert runner.active_process_count == 0
    finally:
        if child is not None and psutil.pid_exists(child):
            psutil.Process(child).kill()


def test_malformed_worker_envelope_reports_adapter_failure_and_releases_resources(conversion, tmp_path):
    """A syntactically valid but wrong result shape cannot escape as an unclassified exception."""
    runner, source = conversion
    with pytest.raises(IngestionError) as failed:
        runner.convert_many([source("malformed.pdf")], policy=ConversionPolicy(), workspace=tmp_path / "run")
    assert failed.value.code == "DOCUMENT_ADAPTER_ERROR"
    assert runner.active_process_count == 0
    assert not (tmp_path / "run").exists()


def test_worker_rejects_policy_fingerprint_mismatch_before_loading_models(conversion, tmp_path):
    """A worker cannot label one extraction configuration with another configuration's cache identity."""
    policy = ConversionPolicy()
    _, source = conversion
    request = tmp_path / "request.json"
    result = tmp_path / "result.json"
    request.write_text(json.dumps({"files": [source().model_dump(mode="json")], "policy": policy.model_dump(mode="json"),
                                  "parser_config": {**policy.extraction_options(), "do_ocr": False}}), encoding="utf-8")
    process = subprocess.run([sys.executable, "-m", "src.infra.docling_worker", "--request", str(request), "--result", str(result)],
                             capture_output=True, timeout=20)
    assert process.returncode == 1
    assert json.loads(result.read_bytes())["error"]["code"] == "DOCUMENT_ADAPTER_ERROR"


def test_model_fingerprint_supports_artifacts_stored_under_a_cache_directory(tmp_path, monkeypatch):
    """A conventional .cache ancestor is not mistaken for disposable files inside the model directory."""
    root = tmp_path / ".cache" / "docling"
    root.mkdir(parents=True)
    (root / "weights.bin").write_bytes(b"model version one")
    monkeypatch.setattr(runner_module.importlib.metadata, "version", lambda name: "test")
    identity = runner_module.conversion_identity(ConversionPolicy(artifacts_path=str(root)))
    assert identity["models_digest"]
    (root / "weights.bin").write_bytes(b"changed model contents")
    changed = runner_module.conversion_identity(ConversionPolicy(artifacts_path=str(root)))
    assert identity["models_digest"] != changed["models_digest"]


def test_combined_cache_hits_cannot_bypass_the_request_output_budget(conversion, tmp_path):
    """Individually valid cache entries still respect the combined result-size limit of a new request."""
    runner, source = conversion
    first, second = source("large.pdf"), source("large-two.pdf", "second")
    policy = ConversionPolicy(max_output_mib=1)
    runner.convert_many([first], policy=policy, workspace=tmp_path / "first", cache_dir=tmp_path / "cache")
    with pytest.raises(IngestionError) as failed:
        runner.convert_many([first, second], policy=policy, workspace=tmp_path / "second", cache_dir=tmp_path / "cache")
    assert failed.value.code == "DOCUMENT_LIMIT_EXCEEDED"
    assert (tmp_path / "worker_runs").read_text() == "x"


@pytest.mark.parametrize("expected_owner", [False, True])
def test_worker_and_descendant_exit_when_parent_dies_without_cleanup(tmp_path, expected_owner):
    """A worker stuck before conversion cannot survive the abrupt loss of its API parent."""
    worker_script = tmp_path / "orphan_worker.py"
    worker_script.write_text('''
import json, os, subprocess, sys, threading
from pathlib import Path
from src.infra.docling_worker import main

original_read = Path.read_bytes
def blocked_read(path):
    if path.name == 'blocked-request.json':
        child = subprocess.Popen([sys.executable, '-c', 'import threading; threading.Event().wait()'],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        print(json.dumps({'worker': os.getpid(), 'child': child.pid}), flush=True)
        threading.Event().wait()
    return original_read(path)
Path.read_bytes = blocked_read
raise SystemExit(main())
''', encoding="utf-8")
    parent_script = tmp_path / "abrupt_parent.py"
    parent_script.write_text('''
import os, psutil, subprocess, sys
environment = dict(os.environ)
for key in ('DOCUMATE_CONVERSION_OWNER_PID', 'DOCUMATE_CONVERSION_OWNER_CREATED'):
    environment.pop(key, None)
if sys.argv[4] == 'owned':
    environment['DOCUMATE_CONVERSION_OWNER_PID'] = str(os.getpid())
    environment['DOCUMATE_CONVERSION_OWNER_CREATED'] = str(psutil.Process(os.getpid()).create_time())
worker = subprocess.Popen([sys.executable, sys.argv[1], '--request', sys.argv[2], '--result', sys.argv[3]],
    env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
print(worker.stdout.readline(), end='', flush=True)
os._exit(0)
''', encoding="utf-8")
    request = tmp_path / "blocked-request.json"
    request.write_text("{}", encoding="utf-8")
    processes = []
    try:
        environment = dict(runner_module.os.environ)
        environment["PYTHONPATH"] = str(runner_module.get_project_root_path())
        parent = subprocess.run([sys.executable, str(parent_script), str(worker_script), str(request), str(tmp_path / "result.json"),
                                 "owned" if expected_owner else "legacy"],
                                env=environment, capture_output=True, text=True, timeout=15,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        assert parent.returncode == 0
        identities = json.loads(parent.stdout)
        for pid in identities.values():
            try:
                processes.append(psutil.Process(pid))
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(processes, timeout=5)
        assert alive == []
    finally:
        for process in reversed(processes):
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(processes, timeout=5)


@pytest.mark.parametrize("owner", [
    {"DOCUMATE_CONVERSION_OWNER_PID": "missing"},
    {"DOCUMATE_CONVERSION_OWNER_CREATED": "1"},
    {"DOCUMATE_CONVERSION_OWNER_PID": "self", "DOCUMATE_CONVERSION_OWNER_CREATED": "invalid"},
    {"DOCUMATE_CONVERSION_OWNER_PID": "self", "DOCUMATE_CONVERSION_OWNER_CREATED": "nan"},
    {"DOCUMATE_CONVERSION_OWNER_PID": "self", "DOCUMATE_CONVERSION_OWNER_CREATED": "1"},
])
def test_worker_rejects_missing_malformed_or_reused_owner_identity(tmp_path, owner):
    """A worker with an unprovable expected owner exits before reading or converting its request."""
    policy = ConversionPolicy()
    request, result = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text(json.dumps({"files": [], "policy": policy.model_dump(mode="json"),
                                  "parser_config": policy.extraction_options()}), encoding="utf-8")
    environment = dict(os.environ)
    for key in ("DOCUMATE_CONVERSION_OWNER_PID", "DOCUMATE_CONVERSION_OWNER_CREATED"):
        environment.pop(key, None)
    environment.update({key: str(os.getpid()) if value == "self" else value for key, value in owner.items()})
    process = subprocess.run([sys.executable, "-m", "src.infra.docling_worker", "--request", str(request), "--result", str(result)],
                             env=environment, capture_output=True, timeout=15,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert process.returncode == 1
    assert not result.exists()


def test_worker_with_verified_expected_owner_reaches_normal_request_handling(tmp_path):
    """A live API owner's exact identity allows request handling despite a Windows redirector parent."""
    policy = ConversionPolicy()
    request, result = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text(json.dumps({"files": [], "policy": policy.model_dump(mode="json"),
                                  "parser_config": policy.extraction_options()}), encoding="utf-8")
    environment = {**os.environ, "DOCUMATE_CONVERSION_OWNER_PID": str(os.getpid()),
                   "DOCUMATE_CONVERSION_OWNER_CREATED": str(psutil.Process(os.getpid()).create_time())}
    process = subprocess.run([sys.executable, "-m", "src.infra.docling_worker", "--request", str(request), "--result", str(result)],
                             env=environment, capture_output=True, timeout=15,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert process.returncode == 1
    assert json.loads(result.read_bytes())["error"]["code"] == "DOCUMENT_ADAPTER_ERROR"
