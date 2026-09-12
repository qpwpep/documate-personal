from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException
from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from src.app.agent_manager import AgentFlowManager
from src.app.web.schemas import AgentRequest
from src.app.web.session_store import InMemorySessionStore
from src.app.web.upload_service import UploadService
from src.core.uploads import UploadAddition, UploadContext, UploadSyncRequest
from src.infra.settings import AppSettings
from src.infra.tools.local_rag import build_upload_search_tool


class DeterministicEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    from src.infra import runtime_paths
    from src.infra.tools.local_rag import client

    monkeypatch.setattr(runtime_paths, "get_project_root_path", lambda: tmp_path)
    monkeypatch.setattr("src.app.web.cleanup.get_project_root_path", lambda: tmp_path)
    monkeypatch.setattr("src.app.web.upload_service.get_project_root_path", lambda: tmp_path)
    monkeypatch.setattr(client, "build_openai_embeddings", lambda _key: DeterministicEmbeddings())
    settings = AppSettings(openai_api_key="test-key", tavily_api_key="test")
    store = InMemorySessionStore(settings, lambda: AgentFlowManager(settings))
    service = UploadService(settings=settings, session_store=store)
    yield service, store, tmp_path
    store.close_all()


def staged(root, name, text, *, session="session-a"):
    path = root / "uploads" / session / "staging" / uuid4().hex / name
    path.parent.mkdir(parents=True)
    path.write_bytes(text.encode("utf-8"))
    return UploadAddition(path=str(path), name=name)


def change(service, *, session="session-a", add=(), remove=(), clear=False, operation_id=None):
    manifest = service.get_manifest(session)
    return service.sync(session, UploadSyncRequest(
        epoch=manifest.epoch, expected_revision=manifest.revision,
        operation_id=operation_id or uuid4().hex, add=list(add), remove=list(remove), clear=clear,
    ))


def search(store, query, *, session="session-a"):
    handle = store.get_or_create(session).upload_retriever_handle
    return build_upload_search_tool()(query=query, k=10, retriever=handle.retriever if handle else None)


def test_addition_preserves_existing_files_and_both_sources_are_searchable(uploads):
    """Adding a second file keeps the first file and exposes both original sources."""
    service, store, root = uploads
    first = change(service, add=[staged(root, "alpha.py", "def alpha():\n    return 'ALPHA'\n")])
    first_path = Path(store.get_or_create("session-a")._ensure_session().upload_records[0].path)
    second = change(service, add=[staged(root, "beta.py", "def beta():\n    return 'BETA'\n")])
    assert [item.name for item in second.manifest.files] == ["alpha.py", "beta.py"]
    assert second.manifest.files[0] == first.manifest.files[0]
    assert first_path.is_file()
    assert {hit["evidence"]["snapshot"]["title"] for hit in search(store, "return")["hits"]} == {"alpha.py", "beta.py"}


def test_failed_batch_keeps_previous_manifest_and_search_results(uploads):
    """A malformed file prevents the whole batch from changing a working attachment set."""
    service, store, root = uploads
    before = change(service, add=[staged(root, "alpha.py", "def alpha():\n    return 1\n")]).manifest
    originals = {path for path in (root / "uploads" / "session-a" / "objects").rglob("*.py")}
    with pytest.raises(HTTPException):
        change(service, add=[staged(root, "beta.py", "beta = 2\n"), staged(root, "broken.ipynb", "not-json")])
    assert service.get_manifest("session-a") == before
    assert {path for path in (root / "uploads" / "session-a" / "objects").rglob("*.py")} == originals
    assert {hit["evidence"]["snapshot"]["title"] for hit in search(store, "alpha")["hits"]} == {"alpha.py"}


def test_replacement_keeps_file_identity_and_previous_citation_bytes(uploads):
    """Replacing a file changes its snapshot while an already returned citation keeps its source."""
    service, store, root = uploads
    original = "def alpha():\n    return 1\n"
    before = change(service, add=[staged(root, "alpha.py", original)]).manifest
    old_path = Path(store.get_or_create("session-a")._ensure_session().upload_records[0].path)
    old_evidence = search(store, "extract alpha function definition")["hits"][0]["evidence"]
    replacement = staged(root, "alpha.py", "def alpha():\n    return 2\n").model_copy(update={"replace_file_id": before.files[0].file_id})
    after = change(service, add=[replacement]).manifest
    assert after.files[0].file_id == before.files[0].file_id
    assert after.files[0].content_hash != before.files[0].content_hash
    assert not old_path.exists()
    assert old_evidence["element"]["text"] == original
    new_evidence = search(store, "extract alpha function definition")["hits"][0]["evidence"]
    assert new_evidence["snapshot"]["document_id"] == old_evidence["snapshot"]["document_id"]
    assert new_evidence["snapshot"]["snapshot_id"] != old_evidence["snapshot"]["snapshot_id"]


def test_identical_name_and_bytes_are_noop_but_new_name_preserves_provenance(uploads):
    """An identical resend is a no-op while another filename remains a distinct source."""
    service, _store, root = uploads
    content = "value = 1\n"
    before = change(service, add=[staged(root, "alpha.py", content)]).manifest
    noop = change(service, add=[staged(root, "ALPHA.py", content)])
    assert not noop.changed
    assert noop.manifest == before
    after = change(service, add=[staged(root, "beta.py", content)]).manifest
    assert len(after.files) == 2
    assert after.files[0].content_hash == after.files[1].content_hash
    assert after.files[0].file_id != after.files[1].file_id


def test_name_conflict_does_not_silently_replace_existing_file(uploads):
    """A changed file at an existing name requires an explicit replacement target."""
    service, _store, root = uploads
    before = change(service, add=[staged(root, "alpha.py", "value = 1\n")]).manifest
    with pytest.raises(HTTPException) as error:
        change(service, add=[staged(root, "alpha.py", "value = 2\n")])
    assert error.value.detail["code"] == "UPLOAD_NAME_CONFLICT"
    assert service.get_manifest("session-a") == before


def test_delete_and_clear_remove_search_sources_without_mutating_old_evidence(uploads):
    """Removal affects subsequent retrieval but keeps the original returned evidence intact."""
    service, store, root = uploads
    before = change(service, add=[staged(root, "alpha.py", "alpha = 1\n"), staged(root, "beta.py", "beta = 2\n")]).manifest
    evidence = search(store, "alpha")["hits"][0]["evidence"]
    saved = repr(evidence)
    after = change(service, remove=[before.files[0].file_id]).manifest
    assert [item.name for item in after.files] == ["beta.py"]
    assert {hit["evidence"]["snapshot"]["title"] for hit in search(store, "beta")["hits"]} == {"beta.py"}
    cleared = change(service, clear=True).manifest
    assert cleared.files == []
    assert store.get_or_create("session-a").upload_retriever_handle is None
    assert repr(evidence) == saved


def test_retry_is_idempotent_and_stale_changes_are_rejected(uploads):
    """Repeating one operation is safe and another stale operation cannot lose new files."""
    service, _store, root = uploads
    empty = service.get_manifest("session-a")
    request = UploadSyncRequest(epoch=empty.epoch, expected_revision=0, operation_id=uuid4().hex,
                                add=[staged(root, "alpha.py", "alpha = 1\n")])
    first = service.sync("session-a", request)
    assert service.sync("session-a", request) == first
    with pytest.raises(HTTPException) as error:
        service.sync("session-a", request.model_copy(update={"operation_id": uuid4().hex}))
    assert error.value.status_code == 409
    assert service.get_manifest("session-a") == first.manifest


def test_other_session_file_is_rejected_before_it_is_searchable(uploads):
    """A file stored by another session cannot enter this session's attachment set."""
    service, _store, root = uploads
    with pytest.raises(HTTPException):
        change(service, add=[staged(root, "secret.py", "secret = 1\n", session="session-b")])
    assert service.get_manifest("session-a").files == []


def test_new_upload_context_cannot_be_combined_with_legacy_path():
    """The two attachment protocols must not silently override each other."""
    with pytest.raises(ValidationError):
        AgentRequest(query="question", session_id="session-a", uploads=UploadContext(epoch="epoch", revision=1), upload_file_path="uploads/session-a/a.py")


def test_over_total_batch_stops_reading_and_preserves_active_state(uploads, monkeypatch):
    """An oversized batch stops allocating source buffers as soon as the aggregate limit is exceeded."""
    service, _store, root = uploads
    service.settings = service.settings.model_copy(update={"upload_max_total_mib": 1})
    before = service.get_manifest("session-a")
    additions = [staged(root, name, "#" + "x" * (600 * 1024)) for name in ("first.py", "second.py", "unread.py")]
    original_open = Path.open

    def file_boundary(path, *args, **kwargs):
        if path == Path(additions[2].path) and args and args[0] == "rb":
            raise AssertionError("The service read another file after exceeding the batch memory limit")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", file_boundary)
    with pytest.raises(HTTPException) as error:
        change(service, add=additions)
    assert error.value.status_code == 413
    assert error.value.detail["code"] == "UPLOAD_TOTAL_TOO_LARGE"
    assert service.get_manifest("session-a") == before
    assert not list((root / "uploads" / "session-a").glob("objects/*/*/*.py"))


def test_full_staging_space_does_not_block_individual_removal(uploads):
    """Removing an attachment frees its active storage even when abandoned staging files fill the allowance."""
    service, store, root = uploads
    service.settings = service.settings.model_copy(update={"upload_max_total_mib": 1})
    before = change(service, add=[staged(root, "first.py", "first = 1\n"), staged(root, "second.py", "second = 2\n")]).manifest
    with (root / "uploads" / "session-a" / "staging" / "orphan.bin").open("wb") as stream:
        stream.truncate(3 * 1024 * 1024)
    after = change(service, remove=[before.files[0].file_id]).manifest
    assert [file.name for file in after.files] == ["second.py"]
    assert {hit["evidence"]["snapshot"]["title"] for hit in search(store, "second")["hits"]} == {"second.py"}


def test_session_case_variants_share_one_manifest_lock_and_storage_path(uploads):
    """Case variants identify the same logical session consistently on every filesystem."""
    from src.infra.runtime_paths import get_upload_session_dir

    service, store, root = uploads
    original = change(service, session="CaseSession", add=[staged(root, "first.py", "first = 1\n", session="casesession")]).manifest
    assert service.get_manifest("casesession") == original
    assert store.get_or_create("CaseSession") is store.get_or_create("casesession")
    assert store.active_session_ids() == {"casesession"}
    assert store.active_agents["casesession"].active_request_count == 0
    assert get_upload_session_dir("CaseSession").name == "casesession"
    assert AgentRequest(query="question", session_id="CaseSession").session_id == "casesession"
    assert original.files[0].source_uri.startswith("upload:///casesession/")


@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_contents_in_one_batch_are_rejected_in_either_order(uploads, reverse):
    """An unchanged resend cannot hide a conflicting same-name replacement in its batch."""
    service, _store, root = uploads
    before = change(service, add=[staged(root, "same.py", "value = 1\n")]).manifest
    identical = staged(root, "SAME.py", "value = 1\n")
    replacement = staged(root, "same.py", "value = 2\n").model_copy(update={"replace_file_id": before.files[0].file_id})
    with pytest.raises(HTTPException) as error:
        change(service, add=[replacement, identical] if reverse else [identical, replacement])
    assert error.value.status_code == 409
    assert error.value.detail["code"] == "UPLOAD_NAME_CONFLICT"
    assert service.get_manifest("session-a") == before


def test_objects_directory_link_cannot_write_outside_the_session(uploads):
    """Managed storage rejects an objects directory redirected beyond its session boundary."""
    import os

    service, _store, root = uploads
    addition = staged(root, "first.py", "value = 1\n")
    outside = root / "outside"
    outside.mkdir()
    link = root / "uploads" / "session-a" / "objects"
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(outside), str(link))
    else:
        link.symlink_to(outside, target_is_directory=True)
    before = service.get_manifest("session-a")
    try:
        with pytest.raises(HTTPException) as error:
            change(service, add=[addition])
        assert error.value.detail["code"] == "UPLOAD_PATH_INVALID"
        assert list(outside.iterdir()) == []
        assert service.get_manifest("session-a") == before
    finally:
        if os.name == "nt":
            link.rmdir()
        else:
            link.unlink()


def _age_staging_directory(folder, timestamp):
    import os

    for path in folder.iterdir():
        os.utime(path, (timestamp, timestamp))
    os.utime(folder, (timestamp, timestamp))


def test_manifest_reclaims_expired_staging_but_preserves_objects_and_recent_writes(uploads):
    """An active session releases abandoned batches without removing its indexed files or in-progress writes."""
    import os
    import time

    service, store, root = uploads
    before = change(service, add=[staged(root, "active.py", "active = 1\n")]).manifest
    old = staged(root, "abandoned.py", "abandoned = 1\n")
    recent = staged(root, "recent.py", "recent = 1\n")
    writing = staged(root, "writing.py", "writing = 1\n")
    expired = time.time() - service.settings.session_ttl_seconds - 60
    _age_staging_directory(Path(old.path).parent, expired)
    _age_staging_directory(Path(writing.path).parent, expired)
    active_path = Path(store.get_or_create("session-a")._ensure_session().upload_records[0].path)
    os.utime(active_path, (expired, expired))
    partial = Path(writing.path).parent / ".upload-part"
    partial.write_text("new bytes", encoding="utf-8")
    # A recent child still protects the batch when its parent timestamp is stale.
    os.utime(partial.parent, (expired, expired))
    assert service.get_manifest("session-a") == before
    assert not Path(old.path).parent.exists()
    assert Path(recent.path).is_file()
    assert Path(writing.path).is_file() and partial.is_file()
    assert active_path.is_file()
    assert {hit["evidence"]["snapshot"]["title"] for hit in search(store, "active")["hits"]} == {"active.py"}


def test_sync_preserves_referenced_old_staging_and_reclaims_unreferenced_batches(uploads):
    """The operation holding the session lock keeps its input files while reclaiming unrelated expired batches."""
    import time

    service, _store, root = uploads
    manifest = service.get_manifest("session-a")
    referenced = staged(root, "referenced.py", "referenced = 1\n")
    abandoned = staged(root, "abandoned.py", "abandoned = 1\n")
    expired = time.time() - service.settings.session_ttl_seconds - 60
    for item in (referenced, abandoned):
        _age_staging_directory(Path(item.path).parent, expired)
    response = service.sync("session-a", UploadSyncRequest(
        epoch=manifest.epoch, expected_revision=manifest.revision, operation_id=uuid4().hex, add=[referenced]))
    assert [file.name for file in response.manifest.files] == ["referenced.py"]
    assert Path(referenced.path).is_file()
    assert not Path(abandoned.path).parent.exists()


def test_staging_cleanup_skips_external_directory_links(uploads):
    """Expired staging cleanup never traverses a junction or symlink to external files."""
    import os
    import time

    service, _store, root = uploads
    abandoned = staged(root, "abandoned.py", "abandoned = 1\n")
    external = root / "external-staging"
    external.mkdir()
    sentinel = external / "sentinel.py"
    sentinel.write_text("keep = 1\n", encoding="utf-8")
    expired = time.time() - service.settings.session_ttl_seconds - 60
    _age_staging_directory(external, expired)
    _age_staging_directory(Path(abandoned.path).parent, expired)
    link = Path(abandoned.path).parent.parent / "external-link"
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(external), str(link))
    else:
        link.symlink_to(external, target_is_directory=True)
    try:
        service.get_manifest("session-a")
        assert sentinel.read_text(encoding="utf-8") == "keep = 1\n"
        assert link.exists()
        assert not Path(abandoned.path).parent.exists()
    finally:
        if os.name == "nt":
            link.rmdir()
        else:
            link.unlink()


@pytest.mark.parametrize("termination", ["close", "ttl", "lru", "shutdown"])
def test_session_termination_releases_owned_files_and_preserves_input_and_citations(uploads, termination):
    """Every session termination releases managed originals while borrowed inputs and returned evidence survive."""
    service, store, root = uploads
    addition = staged(root, "source.py", "value = 1\n")
    previous = change(service, add=[addition]).manifest
    agent = store.get_or_create("session-a")
    managed = Path(agent._ensure_session().upload_records[0].path)
    evidence = search(store, "value")["hits"][0]["evidence"]
    saved = repr(evidence)

    if termination == "close":
        agent.close()
    elif termination == "ttl":
        entry = store.get_or_create_entry("session-a")
        store.cleanup_expired(now=entry.last_accessed_monotonic + 10, ttl_seconds=1)
    elif termination == "lru":
        store.evict_lru_if_needed(max_active_sessions=0)
    else:
        store.close_all()

    assert not managed.exists()
    assert Path(addition.path).is_file()
    assert repr(evidence) == saved
    current = service.get_manifest("session-a")
    assert current.files == [] and current.revision == 0
    assert current.epoch != previous.epoch


def test_manifest_reclaims_orphan_objects_but_preserves_active_files_and_staging(uploads):
    """A recreated session can reclaim unreferenced managed versions without deleting live sources or UI inputs."""
    service, store, root = uploads
    before = change(service, add=[staged(root, "active.py", "active = 1\n")]).manifest
    active = Path(store.get_or_create("session-a")._ensure_session().upload_records[0].path)
    orphan = root / "uploads" / "session-a" / "objects" / uuid4().hex / uuid4().hex / "orphan.py"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("orphan = 1\n", encoding="utf-8")
    pending = staged(root, "pending.py", "pending = 1\n")

    assert service.get_manifest("session-a") == before

    assert not orphan.exists()
    assert active.is_file() and Path(pending.path).is_file()
    assert {hit["evidence"]["snapshot"]["title"] for hit in search(store, "active")["hits"]} == {"active.py"}


def test_failed_file_deletion_is_retried_after_committing_empty_manifest(uploads, monkeypatch):
    """A transient unlink failure cannot permanently charge storage to an empty attachment set."""
    service, store, root = uploads
    change(service, add=[staged(root, "source.py", "value = 1\n")])
    managed = Path(store.get_or_create("session-a")._ensure_session().upload_records[0].path)
    unlink = Path.unlink
    blocked = True

    def fail_managed_once(path, *args, **kwargs):
        nonlocal blocked
        if path == managed and blocked:
            blocked = False
            raise PermissionError("temporary file lock")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_managed_once)
    cleared = change(service, clear=True).manifest
    assert cleared.files == [] and managed.exists()

    assert service.get_manifest("session-a") == cleared
    assert not managed.exists()


def test_retired_index_cleanup_failure_is_retried_without_losing_committed_sources(uploads, monkeypatch):
    """A failed collection deletion is retried during later housekeeping while the committed generation stays searchable."""
    from langchain_chroma import Chroma

    service, store, root = uploads
    change(service, add=[staged(root, "first.py", "first = 1\n")])
    retired = store.get_or_create("session-a").upload_retriever_handle
    database = retired._vectorstore._client
    delete = Chroma.delete_collection
    blocked = True

    def fail_retired_once(vectorstore):
        nonlocal blocked
        if vectorstore._collection.name == retired.collection_name and blocked:
            blocked = False
            raise RuntimeError("temporary collection deletion failure")
        return delete(vectorstore)

    monkeypatch.setattr(Chroma, "delete_collection", fail_retired_once)
    committed = change(service, add=[staged(root, "second.py", "second = 2\n")]).manifest
    assert retired.collection_name in {collection.name for collection in database.list_collections()}

    assert service.get_manifest("session-a") == committed
    assert retired.collection_name not in {collection.name for collection in database.list_collections()}
    assert {hit["evidence"]["snapshot"]["title"] for hit in search(store, "value")["hits"]} == {"first.py", "second.py"}


@pytest.mark.parametrize("legacy", [False, True])
def test_current_operation_can_borrow_an_orphan_managed_path_before_reconciliation(uploads, legacy):
    """Both upload protocols protect their current input even when it lives in the objects namespace."""
    service, store, root = uploads
    before = service.get_manifest("session-a")
    source = root / "uploads" / "session-a" / "objects" / uuid4().hex / uuid4().hex / "borrowed.py"
    source.parent.mkdir(parents=True)
    source.write_text("borrowed = 1\n", encoding="utf-8")
    relative = str(source.relative_to(root))

    if legacy:
        with store.locked_session("session-a") as (entry, _wait_ms):
            service.sync_legacy_locked("session-a", entry.agent, relative)
            current = entry.agent._ensure_session().upload_manifest()
    else:
        current = service.sync("session-a", UploadSyncRequest(
            epoch=before.epoch, expected_revision=before.revision, operation_id=uuid4().hex,
            add=[UploadAddition(path=relative, name=source.name)])).manifest

    assert [item.name for item in current.files] == ["borrowed.py"]
    assert source.is_file()
    assert {hit["evidence"]["snapshot"]["title"] for hit in search(store, "borrowed")["hits"]} == {"borrowed.py"}


def test_cleanup_backlog_survives_session_eviction_and_blocks_new_indexes_until_recovered(uploads, monkeypatch):
    """Unreleased collections outlive an evicted context and cannot accumulate through continued index builds."""
    service, store, root = uploads
    change(service, add=[staged(root, "source.py", "source = 1\n")])
    retired = store.get_or_create("session-a").upload_retriever_handle
    database = retired._vectorstore._client
    delete = database.delete_collection
    blocked = True

    def fail_while_blocked(name):
        if name == retired.collection_name and blocked:
            raise RuntimeError("collection storage unavailable")
        return delete(name)

    monkeypatch.setattr(database, "delete_collection", fail_while_blocked)
    store.evict_lru_if_needed(max_active_sessions=0)
    assert store.active_session_ids() == set()
    assert retired.retriever.source_documents == ()
    try:
        with pytest.raises(HTTPException) as error:
            change(service, add=[staged(root, "new.py", "new = 2\n")])
        assert error.value.status_code == 503
        assert service.get_manifest("session-a").files == []
        assert {collection.name for collection in database.list_collections()} == {retired.collection_name}
    finally:
        blocked = False
        service.get_manifest("session-a")

    current = change(service, add=[staged(root, "new.py", "new = 2\n")]).manifest
    assert [item.name for item in current.files] == ["new.py"]
    assert retired.collection_name not in {collection.name for collection in database.list_collections()}


def test_managed_collection_retry_does_not_delete_a_reused_name(uploads, monkeypatch):
    """A delayed cleanup releases only the failed collection identity, never a later collection sharing its name."""
    service, store, root = uploads
    change(service, add=[staged(root, "source.py", "source = 1\n")])
    retired = store.get_or_create("session-a").upload_retriever_handle
    database = retired._vectorstore._client
    delete = database.delete_collection

    def fail_deletion(name):
        raise RuntimeError("blocked")

    with monkeypatch.context() as failure:
        failure.setattr(database, "delete_collection", fail_deletion)
        store.close_all()
    delete(retired.collection_name)
    replacement = database.create_collection(retired.collection_name)
    try:
        service.get_manifest("session-a")
        retired.cleanup()
        assert database.get_collection(retired.collection_name).id == replacement.id
    finally:
        if retired.collection_name in {collection.name for collection in database.list_collections()}:
            delete(retired.collection_name)


def test_session_storage_owner_cannot_be_rebound_to_another_session(uploads):
    """A context cannot transfer cleanup authority to a different session's storage."""
    service, store, root = uploads
    before = change(service, add=[staged(root, "owned.py", "owned = 1\n")]).manifest
    session = store.get_or_create("session-a")._ensure_session()
    with pytest.raises(ValueError, match="another session"):
        session.bind_upload_storage("session-b")
    assert session.upload_manifest() == before
    assert Path(session.upload_records[0].path).is_file()
