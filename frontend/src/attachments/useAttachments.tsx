import { useRef } from 'react';
import { AgentSessionClient, ApiError } from '../api/client';
import type { BrowserUploadSyncRequest, ClientConfig, UploadManifest } from '../api/types';
import { AttachmentManifest, AttachmentPicker, Attachments } from '../components/Attachments';
import { createId } from '../id';
import { attachmentName, createAttachmentState, replacementFiles, type AttachmentState } from './state';

interface Options {
  client: AgentSessionClient;
  config: ClientConfig | null;
  state: AttachmentState;
  onChange(state: AttachmentState): void;
  manifest: UploadManifest | null;
  manifestConfirmed: boolean;
  ensureSession(): Promise<UploadManifest | null>;
  onManifestChange(manifest: UploadManifest | null): void;
  onBusyChange(busy: boolean): void;
  onBeforeRequest?(): void;
  onSendPrompt(prompt: string): void;
  onRestorePrompt(prompt: string): boolean;
  disabled: boolean;
}

export function useAttachments(options: Options) {
  const active = useRef(false);
  const pickerButton = useRef<HTMLButtonElement>(null);

  function change(state: AttachmentState) {
    options.onChange(state);
  }

  async function run(initial: AttachmentState, retry: boolean) {
    if (active.current || options.disabled || !options.config) return;
    active.current = true;
    options.onBusyChange(true);
    let state = { ...initial, errors: [], fileErrors: {}, notice: null };
    change({ ...state, phase: 'staging' });
    options.onBeforeRequest?.();
    try {
      const manifest = await options.ensureSession();
      if (!manifest) throw new Error('첨부 목록을 확인하지 못했습니다. 연결을 확인해 주세요.');
      options.onManifestChange(manifest);
      const changedSinceReview = state.reviewManifest !== null && (
        state.reviewManifest.epoch !== manifest.epoch || state.reviewManifest.revision !== manifest.revision
      );
      state = { ...state, reviewManifest: manifest, approvedFileIds: changedSinceReview ? [] : state.approvedFileIds };
      if (retry && state.operation) {
        change({ ...state, phase: 'syncing' });
        complete(await options.client.syncUploads(state.operation), state);
        return;
      }
      if (state.needsRefreshReview && (state.clearAll || state.removeIds.length > 0)) {
        change({ ...state, phase: 'review', needsRefreshReview: false });
        return;
      }
      const { errors, fileErrors } = selectionErrors(state.files, manifest, options.config);
      if (errors.length) {
        change({ ...state, phase: 'failed', hadFailure: true, errors, fileErrors });
        return;
      }
      state = { ...state, phase: 'staging', needsRefreshReview: false };
      change(state);
      for (const file of state.files.slice(state.staged.length)) {
        const staged = await options.client.stageFile(file);
        state = { ...state, staged: [...state.staged, staged] };
        change(state);
      }
      const duplicateNames = new Map<string, string>();
      for (const file of state.staged) {
        const key = attachmentName(file.name);
        if (duplicateNames.has(key) && duplicateNames.get(key) !== file.content_hash) {
          throw new Error(`${file.name}: 한 번에 같은 이름의 서로 다른 파일을 추가할 수 없습니다.`);
        }
        duplicateNames.set(key, file.content_hash);
      }
      const replacements = replacementFiles(state);
      if (replacements.some(({ current }) => !state.approvedFileIds.includes(current.file_id))) {
        change({ ...state, phase: 'review' });
        return;
      }
      const additions: NonNullable<BrowserUploadSyncRequest['add']> = [];
      const addedNames = new Set<string>();
      for (const file of state.staged) {
        const name = attachmentName(file.name);
        const current = manifest.files.find((existing) => attachmentName(existing.name) === name);
        // Sync also releases unchanged and duplicate staged files. Only the first
        // addition for a changed name owns the approved replacement.
        const replaceFileId = current && current.content_hash !== file.content_hash && !addedNames.has(name)
          ? current.file_id : null;
        additions.push({
          upload_id: file.upload_id, name: file.name, content_hash: file.content_hash,
          replace_file_id: replaceFileId,
        });
        addedNames.add(name);
      }
      const remove = state.removeIds.filter((fileId) => manifest.files.some((file) => file.file_id === fileId));
      if (!additions.length && !remove.length && !state.clearAll) {
        complete(manifest, { ...state, notice: '동일한 파일은 기존 첨부를 유지했습니다.' });
        return;
      }
      const operation: BrowserUploadSyncRequest = {
        epoch: manifest.epoch, expected_revision: manifest.revision, operation_id: createId(),
        add: additions, remove, clear: state.clearAll,
      };
      state = { ...state, phase: 'syncing', operation };
      change(state);
      complete(await options.client.syncUploads(operation), state);
    } catch (error) {
      const conflict = error instanceof ApiError && ['UPLOAD_REVISION_CONFLICT', 'UPLOAD_EPOCH_MISMATCH', 'UPLOAD_NAME_CONFLICT', 'UPLOAD_FILE_NOT_FOUND'].includes(error.code.toUpperCase());
      if (options.client.manifest === null) options.onManifestChange(null);
      change({
        ...state, phase: 'failed', hadFailure: true, needsRefreshReview: conflict,
        operation: conflict ? null : state.operation,
        approvedFileIds: conflict ? [] : state.approvedFileIds,
        errors: [error instanceof Error ? error.message : '첨부 변경을 완료하지 못했습니다. 다시 확인해 주세요.'],
        fileErrors: error instanceof ApiError ? Object.fromEntries(state.files.map((file, index) => [index,
          error.files.filter((failure) => failure.name === file.name).map((failure) => failure.message ?? '파일을 처리하지 못했습니다.'),
        ])) : state.fileErrors,
      });
    } finally {
      if (active.current) {
        active.current = false;
        options.onBusyChange(false);
      }
    }
  }

  function complete(manifest: UploadManifest, state: AttachmentState) {
    options.onManifestChange(manifest);
    const holdPrompt = state.hadFailure && state.pendingPrompt !== null;
    change({
      ...createAttachmentState(), phase: holdPrompt ? 'ready' : 'idle',
      pendingPrompt: holdPrompt ? state.pendingPrompt : null,
      notice: state.notice ?? '첨부 변경을 확정했습니다.',
    });
    if (!state.hadFailure && state.pendingPrompt !== null) {
      active.current = false;
      options.onBusyChange(false);
      options.onSendPrompt(state.pendingPrompt);
    }
  }

  async function submit(prompt: string | null) {
    if (active.current || options.disabled || options.state.phase !== 'idle') return;
    await run({ ...options.state, pendingPrompt: prompt, hadFailure: false }, false);
  }

  function recover(reselect: boolean) {
    if (active.current || options.disabled) return;
    const prompt = options.state.pendingPrompt;
    const restored = prompt === null || options.onRestorePrompt(prompt);
    const recoveryNotice = reselect ? '파일을 다시 선택하고 질문을 확인한 뒤 보내 주세요.' : '기존 첨부를 유지했습니다. 질문을 확인한 뒤 보내 주세요.';
    change({
      ...createAttachmentState(), phase: restored ? 'idle' : 'ready', pendingPrompt: restored ? null : prompt,
      notice: restored ? recoveryNotice : '작성 중인 질문을 유지했습니다. 보류한 질문을 보내거나 해제한 뒤 계속할 수 있습니다. 보류 질문과 첨부를 수정하려면 먼저 입력창을 비워 주세요.',
    });
    if (reselect && restored) requestAnimationFrame(() => pickerButton.current?.focus());
  }

  async function removeFiles(fileId: string | null) {
    await run({ ...createAttachmentState(), removeIds: fileId ? [fileId] : [], clearAll: fileId === null }, false);
  }

  async function refreshManifest() {
    if (active.current || options.disabled) return;
    active.current = true;
    options.onBusyChange(true);
    try {
      options.onManifestChange(await options.ensureSession());
    } finally {
      active.current = false;
      options.onBusyChange(false);
    }
  }

  function sendPending() {
    if (active.current || options.disabled || !options.manifest || options.state.restoreFilesRequired || options.state.pendingPrompt === null) return;
    const prompt = options.state.pendingPrompt;
    change(createAttachmentState());
    options.onSendPrompt(prompt);
  }

  const canChange = !options.disabled && !active.current && options.state.phase === 'idle' && options.state.pendingPrompt === null;
  const picker = <AttachmentPicker config={options.config} disabled={!canChange} buttonRef={pickerButton}
    manifest={options.manifest} confirmed={options.manifestConfirmed}
    onSelect={(files) => change({ ...createAttachmentState(), files: [...options.state.files, ...files] })} />;
  const manifestPanel = <AttachmentManifest manifest={options.manifest} confirmed={options.manifestConfirmed}
    disabled={options.disabled || active.current} changesDisabled={!canChange || options.state.files.length > 0}
    onRefresh={() => { void refreshManifest(); }}
    onRemove={(fileId) => { void removeFiles(fileId); }} onClear={() => { void removeFiles(null); }} />;
  const panel = <Attachments state={options.state} manifest={options.manifest} disabled={options.disabled || active.current}
    onRemoveSelected={(index) => change({
      ...createAttachmentState(), files: options.state.files.filter((_, position) => position !== index),
      phase: options.state.hadFailure ? 'failed' : 'idle', pendingPrompt: options.state.pendingPrompt,
      hadFailure: options.state.hadFailure,
      notice: options.state.hadFailure ? '남은 파일을 다시 확인해 주세요. 보류한 질문은 자동으로 보내지 않습니다.' : null,
    })}
    onApprove={(fileId, checked) => change({ ...options.state, approvedFileIds: checked ? [...options.state.approvedFileIds, fileId] : options.state.approvedFileIds.filter((id) => id !== fileId) })}
    onSubmit={() => { void submit(null); }} onConfirm={() => { void run(options.state, false); }}
    onRetry={() => { void run(options.state, true); }} onRecover={recover}
    onSendPending={sendPending} onEditPending={() => recover(true)}
    onDiscardPending={() => change(createAttachmentState())} />;
  return { submit, panel, picker, manifestPanel, hasPending: options.state.phase !== 'idle' || options.state.pendingPrompt !== null };
}

function selectionErrors(files: File[], manifest: UploadManifest, config: ClientConfig) {
  const errors: string[] = [];
  const fileErrors: Record<number, string[]> = {};
  const selectedSizes = new Map<string, number>();
  for (const [index, file] of files.entries()) {
    const name = attachmentName(file.name);
    selectedSizes.set(name, Math.max(selectedSizes.get(name) ?? 0, file.size));
    const suffix = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
    const messages = [];
    if (!config.upload.accepted_suffixes.includes(suffix)) messages.push('지원하지 않는 파일 형식입니다.');
    if (file.size > config.upload.max_file_bytes) messages.push('파일당 크기 한도를 넘었습니다.');
    if (messages.length) {
      fileErrors[index] = messages;
      errors.push(...messages.map((message) => `${file.name}: ${message}`));
    }
  }
  // One normalized name can occupy only one final slot. Keep every selected file
  // for staging: content hashes, not file metadata, still decide duplicates/conflicts.
  const retained = manifest.files.filter((file) => !selectedSizes.has(attachmentName(file.name)));
  if (selectedSizes.size + retained.length > config.upload.max_files) errors.push(`한 대화에는 최대 ${config.upload.max_files}개 파일을 첨부할 수 있습니다.`);
  if ([...selectedSizes.values()].reduce((total, size) => total + size, 0) + retained.reduce((total, file) => total + file.size_bytes, 0) > config.upload.max_total_bytes) errors.push('전체 첨부 크기 한도를 넘었습니다.');
  return { errors, fileErrors };
}
