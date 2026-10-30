import { useId, useRef, type RefObject } from 'react';
import type { ClientConfig, UploadManifest } from '../api/types';
import { replacementFiles, type AttachmentState } from '../attachments/state';
import './Attachments.css';

interface Props {
  state: AttachmentState;
  manifest: UploadManifest | null;
  disabled: boolean;
  onRemoveSelected(index: number): void;
  onApprove(fileId: string, checked: boolean): void;
  onSubmit(): void;
  onConfirm(): void;
  onRetry(): void;
  onRecover(reselect: boolean): void;
  onSendPending(): void;
  onEditPending(): void;
  onDiscardPending(): void;
}

export function Attachments(props: Props) {
  const errorId = useId();
  const { state, manifest, disabled } = props;
  const working = state.phase === 'staging' || state.phase === 'syncing';
  const replacements = replacementFiles(state);
  const canChange = !disabled && !working && state.phase === 'idle' && !state.pendingPrompt;
  const canRemoveSelected = canChange || (!disabled && state.phase === 'failed');
  if (!state.restoreFilesRequired && !state.files.length && !state.errors.length && !state.notice
      && state.phase === 'idle' && state.pendingPrompt === null) return null;
  return <section className="attachments" aria-label="대화 첨부">
    {state.restoreFilesRequired && <p className="notice">선택했던 파일은 새로고침 후 다시 골라야 합니다. 아직 첨부하거나 질문을 보내지 않았습니다.</p>}
    {state.files.length > 0 && <div className="selected-files">
      <p className="attachment-state">{working ? '첨부 준비 중' : state.phase === 'failed' ? '선택 오류 또는 첨부 실패' : '전송 대기'}</p>
      <ul>{state.files.map((file, index) => <li key={`${index}-${file.name}`}>
        <span>{file.name} <small>{formatBytes(file.size)}</small>
          {state.fileErrors[index]?.map((error) => <small className="selected-file-error" key={error}>선택 오류: {error}</small>)}
        </span>
        <button type="button" className="text-button" aria-label={`${file.name} 선택 취소 ${index + 1}`} disabled={!canRemoveSelected} onClick={() => props.onRemoveSelected(index)}>선택 취소</button>
      </li>)}</ul>
      {canChange && <button type="button" className="secondary-button" onClick={props.onSubmit}>첨부만 확정</button>}
    </div>}
    {state.errors.length > 0 && <div id={errorId} role="alert" className="notice">{state.errors.map((error, index) => <p key={index}>{error}</p>)}</div>}
    {state.notice && <p role="status" className="attachment-hint">{state.notice}</p>}
    {state.phase === 'review' && <div className="attachment-review">
      <p>{replacements.length ? '같은 이름의 다른 파일입니다. 기존 첨부를 교체할지 확인해 주세요.' : '최신 첨부 목록을 확인했습니다. 선택한 첨부 해제를 다시 확정해 주세요.'} 과거 답변의 근거는 그대로 보존됩니다.</p>
      {state.clearAll && <p>현재 첨부 {manifest?.files.length ?? 0}개를 모두 해제합니다.</p>}
      {state.removeIds.length > 0 && <p>{state.reviewManifest?.files.filter((file) => state.removeIds.includes(file.file_id)).map((file) => file.name).join(', ')} 첨부를 해제합니다.</p>}
      {replacements.map(({ current, staged }) => <label key={staged.upload_id}>
        <input type="checkbox" checked={state.approvedFileIds.includes(current.file_id)} disabled={disabled}
          onChange={(event) => props.onApprove(current.file_id, event.target.checked)} />
        {staged.name} 교체 승인
      </label>)}
      <button type="button" disabled={disabled || replacements.some(({ current }) => !state.approvedFileIds.includes(current.file_id))} onClick={props.onConfirm}>첨부 변경 확정</button>
    </div>}
    {state.phase === 'failed' && <button type="button" disabled={disabled} onClick={props.onRetry}>{state.needsRefreshReview ? '최신 첨부로 다시 검토' : '첨부 다시 시도'}</button>}
    {(state.phase === 'failed' || state.phase === 'review' || state.restoreFilesRequired) && <div className="attachment-recovery">
      <button type="button" className="secondary-button" disabled={disabled} onClick={() => props.onRecover(true)}>파일 다시 선택</button>
      <button type="button" className="secondary-button" disabled={disabled} onClick={() => props.onRecover(false)}>{manifest?.files.length ? '기존 자료로 계속' : '첨부 없이 계속'}</button>
    </div>}
    {state.pendingPrompt !== null && <div className="held-question">
      <p>첨부를 위해 보류한 질문</p><pre>{state.pendingPrompt}</pre>
      {(state.phase === 'ready' || state.phase === 'idle') && <div className="attachment-recovery">
        <button type="button" disabled={disabled || !manifest || state.restoreFilesRequired} onClick={props.onSendPending}>보류한 질문 보내기</button>
        <button type="button" className="secondary-button" disabled={disabled} onClick={props.onEditPending}>질문과 첨부 수정</button>
        <button type="button" className="text-button" disabled={disabled} onClick={props.onDiscardPending}>보류 질문 해제</button>
      </div>}
    </div>}
  </section>;
}

interface PickerProps {
  config: ClientConfig | null;
  disabled: boolean;
  buttonRef: RefObject<HTMLButtonElement | null>;
  onSelect(files: File[]): void;
}

export function AttachmentPicker({ config, disabled, buttonRef, onSelect }: PickerProps) {
  const inputId = useId();
  const input = useRef<HTMLInputElement>(null);
  const guidance = config
    ? `${config.upload.accepted_suffixes.join(' · ')} · 최대 ${config.upload.max_files}개 · 파일당 ${formatBytes(config.upload.max_file_bytes)}`
    : '서버 설정을 확인하고 있습니다.';
  return <div className="attachment-picker">
    <button ref={buttonRef} type="button" className="attachment-add" aria-label="파일 추가" title={guidance}
      disabled={disabled || !config} aria-describedby={`${inputId}-hint`} onClick={() => input.current?.click()}>
      <span aria-hidden="true">+</span>
    </button>
    <label className="visually-hidden" htmlFor={inputId}>파일 첨부</label>
    <input id={inputId} ref={input} type="file" multiple className="visually-hidden" tabIndex={-1}
      accept={config?.upload.accepted_suffixes.join(',')} disabled={disabled || !config}
      onChange={(event) => { onSelect(Array.from(event.target.files ?? [])); event.target.value = ''; }} />
    <span className="visually-hidden" id={`${inputId}-hint`}>{guidance}</span>
  </div>;
}

interface ManifestProps {
  manifest: UploadManifest | null;
  confirmed: boolean;
  disabled: boolean;
  changesDisabled: boolean;
  onRefresh(): void;
  onRemove(fileId: string): void;
  onClear(): void;
}

export function AttachmentManifest({ manifest, confirmed, disabled, changesDisabled, onRefresh, onRemove, onClear }: ManifestProps) {
  return <section className="attachments attachment-manifest" aria-label="함께 검색할 파일">
    <h2 className="sidebar-section-title">함께 검색할 파일</h2>
    <button type="button" className="secondary-button" disabled={disabled} onClick={onRefresh}>첨부 목록 새로고침</button>
    {manifest ? <>
      <p className="attachment-state"><span>{confirmed ? '검색 가능' : '이전 첨부 기록'}</span> {manifest.files.length}개 · {(manifest.files.reduce((sum, file) => sum + file.size_bytes, 0) / 1024 / 1024).toFixed(2)} MiB</p>
      {!confirmed && <p className="attachment-hint">이 목록은 이전에 확인한 기록입니다. 현재 서버에서 사용할 수 있는지 다시 확인해야 합니다.</p>}
      {manifest.files.length ? <div className="confirmed-files">
        <ul>{manifest.files.map((file) => <li key={file.file_id}>
          <span><strong>{file.name}</strong> <small>{formatBytes(file.size_bytes)} · {confirmed ? '검색 가능' : '이전 첨부 기록'}</small></span>
          <button type="button" className="text-button" aria-label={`${file.name} 첨부 해제`} disabled={!confirmed || changesDisabled} onClick={() => onRemove(file.file_id)}>삭제</button>
        </li>)}</ul>
        <button type="button" className="text-button" aria-label="첨부 전체 해제" disabled={!confirmed || changesDisabled} onClick={onClear}>전체 첨부 해제</button>
        <p className="attachment-hint">첨부를 해제해도 대화와 기존 답변의 인용은 유지됩니다.</p>
      </div> : <p className="attachment-hint">검색 준비가 끝난 파일이 없습니다. 입력창에서 선택한 파일은 전송 후 여기에 표시됩니다.</p>}
    </> : <p className="attachment-hint">첨부 상태를 확인하지 못했습니다. 서버 목록을 확인한 뒤 질문할 수 있습니다.</p>}
  </section>;
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} B`;
}
