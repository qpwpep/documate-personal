import type { BrowserUploadSyncRequest, StagedUpload, UploadManifest } from '../api/types';
import { uploadCasefoldExceptions } from './casefold.generated';

export interface AttachmentState {
  files: File[];
  phase: 'idle' | 'staging' | 'review' | 'syncing' | 'failed' | 'ready';
  staged: StagedUpload[];
  reviewManifest: UploadManifest | null;
  operation: BrowserUploadSyncRequest | null;
  approvedFileIds: string[];
  removeIds: string[];
  clearAll: boolean;
  pendingPrompt: string | null;
  hadFailure: boolean;
  needsRefreshReview: boolean;
  restoreFilesRequired: boolean;
  errors: string[];
  fileErrors: Record<number, string[]>;
  notice: string | null;
}

export function createAttachmentState(): AttachmentState {
  return {
    files: [], phase: 'idle', staged: [], reviewManifest: null, operation: null,
    approvedFileIds: [], removeIds: [], clearAll: false, pendingPrompt: null, hadFailure: false,
    needsRefreshReview: false, restoreFilesRequired: false, errors: [], fileErrors: {}, notice: null,
  };
}

export function attachmentName(name: string): string {
  return Array.from(name.normalize('NFC'), (character) => uploadCasefoldExceptions[character] ?? character.toLowerCase()).join('');
}

export function replacementFiles(state: AttachmentState) {
  return state.staged.flatMap((staged) => {
    const current = state.reviewManifest?.files.find((file) => attachmentName(file.name) === attachmentName(staged.name));
    return current && current.content_hash !== staged.content_hash ? [{ staged, current }] : [];
  });
}
