export type UploadFailure = { name?: string; code?: string; message?: string; file_id?: string };

export class ApiError extends Error {
  readonly code: string;
  readonly executionId?: string;
  readonly status?: number;
  readonly files: UploadFailure[];

  constructor(code: string, message: string, details: { executionId?: string; status?: number; cause?: unknown; files?: UploadFailure[] } = {}) {
    super(message, { cause: details.cause });
    this.name = 'ApiError';
    this.code = code;
    this.executionId = details.executionId;
    this.status = details.status;
    this.files = details.files ?? [];
  }
}
