export type UploadFailure = { name?: string; code?: string; message?: string; file_id?: string };

export class ApiError extends Error {
  readonly code: string;
  readonly requestId?: string;
  readonly status?: number;
  readonly files: UploadFailure[];

  constructor(code: string, message: string, details: { requestId?: string; status?: number; cause?: unknown; files?: UploadFailure[] } = {}) {
    super(message, { cause: details.cause });
    this.name = 'ApiError';
    this.code = code;
    this.requestId = details.requestId;
    this.status = details.status;
    this.files = details.files ?? [];
  }
}
