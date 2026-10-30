export class ApiError extends Error {
  readonly code: string;
  readonly requestId?: string;
  readonly status?: number;

  constructor(code: string, message: string, details: { requestId?: string; status?: number; cause?: unknown } = {}) {
    super(message, { cause: details.cause });
    this.name = 'ApiError';
    this.code = code;
    this.requestId = details.requestId;
    this.status = details.status;
  }
}
