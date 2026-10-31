import { ApiError, type UploadFailure } from './errors';
import { parseAnswerExport, parseSlackDefault } from './delivery-validation';
import { parseSseStream } from './sse';
import type { AgentRequest, AgentStreamEvent, AnswerExportResult, BrowserUploadSyncRequest, ClientConfig, RecipientInput, SlackDefault, StagedUpload, UploadManifest } from './types';
import { parseAgentEvent, parseClientConfig, parseStagedUpload, parseUploadManifest } from './validation';

export { ApiError } from './errors';

type RequestOptions = { signal?: AbortSignal };
type StreamOptions = RequestOptions & { slackRecipient?: RecipientInput | null };

// Session JSON requests use a total timeout; streams reset it on incoming bytes.
const DEFAULT_SESSION_TIMEOUT_MS = 60_000;
const UPLOAD_TIMEOUT_MS = 180_000;
const DEFAULT_LOOKUP_TIMEOUT_MS = 15_000;

export async function fetchClientConfig(baseUrl = '', options: RequestOptions = {}): Promise<ClientConfig> {
  return parseClientConfig(await requestJson(`${baseUrl.replace(/\/$/, '')}/client-config`, {}, options));
}

export async function fetchSlackDefault(baseUrl = '', options: RequestOptions = {}): Promise<SlackDefault> {
  return parseSlackDefault(await requestJson(`${baseUrl.replace(/\/$/, '')}/slack/default-recipient`, {}, options));
}

export class AgentSessionClient {
  manifest: UploadManifest | null = null;
  readonly sessionId: string;
  private readonly baseUrl: string;
  private readonly timeoutMs: number;

  constructor({ sessionId, baseUrl = '', timeoutMs = DEFAULT_SESSION_TIMEOUT_MS }: { sessionId: string; baseUrl?: string; timeoutMs?: number }) {
    if (!/^[A-Za-z0-9_-]{1,128}$/.test(sessionId)) {
      throw new ApiError('invalid_session', '대화 식별자가 올바르지 않습니다.');
    }
    this.sessionId = sessionId.toLowerCase();
    this.baseUrl = baseUrl.replace(/\/$/, '');
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new RangeError('timeoutMs must be positive');
    this.timeoutMs = timeoutMs;
  }

  async refreshUploads(options: RequestOptions = {}): Promise<UploadManifest> {
    this.manifest = null;
    const value = await requestJson(`${this.baseUrl}/sessions/${encodeURIComponent(this.sessionId)}/uploads`, {}, options, this.timeoutMs);
    this.manifest = parseUploadManifest(value);
    return this.manifest;
  }

  async stageFile(file: File, options: RequestOptions = {}): Promise<StagedUpload> {
    const body = new FormData();
    body.append('file', file, file.name);
    const result = await requestJson(`${this.baseUrl}/sessions/${encodeURIComponent(this.sessionId)}/uploads/staging`, {
      method: 'POST', body,
    }, options, UPLOAD_TIMEOUT_MS);
    return parseStagedUpload(result);
  }

  async syncUploads(request: BrowserUploadSyncRequest, options: RequestOptions = {}): Promise<UploadManifest> {
    try {
      const result = await requestJson(`${this.baseUrl}/sessions/${encodeURIComponent(this.sessionId)}/uploads/sync`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request),
      }, options, UPLOAD_TIMEOUT_MS);
      this.manifest = parseUploadManifest(result);
      return this.manifest;
    } catch (error) {
      this.manifest = null;
      throw error;
    }
  }

  async saveAnswer(answerId: string, options: RequestOptions = {}): Promise<AnswerExportResult> {
    const result = await requestJson(`${this.baseUrl}/sessions/${encodeURIComponent(this.sessionId)}/answers/${encodeURIComponent(answerId)}/save`, {
      method: 'POST',
    }, options, this.timeoutMs);
    return parseAnswerExport(result, answerId);
  }

  downloadUrl(filename: string): string {
    return `${this.baseUrl}/download/${encodeURIComponent(filename)}`;
  }

  async *stream(query: string, options: StreamOptions = {}): AsyncGenerator<AgentStreamEvent> {
    if (!query.trim()) throw new ApiError('invalid_query', '질문을 입력해 주세요.');
    const controller = new AbortController();
    const signal = options.signal ? AbortSignal.any([options.signal, controller.signal]) : controller.signal;
    let receivedFinal = false;
    let requestId: string | undefined;
    let timedOut = false;
    let idleTimer: ReturnType<typeof setTimeout> | undefined;
    const resetIdleTimeout = () => {
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, this.timeoutMs);
    };
    try {
      const manifest = this.manifest ?? await this.refreshUploads({ signal });
      const payload: AgentRequest = {
        query,
        session_id: this.sessionId,
        uploads: { epoch: manifest.epoch, revision: manifest.revision },
      };
      if (options.slackRecipient != null) payload.slack_recipient = options.slackRecipient;
      // A lost response may hide completed actions. Never retry or forward this POST.
      resetIdleTimeout();
      const response = await fetch(`${this.baseUrl}/agent/stream`, {
        method: 'POST',
        headers: { Accept: 'text/event-stream', 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        redirect: 'manual',
        signal,
      });
      resetIdleTimeout();
      requestId = response.headers.get('X-Request-ID') ?? undefined;
      if (response.status !== 200) {
        throw new ApiError('http_error', '서버의 처리 결과를 확인하지 못했습니다. 질문을 자동으로 다시 보내지 않습니다.', {
          status: response.status, requestId,
        });
      }
      if (response.headers.get('Content-Type')?.split(';')[0].trim().toLowerCase() !== 'text/event-stream' || !response.body) {
        throw new ApiError('invalid_stream', '서버의 스트리밍 응답 형식이 올바르지 않습니다.', { requestId });
      }
      const body = response.body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
        transform(chunk, streamController) {
          resetIdleTimeout();
          streamController.enqueue(chunk);
        },
      }));
      for await (const raw of parseSseStream(body)) {
        const event = parseAgentEvent(raw.event, raw.data);
        if (event.event === 'request_started') requestId = requestId ?? event.data.request_id;
        if (event.event === 'final_response') {
          receivedFinal = true;
          this.manifest = event.data.upload_manifest;
          yield event;
          return;
        }
        yield event;
        if (event.event === 'done') break;
      }
      throw new ApiError('missing_final_response', '최종 응답을 받기 전에 연결이 끝났습니다. 서버에서 요청이 처리되었을 수 있습니다.', { requestId });
    } catch (error) {
      if (timedOut) {
        throw new ApiError('timeout', '응답 대기 시간이 초과되었습니다. 서버에서 요청이 처리되었을 수 있습니다.', { requestId, cause: error });
      }
      if (signal.aborted) {
        throw new ApiError('aborted', '응답 수신을 중단했습니다. 서버 작업은 계속될 수 있습니다.', { requestId, cause: error });
      }
      if (error instanceof ApiError) {
        if (error.code === 'invalid_response') {
          throw new ApiError('invalid_stream', '스트리밍 응답을 검증하지 못했습니다. 서버의 처리 결과를 확인해 주세요.', { requestId, cause: error });
        }
        throw error;
      }
      throw new ApiError('stream_error', '응답을 끝까지 받지 못했습니다. 서버에서 요청이 처리되었을 수 있습니다.', { requestId, cause: error });
    } finally {
      clearTimeout(idleTimer);
      controller.abort();
      if (!receivedFinal) this.manifest = null;
    }
  }
}

async function requestJson(url: string, init: RequestInit, options: RequestOptions, timeoutMs = DEFAULT_LOOKUP_TIMEOUT_MS): Promise<unknown> {
  let requestId: string | undefined;
  const controller = new AbortController();
  const signal = options.signal ? AbortSignal.any([options.signal, controller.signal]) : controller.signal;
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  try {
    const headers = new Headers(init.headers);
    headers.set('Accept', 'application/json');
    const response = await fetch(url, { ...init, headers, redirect: 'manual', signal });
    requestId = response.headers.get('X-Request-ID') ?? undefined;
    let value: unknown;
    try {
      value = await response.json();
    } catch (error) {
      if (!(error instanceof SyntaxError)) throw error;
      if (response.status === 200) throw new ApiError('invalid_response', '서버 응답의 형식을 확인하지 못했습니다.', { requestId, cause: error });
    }
    if (response.status !== 200) throw httpError(response.status, requestId, value);
    return value;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (timedOut) throw new ApiError('timeout', '서버 상태 확인 시간이 초과되었습니다.', { requestId, cause: error });
    if (options.signal?.aborted) throw new ApiError('aborted', '상태 확인을 중단했습니다.', { requestId, cause: error });
    throw new ApiError('connection_error', '서버 상태를 확인하지 못했습니다. 다시 확인해 주세요.', { requestId, cause: error });
  } finally {
    clearTimeout(timer);
  }
}

function httpError(status: number, requestId: string | undefined, value: unknown): ApiError {
  const fallback = '서버의 처리 결과를 확인하지 못했습니다. 자동으로 다시 요청하지 않습니다.';
  if (value === null || typeof value !== 'object' || !('detail' in value)) {
    return new ApiError('http_error', fallback, { status, requestId });
  }
  const detail = value.detail;
  if (detail === null || typeof detail !== 'object' || Array.isArray(detail)) {
    return new ApiError('http_error', typeof detail === 'string' ? detail : fallback, { status, requestId });
  }
  const files: UploadFailure[] = [];
  if ('files' in detail && Array.isArray(detail.files)) {
    const reportedFiles: unknown[] = detail.files;
    for (const item of reportedFiles) {
      if (item === null || typeof item !== 'object' || Array.isArray(item)) continue;
      const fields = item as Record<string, unknown>;
      const file: UploadFailure = {};
      for (const key of ['name', 'code', 'message', 'file_id'] as const) {
        if (typeof fields[key] === 'string') file[key] = fields[key];
      }
      files.push(file);
    }
  }
  return new ApiError('code' in detail && typeof detail.code === 'string' ? detail.code : 'http_error',
    'message' in detail && typeof detail.message === 'string' ? detail.message : fallback,
    { status, requestId, files });
}
