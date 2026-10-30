import { ApiError } from './errors';
import { parseSseStream } from './sse';
import type { AgentRequest, AgentStreamEvent, ClientConfig, UploadManifest } from './types';
import { parseAgentEvent, parseClientConfig, parseUploadManifest } from './validation';

export { ApiError } from './errors';

type RequestOptions = { signal?: AbortSignal };

export async function fetchClientConfig(baseUrl = '', options: RequestOptions = {}): Promise<ClientConfig> {
  return parseClientConfig(await getJson(`${baseUrl.replace(/\/$/, '')}/client-config`, options));
}

export class AgentSessionClient {
  manifest: UploadManifest | null = null;
  readonly sessionId: string;
  private readonly baseUrl: string;
  private readonly timeoutMs: number;

  constructor({ sessionId, baseUrl = '', timeoutMs = 60_000 }: { sessionId: string; baseUrl?: string; timeoutMs?: number }) {
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
    const value = await getJson(`${this.baseUrl}/sessions/${encodeURIComponent(this.sessionId)}/uploads`, options, this.timeoutMs);
    this.manifest = parseUploadManifest(value);
    return this.manifest;
  }

  async *stream(query: string, options: RequestOptions = {}): AsyncGenerator<AgentStreamEvent> {
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

async function getJson(url: string, options: RequestOptions, timeoutMs = 15_000): Promise<unknown> {
  let requestId: string | undefined;
  const controller = new AbortController();
  const signal = options.signal ? AbortSignal.any([options.signal, controller.signal]) : controller.signal;
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  try {
    const response = await fetch(url, { headers: { Accept: 'application/json' }, redirect: 'manual', signal });
    requestId = response.headers.get('X-Request-ID') ?? undefined;
    if (response.status !== 200) {
      await response.body?.cancel();
      throw new ApiError('http_error', '서버 상태를 확인하지 못했습니다. 다시 확인해 주세요.', { status: response.status, requestId });
    }
    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (timedOut) throw new ApiError('timeout', '서버 상태 확인 시간이 초과되었습니다.', { requestId, cause: error });
    if (options.signal?.aborted) throw new ApiError('aborted', '상태 확인을 중단했습니다.', { requestId, cause: error });
    throw new ApiError('connection_error', '서버 상태를 확인하지 못했습니다. 다시 확인해 주세요.', { requestId, cause: error });
  } finally {
    clearTimeout(timer);
  }
}
