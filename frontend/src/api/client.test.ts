// @vitest-environment node
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import fixtures from '../../../tests/fixtures/web_contracts/agent_responses.json';
import sourceFixtures from '../components/AnswerView.fixtures.json';
import { AgentSessionClient, ApiError, fetchClientConfig } from './client';

const manifest = { epoch: 'confirmed', revision: 2, files: [] };
const config = {
  upload: {
    accepted_suffixes: ['.ipynb', '.py'],
    max_files: 10,
    max_file_bytes: 10485760,
    max_total_bytes: 52428800,
  },
  query: { max_chars: 8192, max_utf8_bytes: 16384 },
};

function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

async function collect<T>(events: AsyncIterable<T>): Promise<T[]> {
  const result: T[] = [];
  for await (const event of events) result.push(event);
  return result;
}

describe('browser API contract', () => {
  let baseUrl: string;
  let requests: { method: string; path: string; body: string }[];
  let respond: (request: IncomingMessage, response: ServerResponse) => void;
  const server = createServer(async (request, response) => {
    const chunks: Buffer[] = [];
    for await (const chunk of request) chunks.push(Buffer.from(chunk));
    requests.push({ method: request.method ?? '', path: request.url ?? '', body: Buffer.concat(chunks).toString() });
    if (request.url?.endsWith('/uploads')) {
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify(manifest));
      return;
    }
    respond(request, response);
  });

  beforeEach(async () => {
    requests = [];
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'text/event-stream; charset=utf-8' });
      response.end(frame('final_response', fixtures.responses.completed));
    };
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const address = server.address();
    if (!address || typeof address === 'string') throw new Error('TCP address expected');
    baseUrl = `http://127.0.0.1:${address.port}`;
  });

  afterEach(async () => {
    vi.useRealTimers();
    server.closeAllConnections();
    await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  });

  it('loads the public configuration without accepting extra settings or coercing limits', async () => {
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify(config));
    };
    expect(await fetchClientConfig(baseUrl)).toEqual(config);
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify({ ...config, upload: { ...config.upload, max_files: '10' } }));
    };
    await expect(fetchClientConfig(baseUrl)).rejects.toMatchObject({ code: 'invalid_response' });
  });

  it.each(Object.entries(fixtures.responses))('accepts the shared %s result and its manifest meaning', async (_name, result) => {
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'text/event-stream' });
      response.end(frame('final_response', result));
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });
    const events = await collect(client.stream('  질문 🧪\n'));
    expect(events).toEqual([{ event: 'final_response', data: result }]);
    expect(client.manifest).toEqual(result.upload_manifest);
    expect(requests.map(({ method }) => method)).toEqual(['GET', 'POST']);
    expect(JSON.parse(requests[1].body)).toEqual({
      query: '  질문 🧪\n', session_id: 'session-1', uploads: { epoch: 'confirmed', revision: 2 },
    });
  });

  it('resolves citations through the source registry and rejects a missing captured source', async () => {
    const result = { ...fixtures.responses.completed, response: structuredClone(sourceFixtures.sectioned) };
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'text/event-stream' });
      response.end(frame('final_response', result));
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });
    expect(await collect(client.stream('원문을 보여 줘'))).toEqual([{ event: 'final_response', data: result }]);

    const sources: Record<string, unknown> = result.response.sources;
    delete sources[result.response.citations[0].source_id];
    await expect(collect(client.stream('같은 자료를 보여 줘'))).rejects.toMatchObject({ code: 'invalid_stream' });
    expect(client.manifest).toBeNull();
  });

  it.each(['request_timeout', 'answer_validation_failed'])('preserves the server %s problem and request ID without resubmitting', async (code) => {
    const result = {
      ...fixtures.responses.failed,
      message: `서버가 보낸 실행 결과: ${code}`,
      problem: { ...fixtures.responses.failed.problem, code, message: `서버가 확인한 실패 사유: ${code}` },
      execution_id: `server-request-${code}`,
    };
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'text/event-stream' });
      response.end(frame('final_response', result));
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });

    expect(await collect(client.stream('서버 결과를 확인해 줘'))).toEqual([
      { event: 'final_response', data: result },
    ]);
    expect(requests.filter(({ method }) => method === 'POST')).toHaveLength(1);
    expect(client.manifest).toBeNull();
  });

  it.each(
    fixtures.missing_response_fields.map((path) => ({ name: path.join('.'), path })),
  )('rejects a missing serialized $name field instead of filling a default', async ({ path }) => {
    const result = structuredClone(fixtures.responses.completed);
    let container: Record<string, unknown> = result;
    for (const key of path.slice(0, -1)) container = container[key] as Record<string, unknown>;
    delete container[path[path.length - 1]];
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'text/event-stream' });
      response.end(frame('final_response', result));
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });

    await expect(collect(client.stream('질문'))).rejects.toMatchObject({ code: 'invalid_stream' });
    expect(client.manifest).toBeNull();
    expect(requests.filter(({ method }) => method === 'POST')).toHaveLength(1);
  });

  it.each(fixtures.invalid_events)('rejects the shared invalid $event event before accepting a later answer', async ({ event, data }) => {
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'text/event-stream', 'X-Execution-ID': 'invalid-event' });
      response.end(frame(event, data) + frame('final_response', fixtures.responses.completed));
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });

    await expect(collect(client.stream('질문'))).rejects.toMatchObject({ code: 'invalid_stream', executionId: 'invalid-event' });
    expect(client.manifest).toBeNull();
    expect(requests.filter(({ method }) => method === 'POST')).toHaveLength(1);
  });

  it('preserves sentence scope metadata with the exact answer text', async () => {
    const original = fixtures.responses.completed;
    const result = { ...original, response: { ...original.response, content: { blocks: [
      { type: 'paragraph', content: [
        { ...original.response.content.blocks[0].content[0], sentence_scope: 1 },
      ] },
    ] } } };
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'text/event-stream' });
      response.end(frame('final_response', result));
    };

    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });

    expect(await collect(client.stream('한 문장으로 설명해 줘.'))).toEqual([
      { event: 'final_response', data: result },
    ]);
  });

  it.each([{ payload: [] }, { payload: null }, { payload: 'text' }])('rejects a non-object event payload at the API boundary: $payload', async ({ payload }) => {
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'text/event-stream', 'X-Execution-ID': 'invalid-payload' });
      response.end(frame('final_response', payload));
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });

    await expect(collect(client.stream('질문'))).rejects.toMatchObject({ code: 'invalid_stream', executionId: 'invalid-payload' });
    expect(client.manifest).toBeNull();
    expect(requests.filter(({ method }) => method === 'POST')).toHaveLength(1);
  });

  it('keeps a server error before the valid final response', async () => {
    const error = { message: '단계 진단', stage: 'planner' };
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'text/event-stream' });
      response.end(frame('error', error) + frame('final_response', fixtures.responses.completed));
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });
    const events = await collect(client.stream('질문'));
    expect(events).toEqual([
      { event: 'error', data: error },
      { event: 'final_response', data: fixtures.responses.completed },
    ]);
    expect(client.manifest).toEqual(fixtures.responses.completed.upload_manifest);
  });

  it('commits a valid final response without waiting for done or reading trailing frames', async () => {
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'text/event-stream' });
      response.write(frame('final_response', fixtures.responses.completed) + 'event: final_response\ndata: {broken}\n\n');
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });
    expect(await collect(client.stream('질문'))).toEqual([
      { event: 'final_response', data: fixtures.responses.completed },
    ]);
    expect(client.manifest).toEqual(fixtures.responses.completed.upload_manifest);
  });

  it.each([
    ['empty EOF', ''],
    ['done without final', frame('done', {})],
    ['error without final', frame('error', { message: '실패' })],
    ['unfinished frame', frame('final_response', fixtures.responses.completed).slice(0, -1)],
    ['invalid JSON', 'event: final_response\ndata: {broken}\n\n'],
  ])('invalidates confirmation after %s and never repeats the POST', async (_name, body) => {
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'text/event-stream' });
      response.end(body);
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });
    await expect(collect(client.stream('저장해줘'))).rejects.toBeInstanceOf(ApiError);
    expect(client.manifest).toBeNull();
    expect(requests.filter(({ method }) => method === 'POST')).toHaveLength(1);
  });

  it.each([
    ['completed without manifest', { ...fixtures.responses.completed, upload_manifest: null }],
    ['completed without answer', { ...fixtures.responses.completed, response: null }],
    ['partial without problem', { ...fixtures.responses.partial, problem: null }],
    ['failed containing answer', { ...fixtures.responses.failed, response: fixtures.responses.completed.response }],
    ['clarification without a question', { ...fixtures.responses.needs_input, message: '  ' }],
    ['refusal with another problem', { ...fixtures.responses.refused, problem: fixtures.responses.failed.problem }],
    ['malformed manifest', { ...fixtures.responses.completed, upload_manifest: { epoch: 'x', revision: '2', files: [] } }],
  ])('rejects %s at the response boundary', async (_name, result) => {
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'text/event-stream', 'X-Execution-ID': 'request-invalid' });
      response.end(frame('final_response', result));
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });
    await expect(collect(client.stream('질문'))).rejects.toMatchObject({ code: 'invalid_stream', executionId: 'request-invalid' });
    expect(client.manifest).toBeNull();
  });

  it.each([307, 503])('does not redirect or retry an HTTP %s response', async (status) => {
    respond = (_request, response) => {
      response.writeHead(status, { Location: '/forwarded', 'X-Execution-ID': 'http-request' });
      response.end();
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });
    await expect(collect(client.stream('저장해줘'))).rejects.toMatchObject({ code: 'http_error', status, executionId: 'http-request' });
    expect(client.manifest).toBeNull();
    expect(requests.map(({ path }) => path)).toEqual(['/sessions/session-1/uploads', '/agent/stream']);
  });

  it('rejects a JSON body presented as a successful stream', async () => {
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify(fixtures.responses.completed));
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });
    await expect(collect(client.stream('질문'))).rejects.toMatchObject({ code: 'invalid_stream' });
    expect(client.manifest).toBeNull();
  });

  it('stops receiving on abort and leaves server confirmation unknown', async () => {
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'text/event-stream' });
      response.write(frame('request_started', { execution_id: 'request-pending', session_id: 'session-1' }));
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });
    const controller = new AbortController();
    const events = client.stream('저장해줘', { signal: controller.signal });
    expect((await events.next()).value?.event).toBe('request_started');
    controller.abort();
    await expect(events.next()).rejects.toMatchObject({ code: 'aborted', executionId: 'request-pending' });
    expect(client.manifest).toBeNull();
    expect(requests.filter(({ method }) => method === 'POST')).toHaveLength(1);
  });

  it('requires fresh confirmation before another explicit submission after a lost response', async () => {
    let attempts = 0;
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'text/event-stream' });
      response.end(attempts++ === 0 ? frame('done', {}) : frame('final_response', fixtures.responses.completed));
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });
    await expect(collect(client.stream('첫 질문'))).rejects.toBeInstanceOf(ApiError);
    await collect(client.stream('사용자가 명시한 새 질문'));
    expect(requests.map(({ method }) => method)).toEqual(['GET', 'POST', 'GET', 'POST']);
  });

  it.each(['headers', 'body'])('times out while %s are idle without retrying the question', async (stage) => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    let received: () => void = () => {};
    const accepted = new Promise<void>((resolve) => { received = resolve; });
    respond = (_request, response) => {
      if (stage === 'body') {
        response.writeHead(200, { 'Content-Type': 'text/event-stream' });
        response.write(frame('request_started', { execution_id: 'idle-request', session_id: 'session-1' }));
      }
      received();
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl, timeoutMs: 50 });
    const events = client.stream('저장해줘');
    if (stage === 'body') expect((await events.next()).value?.event).toBe('request_started');
    const rejection = expect(collect(events)).rejects.toMatchObject({ code: 'timeout' });
    await accepted;
    await vi.advanceTimersByTimeAsync(51);
    await rejection;
    expect(client.manifest).toBeNull();
    expect(requests.filter(({ method }) => method === 'POST')).toHaveLength(1);
  });

  it('keeps receiving a long request while heartbeat bytes arrive within the idle timeout', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    let pendingResponse: ServerResponse | undefined;
    respond = (_request, response) => {
      pendingResponse = response;
      response.writeHead(200, { 'Content-Type': 'text/event-stream' });
      response.write(frame('request_started', { execution_id: 'active-request', session_id: 'session-1' }));
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl, timeoutMs: 100 });
    const events = client.stream('오래 걸리는 질문');
    expect((await events.next()).value?.event).toBe('request_started');
    for (let index = 1; index <= 3; index += 1) {
      await vi.advanceTimersByTimeAsync(60);
      const next = events.next();
      pendingResponse?.write(frame('heartbeat', { execution_id: 'active-request', elapsed_ms: index * 60 }));
      expect((await next).value?.event).toBe('heartbeat');
    }
    pendingResponse?.end(frame('final_response', fixtures.responses.completed));
    expect((await events.next()).value?.event).toBe('final_response');
    await events.return(undefined);
    expect(client.manifest).toEqual(fixtures.responses.completed.upload_manifest);
  });
});
