// @vitest-environment node
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import fixtures from '../../../tests/fixtures/web_contracts/agent_responses.json';
import { AgentSessionClient, fetchSlackDefault } from './client';

const manifest = { epoch: 'epoch-1', revision: 2, files: [] };
const staged = { upload_id: 'a'.repeat(32), name: '예제.py', size_bytes: 9, content_hash: `sha256:${'b'.repeat(64)}` };
const syncRequest = {
  epoch: 'epoch-1', expected_revision: 2, operation_id: 'operation-1',
  add: [{ upload_id: staged.upload_id, name: staged.name, content_hash: staged.content_hash }],
};

describe('explicit attachment and delivery operations', () => {
  let baseUrl: string;
  let requests: { method: string; url: string; body: string; contentType: string | undefined }[];
  let respond: (request: IncomingMessage, response: ServerResponse) => void;
  const server = createServer(async (request, response) => {
    const chunks: Buffer[] = [];
    for await (const chunk of request) chunks.push(Buffer.from(chunk));
    requests.push({ method: request.method ?? '', url: request.url ?? '', body: Buffer.concat(chunks).toString(), contentType: request.headers['content-type'] });
    respond(request, response);
  });

  beforeEach(async () => {
    requests = [];
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify(manifest));
    };
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const address = server.address();
    if (!address || typeof address === 'string') throw new Error('TCP address expected');
    baseUrl = `http://127.0.0.1:${address.port}`;
  });

  afterEach(async () => {
    server.closeAllConnections();
    await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  });

  it('stages file bytes without committing or changing the confirmed attachment list', async () => {
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });
    await client.refreshUploads();
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify(staged));
    };
    expect(await client.stageFile(new File(['print(1)\n'], '예제.py'))).toEqual(staged);
    expect(client.manifest).toEqual(manifest);
    expect(requests[1].url).toBe('/sessions/session-1/uploads/staging');
    expect(requests[1].contentType).toMatch(/^multipart\/form-data; boundary=/);
    expect(requests[1].body).toContain('name="file"; filename="예제.py"');
    expect(requests[1].body).toContain('print(1)\n');
    expect(requests).toHaveLength(2);
  });

  it('preserves the approved operation payload for an explicit retry after a conflict', async () => {
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });
    await client.refreshUploads();
    respond = (_request, response) => {
      response.writeHead(409, { 'Content-Type': 'application/json', 'X-Request-ID': 'conflict-1' });
      response.end(JSON.stringify({ detail: { code: 'UPLOAD_REVISION_CONFLICT', message: '첨부 목록이 변경되었습니다.', files: [] } }));
    };
    await expect(client.syncUploads(syncRequest)).rejects.toMatchObject({ code: 'UPLOAD_REVISION_CONFLICT', status: 409, requestId: 'conflict-1' });
    expect(client.manifest).toBeNull();
    expect(requests.filter(({ method }) => method === 'POST')).toHaveLength(1);
    const updated = { ...manifest, revision: 3 };
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify(updated));
    };
    expect(await client.syncUploads(syncRequest)).toEqual(updated);
    expect(client.manifest).toEqual(updated);
    expect(requests.filter(({ method }) => method === 'POST').map(({ body }) => JSON.parse(body))).toEqual([syncRequest, syncRequest]);
  });

  it('keeps filename errors from a rejected stage and preserves the existing manifest', async () => {
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });
    await client.refreshUploads();
    const files = [{ name: '예제.py', code: 'UPLOAD_FILE_TOO_LARGE', message: '파일이 너무 큽니다.' }];
    respond = (_request, response) => {
      response.writeHead(413, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify({ detail: { code: 'UPLOAD_FILE_TOO_LARGE', message: '파일을 확인해 주세요.', files } }));
    };
    await expect(client.stageFile(new File(['content'], '예제.py'))).rejects.toMatchObject({ status: 413, code: 'UPLOAD_FILE_TOO_LARGE', files });
    expect(client.manifest).toEqual(manifest);
  });

  it.each(['stage', 'sync'])('rejects malformed %s results without inventing confirmed attachments', async (operation) => {
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });
    await client.refreshUploads();
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end('{}');
    };
    const result = operation === 'stage'
      ? client.stageFile(new File(['print(1)\n'], '예제.py'))
      : client.syncUploads(syncRequest);
    await expect(result).rejects.toMatchObject({ code: 'invalid_response' });
    expect(client.manifest).toEqual(operation === 'stage' ? manifest : null);
    expect(requests.filter(({ method }) => method === 'POST')).toHaveLength(1);
  });

  it.each([307, 503])('never forwards or retries a mutation returning HTTP %s', async (status) => {
    respond = (_request, response) => {
      response.writeHead(status, { Location: '/forwarded' });
      response.end();
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });
    await expect(client.syncUploads(syncRequest)).rejects.toMatchObject({ status });
    expect(client.manifest).toBeNull();
    expect(requests).toHaveLength(1);
    expect(requests[0].url).toBe('/sessions/session-1/uploads/sync');
  });

  it('retries only the selected answer ID and returns a failed save receipt without replaying chat', async () => {
    const result = {
      status: 'failed', answer_id: 'old-answer', request_id: 'export-1', message: '저장에 실패했습니다.',
      response: { ...fixtures.responses.completed.response, actions: [{ kind: 'save_text', status: 'error', error: 'write_failed',
        invocation_id: null, file_path: null, slack: null, message: null, operation: null, artifact: null,
        verification: 'failed', error_code: 'write_failed' }] },
    };
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify(result));
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });
    expect(await client.saveAnswer('old-answer')).toEqual(result);
    expect(requests).toHaveLength(1);
    expect(await client.saveAnswer('old-answer')).toEqual(result);
    expect(requests.map(({ url, body }) => ({ url, body }))).toEqual([
      { url: '/sessions/session-1/answers/old-answer/save', body: '' },
      { url: '/sessions/session-1/answers/old-answer/save', body: '' },
    ]);
  });

  it('keeps omitted Slack defaults distinct from invalid or unavailable defaults', async () => {
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify({ selector: null, failure: null }));
    };
    expect(await fetchSlackDefault(baseUrl)).toEqual({ selector: null, failure: null });
    respond = (_request, response) => {
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify({ selector: { kind: 'user', value: 'invalid' }, failure: null }));
    };
    await expect(fetchSlackDefault(baseUrl)).rejects.toMatchObject({ code: 'invalid_response' });
  });

  it('preserves incomplete explicit recipient input on an ordinary question', async () => {
    respond = (request, response) => {
      if (request.method === 'GET') {
        response.writeHead(200, { 'Content-Type': 'application/json' });
        response.end(JSON.stringify(manifest));
        return;
      }
      response.writeHead(200, { 'Content-Type': 'text/event-stream' });
      response.end(`event: final_response\ndata: ${JSON.stringify(fixtures.responses.completed)}\n\n`);
    };
    const client = new AgentSessionClient({ sessionId: 'session-1', baseUrl });
    for await (const event of client.stream('일반 질문', { slackRecipient: { kind: 'channel', value: '' } })) {
      expect(event.event).toBe('final_response');
    }
    expect(JSON.parse(requests[1].body).slack_recipient).toEqual({ kind: 'channel', value: '' });
  });
});
