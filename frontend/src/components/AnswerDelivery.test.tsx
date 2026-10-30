import { createServer, type ServerResponse } from 'node:http';
import { once } from 'node:events';
import { createHash } from 'node:crypto';
import { useState } from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import fixtures from '../../../tests/fixtures/web_contracts/agent_responses.json';
import { AgentSessionClient } from '../api/client';
import { parseAgentEvent } from '../api/validation';
import type { AnswerExportResult } from '../api/types';
import { AnswerDelivery } from './AnswerDelivery';

const event = parseAgentEvent('final_response', fixtures.responses.partial);
if (event.event !== 'final_response' || !event.data.response) throw new Error('Expected answer fixture');
const answer = event.data.response;
const downloadBytes = Buffer.from('\uFEFF원문\n', 'utf8');
const artifact = { artifact_id: 'a'.repeat(64), filename: '답변 결과.txt', sha256: createHash('sha256').update(downloadBytes).digest('hex'),
  byte_count: downloadBytes.length, created_at: 1_790_000_000, expires_at: 4_000_000_000 };
const operation = { operation_id: 'save-selected-answer', session_id: 'delivery-session', request_id: 'request-selected-answer',
  contract_revision: 1, target_kind: 'copy_answer' as const, source_hash: answer.content_hash,
  answer_hash: answer.content_hash, export_profile: 'answer-text-v1/utf-8-sig' as const,
  payload_sha256: artifact.sha256, byte_count: artifact.byte_count };
const completed: AnswerExportResult = { status: 'completed', answer_id: 'selected-answer', request_id: 'export-1',
  message: 'TXT 파일을 저장했습니다.', response: { ...answer, actions: [{ kind: 'save_text', status: 'success',
    verification: 'verified', artifact, operation, file_path: 'output/save_text/답변 결과.txt',
    invocation_id: null, slack: null, message: 'TXT 파일을 저장했습니다.', error: null, error_code: null }] } };
const failed: AnswerExportResult = { ...completed, status: 'failed', message: '파일을 만들지 못했습니다. 기존 답변은 유지됩니다.',
  response: { ...answer, actions: [{ kind: 'save_text', status: 'error', verification: 'failed', artifact: null,
    operation, error_code: 'write_failed', error: '파일을 만들지 못했습니다.', file_path: null,
    invocation_id: null, slack: null, message: '파일을 만들지 못했습니다.' }] } };

let apiBaseUrl = '';
let mode: 'completed' | 'failed' | 'unavailable' | 'hold' = 'completed';
let requests: { path: string; body: string }[] = [];
let held: ServerResponse | null = null;
const server = createServer(async (request, response) => {
  if (request.url?.startsWith('/download/')) {
    response.writeHead(200, { 'Content-Type': 'text/plain' });
    response.end(downloadBytes);
    return;
  }
  let body = '';
  for await (const chunk of request) body += String(chunk);
  requests.push({ path: request.url ?? '', body });
  response.writeHead(mode === 'unavailable' ? 410 : 200, { 'Content-Type': 'application/json' });
  if (mode === 'hold') { held = response; return; }
  response.end(JSON.stringify(mode === 'unavailable'
    ? { detail: { code: 'session_expired', message: '이 대화의 서버 기록이 만료되었습니다.' } }
    : mode === 'failed' ? failed : completed));
});

beforeAll(async () => {
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Test server did not bind a TCP port');
  apiBaseUrl = `http://127.0.0.1:${address.port}`;
});
beforeEach(() => { requests = []; mode = 'completed'; held = null; });
afterAll(async () => {
  held?.end();
  server.closeAllConnections();
  await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
});

function Delivery({ initial = null, allowSave = true }: { initial?: AnswerExportResult | null; allowSave?: boolean }) {
  const [result, setResult] = useState(initial);
  const [busy, setBusy] = useState(false);
  const [client] = useState(() => new AgentSessionClient({ sessionId: 'delivery-session', baseUrl: apiBaseUrl }));
  return <>
    <p>{answer.issues[0].message}</p>
    <button disabled={busy}>다른 대화</button>
    <AnswerDelivery client={client} answerId="selected-answer" response={answer} currentExport={result}
      onExportResult={setResult} onBusyChange={setBusy} onBeforeSave={async () => allowSave} />
  </>;
}

describe('선택 답변 TXT 저장', () => {
  it('saves only the server answer id, keeps its limitations and links the exact server bytes', async () => {
    const user = userEvent.setup();
    render(<Delivery />);
    expect(requests).toHaveLength(0);
    await user.click(screen.getByRole('button', { name: 'TXT 파일 만들기' }));
    const download = await screen.findByRole('link', { name: 'TXT 다운로드' });
    expect(requests).toEqual([{ path: '/sessions/delivery-session/answers/selected-answer/save', body: '' }]);
    expect(download).toHaveAttribute('href', `${apiBaseUrl}/download/${encodeURIComponent(artifact.filename)}`);
    const downloaded = await fetch(download.getAttribute('href') ?? '');
    expect(Buffer.from(await downloaded.arrayBuffer())).toEqual(downloadBytes);
    expect(screen.getByText(answer.issues[0].message)).toBeVisible();
  });

  it('retries the same answer after a failed receipt without resubmitting a question', async () => {
    mode = 'failed';
    const user = userEvent.setup();
    render(<Delivery />);
    await user.click(screen.getByRole('button', { name: 'TXT 파일 만들기' }));
    await screen.findByRole('button', { name: '저장 다시 시도' });
    expect(requests).toHaveLength(1);
    mode = 'completed';
    await user.click(screen.getByRole('button', { name: '저장 다시 시도' }));
    await screen.findByRole('link', { name: 'TXT 다운로드' });
    expect(requests.map(({ path }) => path)).toEqual(Array(2).fill('/sessions/delivery-session/answers/selected-answer/save'));
  });

  it('blocks duplicate saves and navigation during save, then restores controls', async () => {
    mode = 'hold';
    const user = userEvent.setup();
    render(<Delivery />);
    await user.dblClick(screen.getByRole('button', { name: 'TXT 파일 만들기' }));
    await waitFor(() => expect(held).not.toBeNull());
    expect(screen.getByRole('button', { name: '다른 대화' })).toBeDisabled();
    expect(requests).toHaveLength(1);
    held?.end(JSON.stringify(completed));
    await screen.findByRole('link', { name: 'TXT 다운로드' });
    expect(screen.getByRole('button', { name: '다른 대화' })).toBeEnabled();
  });

  it('offers regeneration for an expired artifact and never substitutes cached answer text for an expired server session', async () => {
    mode = 'unavailable';
    const user = userEvent.setup();
    const expired: AnswerExportResult = { ...completed, response: { ...answer, actions: [{ ...completed.response.actions[0],
      artifact: { ...artifact, created_at: 1, expires_at: 2 } }] } };
    render(<Delivery initial={expired} />);
    expect(screen.queryByRole('link', { name: 'TXT 다운로드' })).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'TXT 다시 만들기' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('서버 기록이 만료');
    expect(screen.getByText(answer.issues[0].message)).toBeVisible();
    expect(requests).toHaveLength(1);
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });

  it('does not save when the parent cannot confirm the current server session', async () => {
    const user = userEvent.setup();
    render(<Delivery allowSave={false} />);
    await user.click(screen.getByRole('button', { name: 'TXT 파일 만들기' }));
    expect(requests).toHaveLength(0);
    expect(screen.getByRole('button', { name: '다른 대화' })).toBeEnabled();
  });
});
