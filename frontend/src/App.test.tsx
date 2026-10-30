import { createServer, type ServerResponse } from 'node:http';
import { once } from 'node:events';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import fixtures from '../../tests/fixtures/web_contracts/agent_responses.json';
import { App } from './App';

const config = {
  upload: {
    accepted_suffixes: ['.ipynb', '.py'],
    max_files: 10,
    max_file_bytes: 10485760,
    max_total_bytes: 52428800,
  },
  query: { max_chars: 8192, max_utf8_bytes: 16384 },
};

const manifest = { epoch: 'contract-epoch', revision: 2, files: [] };
let responseName: keyof typeof fixtures.responses = 'completed';
let holdResponse = false;
let requests: string[] = [];
let pendingResponse: ServerResponse | null = null;
let apiBaseUrl = '';

const server = createServer(async (request, response) => {
  if (request.url === '/client-config') {
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify(config));
    return;
  }
  if (request.method === 'GET' && request.url?.endsWith('/uploads')) {
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify(manifest));
    return;
  }
  if (request.url === '/agent/stream') {
    let body = '';
    for await (const chunk of request) body += String(chunk);
    requests.push(body);
    response.writeHead(200, { 'Content-Type': 'text/event-stream' });
    response.write('event: progress_snapshot\ndata: {"stage":"planner","summary":"질문을 확인하고 있습니다."}\n\n');
    if (holdResponse) {
      pendingResponse = response;
      return;
    }
    response.end(`event: final_response\ndata: ${JSON.stringify(fixtures.responses[responseName])}\n\nevent: done\ndata: {}\n\n`);
    return;
  }
  response.writeHead(404);
  response.end();
});

beforeAll(async () => {
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Test server did not bind a TCP port');
  apiBaseUrl = `http://127.0.0.1:${address.port}`;
});

beforeEach(() => {
  pendingResponse?.end();
  pendingResponse = null;
  requests = [];
  holdResponse = false;
  responseName = 'completed';
  config.query.max_chars = 8192;
  config.query.max_utf8_bytes = 16384;
});

afterAll(async () => {
  pendingResponse?.end();
  server.closeAllConnections();
  await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
});

describe('질문 화면과 실제 HTTP 클라이언트', () => {
  it('명시적 제출 때만 질문 원문을 한 번 보내고 최종 답변을 표시한다', async () => {
    const user = userEvent.setup();
    render(<App apiBaseUrl={apiBaseUrl} />);
    await waitFor(() => expect(screen.getByRole('button', { name: '질문 보내기' })).toBeDisabled());
    await user.type(screen.getByRole('textbox', { name: '질문' }), '  한글 질문  ');
    await waitFor(() => expect(screen.getByRole('button', { name: '질문 보내기' })).toBeEnabled());
    expect(requests).toHaveLength(0);
    await user.click(screen.getByRole('button', { name: '질문 보내기' }));
    expect(await screen.findByText('한글 답변 🧪')).toBeVisible();
    expect(screen.getByText('답변 받음')).toBeVisible();
    expect(requests).toHaveLength(1);
    expect(JSON.parse(requests[0]).query).toBe('  한글 질문  ');
    expect(screen.getByRole('textbox', { name: '질문' })).toHaveValue('');
  });

  it('부분 답변의 본문과 한계를 함께 표시한다', async () => {
    responseName = 'partial';
    const user = userEvent.setup();
    render(<App apiBaseUrl={apiBaseUrl} />);
    await user.type(screen.getByRole('textbox', { name: '질문' }), '두 자료를 비교해 줘');
    await waitFor(() => expect(screen.getByRole('button', { name: '질문 보내기' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: '질문 보내기' }));
    expect(await screen.findByText('확인한 첫 번째 자료의 답변 🧪')).toBeVisible();
    expect(screen.getByText('일부 처리')).toBeVisible();
    expect(screen.getByText('두 번째 자료는 확인하지 못했습니다.')).toBeVisible();
  });

  it('수신 중단 시 질문을 재전송하지 않고 제출한 원문을 남긴다', async () => {
    holdResponse = true;
    const user = userEvent.setup();
    render(<App apiBaseUrl={apiBaseUrl} />);
    await user.type(screen.getByRole('textbox', { name: '질문' }), '결과를 저장해 줘');
    await waitFor(() => expect(screen.getByRole('button', { name: '질문 보내기' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: '질문 보내기' }));
    expect(await screen.findByText('질문을 확인하고 있습니다.')).toBeVisible();
    await user.click(screen.getByRole('button', { name: '응답 수신 중단' }));
    expect(await screen.findByText(/서버 작업은 계속될 수 있습니다/)).toBeVisible();
    expect(screen.getByText('결과를 저장해 줘')).toBeVisible();
    expect(requests).toHaveLength(1);
    expect(screen.queryByText('답변 받음')).not.toBeInTheDocument();
  });

  it('서버의 문자 수와 UTF-8 바이트 제한을 원문 기준으로 적용한다', async () => {
    config.query.max_chars = 2;
    config.query.max_utf8_bytes = 5;
    const user = userEvent.setup();
    render(<App apiBaseUrl={apiBaseUrl} />);
    const question = screen.getByRole('textbox', { name: '질문' });
    await user.type(question, '🧪a');
    await waitFor(() => expect(screen.getByRole('button', { name: '질문 보내기' })).toBeEnabled());
    await user.clear(question);
    await user.type(question, '한글');
    expect(screen.getByRole('button', { name: '질문 보내기' })).toBeDisabled();
    expect(screen.getByRole('alert')).toHaveTextContent('질문이 서버의 길이 제한을 넘었습니다.');
    await user.clear(question);
    await user.type(question, 'abc');
    expect(screen.getByRole('button', { name: '질문 보내기' })).toBeDisabled();
    expect(requests).toHaveLength(0);
  });
});
