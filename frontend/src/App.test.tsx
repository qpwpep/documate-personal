import { createServer, type ServerResponse } from 'node:http';
import { once } from 'node:events';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import fixtures from '../../tests/fixtures/web_contracts/agent_responses.json';
import { App } from './App';
import { createConversationState, serializeConversations, STORAGE_KEY } from './conversations';

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
let serverEpoch = 'contract-epoch';
let manifestUnavailable = false;
let holdManifest = false;
let pendingManifestResponse: ServerResponse | null = null;

const server = createServer(async (request, response) => {
  if (request.url === '/client-config') {
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify(config));
    return;
  }
  if (request.method === 'GET' && request.url?.endsWith('/uploads')) {
    if (holdManifest) {
      pendingManifestResponse = response;
      return;
    }
    if (manifestUnavailable) {
      response.writeHead(503, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify({ detail: 'temporarily unavailable' }));
      return;
    }
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify({ ...manifest, epoch: serverEpoch }));
    return;
  }
  if (request.url === '/slack/default-recipient') {
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify({ selector: null, failure: null }));
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
  sessionStorage.clear();
  pendingResponse?.end();
  pendingResponse = null;
  pendingManifestResponse?.end();
  pendingManifestResponse = null;
  holdManifest = false;
  requests = [];
  holdResponse = false;
  responseName = 'completed';
  serverEpoch = 'contract-epoch';
  manifestUnavailable = false;
  config.query.max_chars = 8192;
  config.query.max_utf8_bytes = 16384;
});

afterAll(async () => {
  pendingResponse?.end();
  pendingManifestResponse?.end();
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
    expect(within(screen.getByRole('region', { name: '부분 답변' })).getByText('두 번째 자료는 확인하지 못했습니다.')).toBeVisible();
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
    expect(within(screen.getByRole('region', { name: '대화 기록' })).getByText('결과를 저장해 줘')).toBeVisible();
    expect(requests).toHaveLength(1);
    expect(screen.queryByText('답변 받음')).not.toBeInTheDocument();
  });

  it('전송 전 연결 확인을 중단하면 초안을 보존하고 재연결 후 명시적으로 한 번 보낸다', async () => {
    const user = userEvent.setup();
    render(<App apiBaseUrl={apiBaseUrl} />);
    await screen.findByText('질문할 준비가 되었습니다.');
    const question = screen.getByRole('textbox', { name: '질문' });
    const query = '  중단 뒤 다시 보낼 질문  ';
    await user.type(question, query);
    holdManifest = true;
    await user.click(screen.getByRole('button', { name: '질문 보내기' }));
    await waitFor(() => expect(pendingManifestResponse).not.toBeNull());
    await user.click(screen.getByRole('button', { name: '응답 수신 중단' }));

    const reconnect = await screen.findByRole('button', { name: '연결 다시 확인' });
    await waitFor(() => expect(reconnect).toBeEnabled());
    expect(question).toHaveValue(query);
    expect(screen.queryByRole('region', { name: '대화 기록' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '질문 보내기' })).toBeDisabled();
    expect(requests).toHaveLength(0);

    holdManifest = false;
    pendingManifestResponse?.end();
    await user.click(reconnect);
    await screen.findByText('질문할 준비가 되었습니다.');
    expect(question).toHaveValue(query);
    expect(requests).toHaveLength(0);
    await user.click(screen.getByRole('button', { name: '질문 보내기' }));
    await screen.findByText('한글 답변 🧪');
    expect(requests).toHaveLength(1);
    expect(JSON.parse(requests[0]).query).toBe(query);
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

  it('예시 선택은 초안만 채우고 작성한 질문을 덮어쓰지 않는다', async () => {
    const user = userEvent.setup();
    render(<App apiBaseUrl={apiBaseUrl} />);
    const example = await screen.findByRole('button', { name: 'pandas merge 사용법을 공식 문서 기준으로 설명해줘' });
    await waitFor(() => expect(example).toBeEnabled());
    await user.click(example);
    expect(screen.getByRole('textbox', { name: '질문' })).toHaveValue('pandas merge 사용법을 공식 문서 기준으로 설명해줘');
    expect(example).toBeDisabled();
    expect(requests).toHaveLength(0);
  });

  it('Shift+Enter는 초안에 개행을 남기고 Enter를 눌러야 정확한 원문을 보낸다', async () => {
    const user = userEvent.setup();
    render(<App apiBaseUrl={apiBaseUrl} />);
    const question = screen.getByRole('textbox', { name: '질문' });
    await user.type(question, '  첫 줄');
    await user.keyboard('{Shift>}{Enter}{/Shift}둘째 줄  ');
    await waitFor(() => expect(screen.getByRole('button', { name: '질문 보내기' })).toBeEnabled());
    expect(question).toHaveValue('  첫 줄\n둘째 줄  ');
    expect(requests).toHaveLength(0);
    await user.keyboard('{Enter}');
    await screen.findByText('한글 답변 🧪');
    expect(requests).toHaveLength(1);
    expect(JSON.parse(requests[0]).query).toBe('  첫 줄\n둘째 줄  ');
  });

  it('새 대화와 이전 대화의 메시지와 정확한 초안을 따로 보존한다', async () => {
    const user = userEvent.setup();
    render(<App apiBaseUrl={apiBaseUrl} />);
    await user.type(screen.getByRole('textbox', { name: '질문' }), '첫 번째 질문');
    await waitFor(() => expect(screen.getByRole('button', { name: '질문 보내기' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: '질문 보내기' }));
    await screen.findByText('한글 답변 🧪');
    await user.type(screen.getByRole('textbox', { name: '질문' }), '  첫 대화 초안  ');
    await user.click(screen.getByRole('button', { name: '새 채팅' }));
    await user.type(screen.getByRole('textbox', { name: '질문' }), '두 번째 초안');
    await user.click(screen.getByRole('button', { name: '대화 선택: 첫 번째 질문' }));
    expect(screen.getByRole('textbox', { name: '질문' })).toHaveValue('  첫 대화 초안  ');
    expect(screen.getByText('한글 답변 🧪')).toBeVisible();
    expect(requests).toHaveLength(1);
  });

  it('다른 대화에서 답변을 받아도 진행 중 대화의 결과와 다음 초안을 섞지 않는다', async () => {
    holdResponse = true;
    const user = userEvent.setup();
    render(<App apiBaseUrl={apiBaseUrl} />);
    await user.type(screen.getByRole('textbox', { name: '질문' }), '오래 걸리는 질문');
    await waitFor(() => expect(screen.getByRole('button', { name: '질문 보내기' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: '질문 보내기' }));
    await screen.findByText('질문을 확인하고 있습니다.');
    const firstResponse = pendingResponse!;
    await user.type(screen.getByRole('textbox', { name: '질문' }), '  첫 대화의 다음 초안  ');
    await user.click(screen.getByRole('button', { name: '새 채팅' }));
    holdResponse = false;
    await user.type(screen.getByRole('textbox', { name: '질문' }), '두 번째 대화 질문');
    await waitFor(() => expect(screen.getByRole('button', { name: '질문 보내기' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: '질문 보내기' }));
    await screen.findByText('한글 답변 🧪');
    await user.type(screen.getByRole('textbox', { name: '질문' }), '두 번째 다음 초안');
    firstResponse.end(`event: final_response\ndata: ${JSON.stringify(fixtures.responses.partial)}\n\n`);
    await waitFor(() => expect(screen.queryByText('응답 받는 중')).not.toBeInTheDocument());
    expect(screen.queryByText('확인한 첫 번째 자료의 답변 🧪')).not.toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: '질문' })).toHaveValue('두 번째 다음 초안');
    await user.click(screen.getByRole('button', { name: '대화 선택: 오래 걸리는 질문' }));
    expect(screen.getByText('확인한 첫 번째 자료의 답변 🧪')).toBeVisible();
    expect(screen.getByRole('textbox', { name: '질문' })).toHaveValue('  첫 대화의 다음 초안  ');
    expect(new Set(requests.map((body) => JSON.parse(body).session_id)).size).toBe(2);
  });

  it('파일을 선택하면 예시와 실패 질문의 복구를 첨부 변경이 끝날 때까지 숨긴다', async () => {
    const user = userEvent.setup();
    render(<App apiBaseUrl={apiBaseUrl} />);
    await screen.findByRole('button', { name: 'pandas merge 사용법을 공식 문서 기준으로 설명해줘' });
    await user.upload(screen.getByLabelText('파일 첨부'), new File(['print(1)'], 'sample.py'));
    expect(screen.queryByRole('button', { name: 'pandas merge 사용법을 공식 문서 기준으로 설명해줘' })).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'sample.py 선택 취소 1' }));
    responseName = 'failed';
    await user.type(screen.getByRole('textbox', { name: '질문' }), '실패한 질문');
    await user.click(screen.getByRole('button', { name: '질문 보내기' }));
    await screen.findByRole('button', { name: '질문 수정' });
    await user.upload(screen.getByLabelText('파일 첨부'), new File(['print(2)'], 'next.py'));
    expect(screen.queryByRole('button', { name: '질문 수정' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '같은 질문 다시 보내기' })).not.toBeInTheDocument();
    expect(requests).toHaveLength(1);
  });

  it('새로고침 뒤 기록과 초안을 복원하고 서버 만료를 읽기 전용으로 표시한다', async () => {
    const user = userEvent.setup();
    const first = render(<App apiBaseUrl={apiBaseUrl} />);
    await user.type(screen.getByRole('textbox', { name: '질문' }), '복원할 질문');
    await waitFor(() => expect(screen.getByRole('button', { name: '질문 보내기' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: '질문 보내기' }));
    await screen.findByText('한글 답변 🧪');
    await user.type(screen.getByRole('textbox', { name: '질문' }), '  남겨 둔 초안  ');
    first.unmount();
    serverEpoch = 'new-server-session';
    render(<App apiBaseUrl={apiBaseUrl} />);
    expect(screen.getByText('한글 답변 🧪')).toBeVisible();
    expect(screen.getByRole('textbox', { name: '질문' })).toHaveValue('  남겨 둔 초안  ');
    expect(await screen.findByText(/서버 세션이 만료되어 읽기 전용/)).toBeVisible();
    expect(screen.getByRole('button', { name: '질문 보내기' })).toBeDisabled();
    expect(requests).toHaveLength(1);
  });

  it('응답 중 새로고침은 제출한 질문을 중단 기록으로 복원하고 다시 보내지 않는다', async () => {
    holdResponse = true;
    const user = userEvent.setup();
    const first = render(<App apiBaseUrl={apiBaseUrl} />);
    await user.type(screen.getByRole('textbox', { name: '질문' }), '저장할 질문');
    await waitFor(() => expect(screen.getByRole('button', { name: '질문 보내기' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: '질문 보내기' }));
    await screen.findByText('질문을 확인하고 있습니다.');
    first.unmount();
    render(<App apiBaseUrl={apiBaseUrl} />);
    expect(within(screen.getByRole('region', { name: '대화 기록' })).getByText('저장할 질문')).toBeVisible();
    expect(await screen.findByRole('button', { name: '질문 수정' })).toBeVisible();
    await user.click(screen.getByRole('button', { name: '질문 수정' }));
    expect(screen.getByRole('textbox', { name: '질문' })).toHaveValue('저장할 질문');
    expect(requests).toHaveLength(1);
  });

  it('보류한 질문을 보내기 전 연결 확인이 실패해도 질문 원문을 보존한다', async () => {
    const restored = createConversationState();
    restored.conversations[0].attachments.pendingPrompt = '  첨부 후 보낼 질문  ';
    restored.conversations[0].manifest = manifest;
    restored.conversations[0].confirmedEpoch = manifest.epoch;
    sessionStorage.setItem(STORAGE_KEY, serializeConversations(restored));
    const user = userEvent.setup();
    render(<App apiBaseUrl={apiBaseUrl} />);
    await screen.findByText('질문할 준비가 되었습니다.');
    manifestUnavailable = true;
    await user.click(screen.getByRole('button', { name: '보류한 질문 보내기' }));
    await screen.findByRole('button', { name: '연결 다시 확인' });
    expect(screen.getByText('첨부 후 보낼 질문')).toHaveTextContent('첨부 후 보낼 질문');
    expect(sessionStorage.getItem(STORAGE_KEY)).toContain('  첨부 후 보낼 질문  ');
    expect(requests).toHaveLength(0);
  });

  it.each(['첨부 없이 계속', '파일 다시 선택'])('%s 복구에서 새 초안과 보류 질문을 모두 보존한다', async (recovery) => {
    const user = userEvent.setup({ applyAccept: false });
    const app = render(<App apiBaseUrl={apiBaseUrl} />);
    await screen.findByText('질문할 준비가 되었습니다.');
    const pending = '  첨부 파일을 설명해 줘  ';
    const newerDraft = '  나중에 보낼 새 질문  ';
    await user.upload(screen.getByLabelText('파일 첨부'), new File(['bad'], 'wrong.exe'));
    await user.type(screen.getByRole('textbox', { name: '질문' }), pending);
    await user.click(screen.getByRole('button', { name: '질문 보내기' }));
    await screen.findByText(/wrong.exe: 지원하지 않는 파일 형식/);
    await user.type(screen.getByRole('textbox', { name: '질문' }), newerDraft);
    await user.click(screen.getByRole('button', { name: recovery }));

    expect(screen.getByRole('textbox', { name: '질문' })).toHaveValue(newerDraft);
    expect(screen.getByText(pending.trim())).toBeVisible();
    expect(requests).toHaveLength(0);
    app.unmount();
    render(<App apiBaseUrl={apiBaseUrl} />);
    await screen.findByText('질문할 준비가 되었습니다.');
    expect(screen.getByRole('textbox', { name: '질문' })).toHaveValue(newerDraft);
    await user.click(screen.getByRole('button', { name: '보류한 질문 보내기' }));
    await screen.findByText('한글 답변 🧪');
    expect(JSON.parse(requests[0]).query).toBe(pending);
    expect(screen.getByRole('textbox', { name: '질문' })).toHaveValue(newerDraft);
    expect(requests).toHaveLength(1);
  });

  it('파일 전송 직전 서버 만료를 확인하면 첨부 오류 뒤에도 읽기 전용을 유지한다', async () => {
    const user = userEvent.setup();
    render(<App apiBaseUrl={apiBaseUrl} />);
    await screen.findByText('질문할 준비가 되었습니다.');
    await user.upload(screen.getByLabelText('파일 첨부'), new File(['print(1)'], 'sample.py'));
    serverEpoch = 'changed-epoch';
    await user.click(screen.getByRole('button', { name: '첨부만 확정' }));
    expect(await screen.findByText(/서버 세션이 만료되어 읽기 전용/)).toBeVisible();
    expect(screen.queryByRole('button', { name: '연결 다시 확인' })).not.toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: '질문' })).toHaveAttribute('readonly');
    expect(requests).toHaveLength(0);
  });
});
