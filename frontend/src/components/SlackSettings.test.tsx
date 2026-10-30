import { createServer } from 'node:http';
import { once } from 'node:events';
import { useState } from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { SlackSettings } from './SlackSettings';
import { createSlackSettings, selectedSlackRecipient } from './SlackSettingsState';

let apiBaseUrl = '';
let responseBody: object = { selector: null, failure: null };
let status = 200;
let requests: string[] = [];
const server = createServer((request, response) => {
  requests.push(`${request.method} ${request.url}`);
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(responseBody));
});

beforeAll(async () => {
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Test server did not bind a TCP port');
  apiBaseUrl = `http://127.0.0.1:${address.port}`;
});
beforeEach(() => {
  responseBody = { selector: null, failure: null };
  status = 200;
  requests = [];
});
afterAll(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
});

function Settings() {
  const [value, setValue] = useState(createSlackSettings);
  return <><SlackSettings baseUrl={apiBaseUrl} value={value} onChange={setValue} />
    <output aria-label="요청 수신자">{JSON.stringify(selectedSlackRecipient(value))}</output></>;
}

describe('Slack 수신자 설정', () => {
  it('preserves each raw recipient draft and never sends a request merely for selecting a destination', async () => {
    const user = userEvent.setup();
    render(<Settings />);
    await user.click(screen.getByText('Slack 수신자 설정'));
    await screen.findByText(/기본 수신자가 없습니다/);
    await user.click(screen.getByRole('radio', { name: '채널' }));
    const recipient = screen.getByRole('textbox', { name: 'Slack 수신자' });
    await user.type(recipient, '  general  ');
    expect(recipient).toHaveAttribute('aria-invalid', 'true');
    expect(screen.getByText(/일반 질문은 계속할 수 있습니다/)).toBeVisible();
    expect(screen.getByLabelText('요청 수신자').textContent).toBe('{"kind":"channel","value":"  general  "}');
    await user.click(screen.getByRole('radio', { name: '사용자' }));
    await user.type(screen.getByRole('textbox', { name: 'Slack 수신자' }), 'U123ABC');
    expect(screen.getByRole('textbox', { name: 'Slack 수신자' })).toHaveAttribute('aria-invalid', 'false');
    await user.click(screen.getByRole('radio', { name: '채널' }));
    expect(screen.getByRole('textbox', { name: 'Slack 수신자' })).toHaveValue('  general  ');
    await user.clear(screen.getByRole('textbox', { name: 'Slack 수신자' }));
    expect(screen.getByLabelText('요청 수신자')).toHaveTextContent('{"kind":"channel","value":""}');
    expect(requests).toEqual(['GET /slack/default-recipient']);
  });

  it('distinguishes unavailable, invalid and omitted defaults and refreshes only explicitly', async () => {
    status = 503;
    const user = userEvent.setup();
    const { rerender } = render(<Settings />);
    await user.click(screen.getByText('Slack 수신자 설정'));
    expect(await screen.findByText(/기본 수신자를 확인하지 못했습니다/)).toBeVisible();
    expect(screen.queryByText(/기본 수신자가 없습니다/)).not.toBeInTheDocument();
    rerender(<Settings />);
    await waitFor(() => expect(requests).toHaveLength(1));
    status = 200;
    responseBody = { selector: null, failure: { stage: 'selection', code: 'configuration_error',
      message: '기본 Slack 수신자 설정이 잘못되었습니다.', next_action: 'fix_configuration',
      retry_after_seconds: null, slack_error: null } };
    await user.click(screen.getByRole('button', { name: '기본 수신자 다시 확인' }));
    expect(await screen.findByText('기본 Slack 수신자 설정이 잘못되었습니다.')).toBeVisible();
    responseBody = { selector: { kind: 'email', value: 'recipient@example.com' }, failure: null };
    await user.click(screen.getByRole('button', { name: '기본 수신자 다시 확인' }));
    expect(await screen.findByText('recipient@example.com')).toBeVisible();
    expect(screen.getByRole('radio', { name: '직접 지정 안 함' })).toBeChecked();
    expect(screen.getByLabelText('요청 수신자')).toHaveTextContent('null');
    expect(requests).toEqual(Array(3).fill('GET /slack/default-recipient'));
  });
});
