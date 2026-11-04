import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import type { AgentResponse, AnswerExportResult } from '../../src/api/types';

const fixtureUrl = 'http://127.0.0.1:18000';
const answer = '실제 FastAPI와 Python 런타임을 거친 한글 답변입니다. 🧪';
type ObservedEvent = { event: string; data: Record<string, unknown> };

declare global {
  interface Window {
    lifecycleEvents: Record<string, ObservedEvent[]>;
    lifecycleExecutionIds: Record<string, string | null>;
    lifecycleErrors: string[];
  }
}

async function observeStreams(page: Page) {
  await page.addInitScript(() => {
    window.lifecycleEvents = {};
    window.lifecycleExecutionIds = {};
    window.lifecycleErrors = [];
    const originalFetch = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      const response = await originalFetch(input, init);
      const url = typeof input === 'string' ? input : input instanceof Request ? input.url : input.href;
      if (!url.endsWith('/agent/stream') || typeof init?.body !== 'string') return response;
      const { query } = JSON.parse(init.body) as { query: string };
      window.lifecycleExecutionIds[query] = response.headers.get('X-Execution-ID');
      const events: ObservedEvent[] = [];
      window.lifecycleEvents[query] = events;
      // Observe a clone; the application's own SSE parser still consumes the
      // untouched response and must accept every event to finish the UI turn.
      const reader = response.clone().body!.getReader();
      void (async () => {
        const decoder = new TextDecoder();
        let pending = '';
        try {
          while (true) {
            const chunk = await reader.read();
            if (chunk.done) break;
            pending += decoder.decode(chunk.value, { stream: true });
            const frames = pending.split('\n\n');
            pending = frames.pop()!;
            for (const frame of frames) {
              const lines = frame.split('\n');
              const event = lines.find((line) => line.startsWith('event: '))?.slice(7);
              const data = lines.filter((line) => line.startsWith('data: ')).map((line) => line.slice(6)).join('\n');
              if (event && data) events.push({ event, data: JSON.parse(data) as Record<string, unknown> });
            }
          }
        } catch (error) {
          // The app deliberately cancels transport after its valid final event.
          if (!events.some((event) => event.event === 'final_response')) window.lifecycleErrors.push(String(error));
        } finally {
          reader.releaseLock();
        }
      })();
      return response;
    };
  });
}

async function ask(page: Page, query: string) {
  await page.getByRole('textbox', { name: '질문', exact: true }).fill(query);
  const submit = page.getByRole('button', { name: '질문 보내기', exact: true });
  await expect(submit).toBeEnabled();
  await submit.click();
}

async function submission(request: APIRequestContext, query: string) {
  const response = await request.get(`${fixtureUrl}/_fixture/requests`);
  const records = await response.json() as {
    submissions: { query: string; session_id: string; uploads: { epoch: string; revision: number } }[];
  };
  const selected = records.submissions.find((item) => item.query === query);
  expect(selected).toBeDefined();
  return selected!;
}

async function eventsFor(page: Page, query: string): Promise<ObservedEvent[]> {
  return await page.evaluate((value) => window.lifecycleEvents[value] ?? [], query);
}

async function captureFinalViewport(page: Page, filename: string) {
  await page.setViewportSize({ width: 1280, height: 1000 });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: test.info().outputPath(filename) });
}

test('heartbeats keep body validation and same-session lock waits alive until both real requests complete', async ({ page, request }) => {
  const firstQuery = '의미 검증 연결 유지 확인';
  const queuedQuery = '같은 세션 잠금 연결 유지 확인';
  await request.post(`${fixtureUrl}/_fixture/decisions/hold`, { data: { query: firstQuery } });
  await observeStreams(page);
  await page.goto('./');
  await ask(page, firstQuery);
  try {
    await expect.poll(async () => (await (await request.get(`${fixtureUrl}/_fixture/decisions`)).json())[firstQuery]).toBe(true);
    await expect.poll(async () => (await eventsFor(page, firstQuery)).filter((event) => event.event === 'heartbeat').length).toBeGreaterThan(0);
    await expect(page.getByRole('button', { name: '응답 수신 중단', exact: true })).toBeVisible();
    const first = await submission(request, firstQuery);
    // The UI correctly blocks another submission in this conversation. A second
    // browser HTTP consumer exercises the real server's session-lock queue.
    await page.evaluate((payload) => {
      void fetch('/agent/stream', {
        method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify(payload),
      }).then((response) => response.text()).catch((error: unknown) => window.lifecycleErrors.push(String(error)));
    }, { query: queuedQuery, session_id: first.session_id, uploads: first.uploads });
    await expect.poll(async () => (await eventsFor(page, queuedQuery)).filter((event) => event.event === 'heartbeat').length).toBeGreaterThan(0);
    const queued = await eventsFor(page, queuedQuery);
    expect(queued.some((event) => event.event === 'stage_started')).toBe(false);
    for (const query of [firstQuery, queuedQuery]) {
      const observed = await eventsFor(page, query);
      const started = observed.find((event) => event.event === 'request_started')!;
      const heartbeat = observed.find((event) => event.event === 'heartbeat')!;
      expect(heartbeat.data).toEqual({ execution_id: started.data.execution_id, elapsed_ms: expect.any(Number) });
      expect(heartbeat.data.elapsed_ms).toBeGreaterThan(0);
    }
  } finally {
    await request.post(`${fixtureUrl}/_fixture/decisions/release`, { data: { query: firstQuery } });
  }
  await expect(page.getByText(answer, { exact: true })).toBeVisible();
  await expect(page.getByText('답변 받음', { exact: true })).toBeVisible();
  await expect.poll(async () => (await eventsFor(page, queuedQuery)).find((event) => event.event === 'final_response')?.data.status).toBe('completed');
  await expect.poll(async () => (await eventsFor(page, firstQuery)).some((event) => event.event === 'final_response')).toBe(true);
  for (const query of [firstQuery, queuedQuery]) {
    const executionId = await page.evaluate((value) => window.lifecycleExecutionIds[value], query);
    expect(executionId).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    const observed = await eventsFor(page, query);
    for (const event of observed.filter((item) => ['request_started', 'heartbeat', 'final_response'].includes(item.event))) {
      expect(event.data.execution_id).toBe(executionId);
    }
  }
  expect(await page.evaluate(([first, queued]) => window.lifecycleExecutionIds[first] !== window.lifecycleExecutionIds[queued],
    [firstQuery, queuedQuery])).toBe(true);
  expect(await page.evaluate(() => window.lifecycleErrors)).toEqual([]);
});

for (const action of ['reload', 'save'] as const) {
  test(`an idle TTL expiry before the next cleanup remains read-only on ${action}`, async ({ page, request }) => {
    const query = `주기 정리 전 실제 유휴 만료 ${action}`;
    const draft = '  만료 후에도 보존할 초안\n';
    await page.goto('./');
    await ask(page, query);
    await expect(page.getByText(answer, { exact: true })).toBeVisible();
    await page.getByRole('textbox', { name: '질문', exact: true }).fill(draft);
    const initial = await submission(request, query);
    const aged = await request.post(`${fixtureUrl}/_fixture/sessions/${initial.session_id}/age-idle`);
    expect(aged.ok()).toBe(true);
    const before = await aged.json() as { epoch: string };
    const saveRequests: string[] = [];
    page.on('request', (outgoing) => {
      if (outgoing.method() === 'POST' && outgoing.url().endsWith('/save')) saveRequests.push(outgoing.url());
    });
    const refreshed = page.waitForResponse((response) => response.request().method() === 'GET'
      && response.url().endsWith(`/sessions/${initial.session_id}/uploads`));
    if (action === 'reload') await page.reload();
    else await page.getByRole('button', { name: 'TXT 파일 만들기', exact: true }).click();
    const manifest = await (await refreshed).json() as { epoch: string };
    expect(manifest.epoch).not.toBe(before.epoch);
    await expect(page.getByText('서버 세션이 만료되어 읽기 전용입니다. 새 채팅에서 계속할 수 있습니다.', { exact: true })).toBeVisible();
    await expect(page.getByText(answer, { exact: true })).toBeVisible();
    const input = page.getByRole('textbox', { name: '질문', exact: true });
    await expect(input).toHaveValue(draft);
    await expect(input).toHaveAttribute('readonly');
    await expect(page.getByRole('button', { name: '질문 보내기', exact: true })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'TXT 파일 만들기', exact: true })).toBeDisabled();
    expect(saveRequests).toEqual([]);
    if (action === 'save') await captureFinalViewport(page, 'idle-expired-read-only.png');
  });
}

test('a natural-language copy save reuses the original artifact after semantic revalidation', async ({ page, request }) => {
  await observeStreams(page);
  await page.goto('./');
  const initialQuery = '한글 답변을 작성해줘';
  await ask(page, initialQuery);
  await expect(page.getByText(answer, { exact: true })).toBeVisible();
  await expect.poll(async () => (await eventsFor(page, initialQuery)).some((item) => item.event === 'final_response')).toBe(true);
  const initialEvent = (await eventsFor(page, initialQuery)).find((item) => item.event === 'final_response')!;
  const initialAnswer = (initialEvent.data as unknown as AgentResponse).response!;
  const saved = page.waitForResponse((response) => response.request().method() === 'POST' && response.url().endsWith('/save'));
  await page.getByRole('button', { name: 'TXT 파일 만들기', exact: true }).click();
  const original = await (await saved).json() as AnswerExportResult;
  const initialReceipt = original.receipt;
  const firstLink = page.getByRole('link', { name: 'TXT 다운로드', exact: true });
  await expect(firstLink).toBeVisible();
  const href = (await firstLink.getAttribute('href'))!;
  const downloadUrl = new URL(href, page.url()).href;
  const initialBytes = await (await request.get(downloadUrl)).body();
  const query = '방금 답변을 그대로 저장해줘. 한글 답변을 유지해줘.';
  await ask(page, query);
  await expect(page.getByText('답변 받음', { exact: true })).toHaveCount(2);
  await expect.poll(async () => (await eventsFor(page, query)).some((item) => item.event === 'final_response')).toBe(true);
  const event = (await eventsFor(page, query)).find((item) => item.event === 'final_response')!;
  const result = event.data as unknown as AgentResponse;
  expect(result.status).toBe('completed');
  expect(result.response!.content).toEqual(initialAnswer.content);
  expect(result.response!.semantic_validation).not.toEqual(initialAnswer.semantic_validation);
  const receipt = result.response!.actions.find((item) => item.kind === 'save_text')!;
  expect(receipt.operation).toEqual(initialReceipt.operation);
  expect(receipt.artifact).toEqual(initialReceipt.artifact);
  const links = page.getByRole('link', { name: 'TXT 다운로드', exact: true });
  await expect(links).toHaveCount(2);
  await expect(links.nth(1)).toHaveAttribute('href', href);
  expect(await (await request.get(downloadUrl)).body()).toEqual(initialBytes);
  expect(initialBytes.toString('utf8')).toContain(answer);
  await captureFinalViewport(page, 'revalidated-copy-save.png');
});
