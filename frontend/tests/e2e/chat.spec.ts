import { expect, test } from '@playwright/test';

const fixtureUrl = 'http://127.0.0.1:18000';

test('an explicit submission reaches the real API and renders its checked answer once', async ({ page, request }) => {
  const query = '  대기 검증: 브라우저 연결을 확인해줘. 🧪\n';
  const consoleErrors: string[] = [];
  page.on('pageerror', (error) => consoleErrors.push(error.message));

  await page.goto('./');
  const submit = page.getByRole('button', { name: '질문 보내기', exact: true });
  await page.getByRole('textbox', { name: '질문', exact: true }).fill(query);
  await expect(submit).toBeEnabled();

  const responsePromise = page.waitForResponse((response) =>
    response.url().endsWith('/agent/stream') && response.request().method() === 'POST');
  await submit.click();
  await expect(page.getByRole('button', { name: '응답 수신 중단', exact: true })).toBeVisible();
  await expect.poll(async () => {
    const response = await request.get(`${fixtureUrl}/_fixture/requests`);
    const received = await response.json() as { queries: string[] };
    return received.queries.filter((value) => value === query).length;
  }).toBe(1);

  await request.post(`${fixtureUrl}/_fixture/release`);
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  expect(response.headers()['content-type']).toContain('text/event-stream');
  await expect(page.getByText('답변 받음', { exact: true })).toBeVisible();
  await expect(page.getByText('실제 FastAPI와 Python 런타임을 거친 한글 답변입니다. 🧪', { exact: true })).toBeVisible();
  await expect(page.getByRole('textbox', { name: '질문', exact: true })).toBeEnabled();
  expect(consoleErrors).toEqual([]);

  const received = await (await request.get(`${fixtureUrl}/_fixture/requests`)).json() as { queries: string[] };
  expect(received.queries.filter((value) => value === query)).toEqual([query]);
  await page.screenshot({ path: test.info().outputPath('completed.png'), fullPage: true });
});

test('a provider failure is shown without resubmitting the question', async ({ page, request }) => {
  const query = '제공자 오류 검증: 질문을 한 번만 보내줘.';
  await page.goto('./');
  await page.getByRole('textbox', { name: '질문', exact: true }).fill(query);
  const submit = page.getByRole('button', { name: '질문 보내기', exact: true });
  await expect(submit).toBeEnabled();
  await submit.click();

  await expect(page.getByText('요청 실패', { exact: true })).toBeVisible();
  await expect(page.getByText(
    '서비스의 요청 처리 설정에 오류가 있어 처리를 완료하지 못했습니다. 질문을 수정할 필요는 없습니다.',
    { exact: true },
  )).toBeVisible();
  await expect(page.getByText('실제 FastAPI와 Python 런타임을 거친 한글 답변입니다. 🧪', { exact: true })).toHaveCount(0);
  const received = await (await request.get(`${fixtureUrl}/_fixture/requests`)).json() as { queries: string[] };
  expect(received.queries.filter((value) => value === query)).toEqual([query]);
});
