import { expect, test } from '@playwright/test';
import type { AgentResponse } from '../../src/api/types';

const fixtureUrl = 'http://127.0.0.1:18000';
const cases = [
  { code: 'request_timeout', query: '전체 시간 소진 검증: 질문의 처리 결과를 확인해줘.', status: 'failed', upload: false },
  { code: 'answer_validation_failed', query: '답변 조건 실패 검증: 요청 조건에 맞는 답변을 작성해줘.', status: 'failed', upload: false },
  { code: 'evidence_insufficient', query: '근거 부족 구분 검증: 첨부에서 alpha와 beta를 설명해줘.', status: 'partial', upload: true },
] as const;

for (const scenario of cases) {
  test(`${scenario.code} keeps the server cause and request ID without automatically resubmitting`, async ({ page, request }) => {
    await page.goto('./');
    if (scenario.upload) {
      await page.getByText('첨부 자료', { exact: true }).click();
      await page.getByLabel('파일 첨부', { exact: true }).setInputFiles({
        name: 'alpha.py', mimeType: 'text/x-python', buffer: Buffer.from('alpha = 1\n'),
      });
      await page.getByRole('button', { name: '첨부만 확정', exact: true }).click();
      await expect(page.getByRole('button', { name: 'alpha.py 첨부 해제', exact: true })).toBeVisible();
    }
    await page.getByRole('textbox', { name: '질문', exact: true }).fill(scenario.query);
    const submit = page.getByRole('button', { name: '질문 보내기', exact: true });
    await expect(submit).toBeEnabled();
    const streamed = page.waitForResponse((response) =>
      response.url().endsWith('/agent/stream') && response.request().method() === 'POST');
    await submit.click();
    expect((await streamed).status()).toBe(200);
    // The client closes SSE after final_response, so observe its persisted
    // result instead of racing Chromium's access to the cancelled stream body.
    const storedResult = await page.waitForFunction((query) => {
      const snapshot = JSON.parse(sessionStorage.getItem('documate.react.conversations.v1') ?? 'null') as {
        conversations: { turns: { query: string; result: AgentResponse | null }[] }[];
      } | null;
      return snapshot?.conversations.flatMap((conversation) => conversation.turns)
        .find((turn) => turn.query === query)?.result;
    }, scenario.query);
    const result = await storedResult.jsonValue() as AgentResponse;

    expect(result.status).toBe(scenario.status);
    expect(result.problem?.code).toBe(scenario.code);
    expect(result.request_id).toMatch(/^[0-9a-f]{8}$/);
    const cause = scenario.upload ? page.getByRole('region', { name: '부분 답변' }) : page;
    await expect(cause.getByText(result.problem!.message, { exact: true })).toBeVisible();
    await page.getByText(scenario.upload ? '답변 정보' : '요청 정보', { exact: true }).click();
    await expect(page.getByText(`요청 ID: ${result.request_id}`, { exact: true })).toBeVisible();
    if (scenario.upload) await expect(page.getByText('alpha = 1', { exact: true }).first()).toBeVisible();
    else await expect(page.locator('.answer-body')).toHaveCount(0);

    await page.reload();
    await expect(cause.getByText(result.problem!.message, { exact: true })).toBeVisible();
    await expect(page.getByRole('textbox', { name: '질문', exact: true })).toBeEnabled();
    const received = await (await request.get(`${fixtureUrl}/_fixture/requests`)).json() as { queries: string[] };
    expect(received.queries.filter((query) => query === scenario.query)).toEqual([scenario.query]);
  });
}
