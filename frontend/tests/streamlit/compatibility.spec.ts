import { expect, test } from '@playwright/test';

test('Streamlit and its Python client retain answers, exact drafts, and TXT exports', async ({ page, request }) => {
  const query = '기존 Streamlit과 Python 클라이언트 호환성을 확인해줘.';
  const draft = '  아직 보내지 않은 다음 질문 🧪\n줄바꿈과 공백  ';
  await page.goto('/');
  const composer = page.getByRole('textbox', { name: '질문을 입력하세요', exact: true });
  await expect(composer).toBeEnabled();
  await composer.fill(query);
  await composer.press('Enter');
  const answer = page.getByRole('group', { name: 'Chat message from assistant', exact: true });
  await expect(answer).toContainText(`회귀 검증 답변: ${query}`);
  await expect(composer).toBeEnabled();

  await composer.fill(draft);
  // The browser bridge stores unsubmitted typing, independently of Streamlit reruns.
  await expect.poll(() => page.evaluate(() =>
    Object.values(sessionStorage).some((value) => value.includes('아직 보내지 않은 다음 질문')),
  )).toBe(true);
  await page.reload();
  await expect(composer).toHaveValue(draft);
  await expect(answer).toContainText(`회귀 검증 답변: ${query}`);

  await page.getByRole('button', { name: 'TXT 파일 만들기', exact: true }).click();
  const download = page.getByRole('link', { name: 'TXT 다운로드', exact: true });
  await expect(download).toBeVisible();
  const url = await download.getAttribute('href');
  expect(url).toBeTruthy();
  const result = await request.get(url!);
  expect(result.status()).toBe(200);
  const text = await result.text();
  expect(text).toContain(`회귀 검증 답변: ${query}`);
  expect(text).toContain('브라우저 회귀 검증 자료');
  await expect(composer).toHaveValue(draft);

  const metricsUrl = new URL('/_fixture/metrics', url!).href;
  const metrics = await (await request.get(metricsUrl)).json() as {
    questions: { query: string }[];
    exports: { answer_id: string }[];
  };
  expect(metrics.questions.filter((item) => item.query === query)).toHaveLength(1);
  expect(metrics.exports).toHaveLength(1);
  await page.screenshot({ path: test.info().outputPath('streamlit-compatible.png'), fullPage: true });
});
