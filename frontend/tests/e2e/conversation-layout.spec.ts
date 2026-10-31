import { expect, test } from '@playwright/test';

test('request information stays out of the answer until opened and keeps its rendered identifier', async ({ page }) => {
  await page.goto('./');
  const query = page.getByRole('textbox', { name: '질문', exact: true });
  await query.fill('대화 화면의 요청 정보를 확인해줘.');
  await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
  await expect(page.getByText('답변 받음', { exact: true })).toBeVisible();
  const requestInformation = page.locator('.request-id');
  const identifier = await requestInformation.textContent();
  expect(identifier).toMatch(/^요청 ID: [0-9a-f]{8}$/);
  await expect(requestInformation).not.toBeVisible();
  await page.getByText('요청 정보', { exact: true }).click();
  await expect(requestInformation).toBeVisible();
  await expect(requestInformation).toHaveText(identifier!);
  await expect(query).toHaveAttribute('placeholder', '이어서 질문하세요');
  await query.fill('  이어서 작성 중인 질문\n둘째 줄  ');
  await page.getByText('요청 정보', { exact: true }).click();
  await expect(requestInformation).not.toBeVisible();
  await expect(query).toHaveValue('  이어서 작성 중인 질문\n둘째 줄  ');
});

test('submitting a follow-up brings the new question into view after a long answer', async ({ page, request }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto('./');
  const query = page.getByRole('textbox', { name: '질문', exact: true });
  const markdown = Array.from({ length: 30 }, (_, index) => `${index + 1}. 리스트와 튜플의 사용 예를 비교합니다.`).join('\n\n');
  await query.fill(`다음 Markdown 원문을 그대로 복사해줘.\n\n${markdown}`);
  await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
  await expect(page.getByText('답변 받음', { exact: true })).toBeVisible();
  const followUp = '위 설명에서 마지막 예시를 더 설명해줘.';
  await request.post('http://127.0.0.1:18000/_fixture/hold', { data: { query: followUp } });
  await query.fill(followUp);
  await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
  await expect(page.getByRole('button', { name: '응답 수신 중단', exact: true })).toBeVisible();
  await expect(page.locator('.user-message').getByText(followUp, { exact: true })).toBeInViewport({ ratio: 1 });
  await request.post('http://127.0.0.1:18000/_fixture/release', { data: { query: followUp } });
  await expect(page.getByText('답변 받음', { exact: true })).toHaveCount(2);
});

for (const viewport of [{ width: 1280, height: 800 }, { width: 390, height: 844 }]) {
  test(`long code and a comparison table leave follow-up input usable at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await page.goto('./');
    const query = page.getByRole('textbox', { name: '질문', exact: true });
    await query.fill('리스트와 튜플을 비교하고 싶어요.');
    await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
    await expect(page.getByText('답변 받음', { exact: true })).toBeVisible();
    const code = 'values = ["a_long_value_that_needs_horizontal_scrolling_without_widening_the_chat_column", "another_long_value"]\n'
      + 'for value in values:\n    print(value)\n';
    const markdown = '## 리스트와 튜플 비교\n\n'
      + '| 구분 | 리스트 | 튜플 |\n| --- | --- | --- |\n| 값 변경 | 가능 | 불가 |\n\n'
      + `\`\`\`python\n${code}\`\`\`\n`;
    await query.fill(`다음 Markdown 원문을 그대로 복사해줘.\n\n${markdown}`);
    await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
    await expect(page.getByText('답변 받음', { exact: true })).toHaveCount(2);
    const table = page.getByRole('table');
    await expect(table.getByRole('columnheader')).toHaveText(['구분', '리스트', '튜플']);
    await expect(table.getByRole('cell')).toHaveText(['값 변경', '가능', '불가']);
    expect(await page.locator('.answer-body pre code').textContent()).toBe(code);
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await query.fill('예제에서 값이 바뀌는 부분을 더 설명해줘.');
    await expect(query).toBeInViewport();
    await expect(page.getByRole('button', { name: '질문 보내기', exact: true })).toBeInViewport();
    await expect(query).toHaveValue('예제에서 값이 바뀌는 부분을 더 설명해줘.');
    await expect(page.getByText('답변 받음', { exact: true })).toHaveCount(2);
    await page.screenshot({ path: test.info().outputPath('conversation.png'), fullPage: true });
  });
}
