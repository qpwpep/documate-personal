import { expect, test } from '@playwright/test';

test('the composer resizes its draft when the available width changes', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.addInitScript(() => {
    const browserErrors: string[] = [];
    Object.defineProperty(window, '__testWindowErrors', { value: browserErrors });
    // Vite handles ResizeObserver error events before Playwright emits pageerror.
    window.addEventListener('error', (event) => browserErrors.push(event.message));
  });
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('./');
  const query = page.getByRole('textbox', { name: '질문', exact: true });
  const draft = '공식 문서의 예시를 읽고 각 단계의 의미를 설명해 주세요.\n작성 중인 질문은 화면 크기를 바꾸어도 모두 읽을 수 있어야 합니다.';
  for (const afterFirstAnswer of [false, true]) {
    if (afterFirstAnswer) {
      await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
      await expect(page.getByText('답변 받음', { exact: true })).toBeVisible();
    }
    await query.fill(draft);
    for (const viewport of [{ width: 320, height: 720 }, { width: 390, height: 844 }, { width: 1280, height: 800 }]) {
      await page.setViewportSize(viewport);
      await expect(query).toHaveValue(draft);
      await expect.poll(() => query.evaluate((element) => element.scrollHeight <= element.clientHeight + 1)).toBe(true);
    }
  }
  expect(errors).toEqual([]);
  expect(await page.evaluate(() => (window as Window & { __testWindowErrors?: string[] }).__testWindowErrors)).toEqual([]);
});

test('the latest-answer shortcut preserves the reading position until explicitly selected', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto('./');
  const query = page.getByRole('textbox', { name: '질문', exact: true });
  const markdown = Array.from({ length: 35 }, (_, index) => `${index + 1}. 문서의 근거를 읽고 이해합니다.`).join('\n\n');
  await query.fill(`다음 Markdown 원문을 그대로 복사해줘.\n\n${markdown}`);
  await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
  await expect(page.getByText('답변 받음', { exact: true })).toBeVisible();
  const jump = page.getByRole('button', { name: '최신 답변으로 이동', exact: true });
  await expect(jump).toBeVisible();
  await expect(page.locator('.user-message')).toBeInViewport();
  await query.fill('작성 중인 후속 질문');
  await jump.click();
  await expect(query).toBeFocused();
  await expect(query).toHaveValue('작성 중인 후속 질문');
  await expect(jump).not.toBeVisible();
  await expect(page.getByRole('button', { name: 'TXT 파일 만들기', exact: true })).toBeInViewport();
});

test('answer information keeps the full execution identifier collapsed without losing the composer draft', async ({ page }) => {
  await page.goto('./');
  const query = page.getByRole('textbox', { name: '질문', exact: true });
  await query.fill('대화 화면의 실행 정보를 확인해줘.');
  const streamed = page.waitForResponse((response) =>
    response.url().endsWith('/agent/stream') && response.request().method() === 'POST');
  await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
  await expect(page.getByText('답변 받음', { exact: true })).toBeVisible();
  const executionId = (await streamed).headers()['x-execution-id'];
  expect(executionId).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  const executionInformation = page.locator('.request-id');
  await expect(executionInformation).toHaveText(`실행 ID: ${executionId}`);
  await expect(executionInformation).not.toBeVisible();
  await page.getByText('답변 정보', { exact: true }).click();
  await expect(executionInformation).toBeVisible();
  await expect(executionInformation).toHaveText(`실행 ID: ${executionId}`);
  await expect(query).toHaveAttribute('placeholder', '이어서 질문하세요');
  await query.fill('  이어서 작성 중인 질문\n둘째 줄  ');
  await page.getByText('답변 정보', { exact: true }).click();
  await expect(executionInformation).not.toBeVisible();
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
