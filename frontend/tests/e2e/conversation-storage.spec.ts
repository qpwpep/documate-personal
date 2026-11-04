import { expect, test, type Page } from '@playwright/test';

const storageKey = 'documate.react.conversations';
const fixtureAnswer = '실제 FastAPI와 Python 런타임을 거친 한글 답변입니다. 🧪';

interface StoredConversations {
  activeId: string;
  conversations: { id: string; draft: string; turns: { query: string; result: { status: string } | null }[] }[];
}

async function ask(page: Page, query: string) {
  await page.getByRole('textbox', { name: '질문', exact: true }).fill(query);
  const submit = page.getByRole('button', { name: '질문 보내기', exact: true });
  await expect(submit).toBeEnabled();
  await submit.click();
  await expect(page.getByText(fixtureAnswer, { exact: true })).toBeVisible();
}

async function deleteConversation(page: Page, title: string) {
  await page.getByRole('button', { name: `대화 삭제: ${title}`, exact: true }).click();
  await page.getByRole('group', { name: `대화 삭제 확인: ${title}`, exact: true })
    .getByRole('button', { name: '삭제', exact: true }).click();
}

test('reloading immediately after repeated draft edits preserves the exact latest draft', async ({ page }) => {
  const query = '저장 경계에서 초안 보존 확인';
  const draft = '  마지막 초안\n\t들여쓰기와 한글 🧪\n  ';
  await page.goto('./');
  await ask(page, query);
  const input = page.getByRole('textbox', { name: '질문', exact: true });
  await input.fill('첫 번째 미전송 초안');
  await input.fill('두 번째 미전송 초안');
  await input.fill(draft);
  // Do not wait for the deferred write: navigation must flush the latest input.
  await page.reload({ waitUntil: 'domcontentloaded' });
  await expect(input).toHaveValue(draft);
  await expect(page.getByRole('region', { name: '대화 기록', exact: true }).getByText(query, { exact: true })).toBeVisible();
  await expect(page.getByText(fixtureAnswer, { exact: true })).toBeVisible();
});

test('deleting old and last conversations persists the remaining records and creates an empty conversation', async ({ page }) => {
  const oldTitle = '삭제할 오래된 대화';
  const keptTitle = '계속 보존할 대화';
  const keptDraft = '  남은 대화의 초안\n';
  await page.goto('./');
  await ask(page, oldTitle);
  await page.getByRole('button', { name: '새 채팅', exact: true }).click();
  await ask(page, keptTitle);
  await page.getByRole('textbox', { name: '질문', exact: true }).fill(keptDraft);
  await deleteConversation(page, oldTitle);
  await page.reload();
  await expect(page.getByRole('button', { name: `대화 선택: ${oldTitle}`, exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: `대화 선택: ${keptTitle}`, exact: true })).toBeVisible();
  await expect(page.getByRole('textbox', { name: '질문', exact: true })).toHaveValue(keptDraft);
  await expect(page.getByText(fixtureAnswer, { exact: true })).toBeVisible();

  await deleteConversation(page, keptTitle);
  await page.reload();
  await expect(page.getByRole('heading', { name: '문서와 자료에 질문하세요', exact: true })).toBeVisible();
  await expect(page.getByRole('textbox', { name: '질문', exact: true })).toHaveValue('');
  await expect(page.getByRole('region', { name: '대화 기록', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: '대화 선택: 새 대화', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: `대화 선택: ${keptTitle}`, exact: true })).toHaveCount(0);
});

test('an actual storage quota failure recovers after deleting an older conversation', async ({ page }) => {
  const oldTitle = '용량을 회수할 대화';
  const keptTitle = '용량 회복 뒤 보존할 대화';
  const draft = `보존할 최신 초안\n${'x'.repeat(65_536)}`;
  await page.goto('./');
  await ask(page, oldTitle);
  await page.getByRole('button', { name: '새 채팅', exact: true }).click();
  await ask(page, keptTitle);
  await expect.poll(async () => page.evaluate((key) => {
    const snapshot = sessionStorage.getItem(key);
    return snapshot === null ? 0 : (JSON.parse(snapshot) as StoredConversations).conversations
      .filter((conversation) => conversation.turns.at(-1)?.result?.status === 'completed').length;
  }, storageKey)).toBe(2);

  // Seed only the older, inactive draft so its deletion can reclaim real space.
  // The rest of the snapshot comes from the application's actual API workflow.
  await page.evaluate(({ key, title }) => {
    const snapshot = JSON.parse(sessionStorage.getItem(key)!) as StoredConversations;
    const old = snapshot.conversations.find((conversation) => conversation.turns[0]?.query === title);
    if (!old) throw new Error('The older conversation was not persisted');
    old.draft = 'old-record-'.repeat(32_000);
    sessionStorage.setItem(key, JSON.stringify(snapshot));
  }, { key: storageKey, title: oldTitle });
  await page.reload();
  await expect(page.getByText('질문할 준비가 되었습니다.', { exact: true })).toHaveCount(1);
  await expect.poll(async () => page.evaluate(({ key, title }) => {
    const snapshot = JSON.parse(sessionStorage.getItem(key)!) as StoredConversations;
    return snapshot.conversations.find((conversation) => conversation.turns[0]?.query === title)?.draft.length;
  }, { key: storageKey, title: oldTitle })).toBe(352_000);

  const quota = await page.evaluate(() => {
    const key = 'documate-test.quota-filler';
    let lower = 0;
    let upper = 5_500_000;
    let exhausted = false;
    while (lower < upper) {
      const middle = Math.ceil((lower + upper) / 2);
      try {
        sessionStorage.setItem(key, 'x'.repeat(middle));
        lower = middle;
      } catch (error) {
        if (!(error instanceof DOMException) || error.name !== 'QuotaExceededError') throw error;
        exhausted = true;
        upper = middle - 1;
      }
    }
    sessionStorage.setItem(key, 'x'.repeat(Math.max(0, lower - 4_096)));
    return { exhausted, filled: lower - 4_096 };
  });
  expect(quota.exhausted).toBe(true);
  expect(quota.filled).toBeGreaterThan(0);
  const input = page.getByRole('textbox', { name: '질문', exact: true });
  await input.fill(draft);
  const storageWarning = page.getByRole('alert').filter({ hasText: '오래된 대화를 삭제하면 저장 공간을 확보할 수 있습니다.' });
  await expect(storageWarning).toBeVisible();
  await deleteConversation(page, oldTitle);
  await expect(storageWarning).toHaveCount(0);
  await expect(input).toHaveValue(draft);
  await page.reload();
  await expect(input).toHaveValue(draft);
  await expect(page.getByRole('button', { name: `대화 선택: ${oldTitle}`, exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: `대화 선택: ${keptTitle}`, exact: true })).toBeVisible();
  await expect(page.getByText(fixtureAnswer, { exact: true })).toBeVisible();
  await expect(storageWarning).toHaveCount(0);
});
