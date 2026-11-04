import { expect, test, type Page } from '@playwright/test';
import type { AgentResponse, AnswerResponse } from '../../src/api/types';
import { answerText, ask, expectCompleted, ready } from './helpers';

test('replacing an attachment changes the current answer while both captured sources survive reload', async ({ page }) => {
  await ready(page);
  const attachments = page.locator('details.sidebar-attachments');
  if (!await attachments.evaluate((element) => (element as HTMLDetailsElement).open)) {
    await attachments.getByText('첨부 자료', { exact: true }).click();
  }
  await page.getByLabel('파일 첨부', { exact: true }).setInputFiles({
    name: 'settings.py', mimeType: 'text/x-python', buffer: Buffer.from('RETRY_LIMIT = 3\n', 'utf8'),
  });
  await page.getByRole('button', { name: '첨부만 확정', exact: true }).click();
  await expect(page.getByRole('button', { name: 'settings.py 첨부 해제', exact: true })).toBeVisible();
  const original = expectSourceAnswer(
    await ask(page, '첨부한 settings.py의 RETRY_LIMIT 값은 얼마야? 첨부 파일만 근거로 답해줘.'),
    3, 'RETRY_LIMIT = 3\n',
  );
  await inspectCapturedSource(page, 0, original, 'RETRY_LIMIT = 3\n');

  await page.getByLabel('파일 첨부', { exact: true }).setInputFiles({
    name: 'settings.py', mimeType: 'text/x-python', buffer: Buffer.from('RETRY_LIMIT = 7\n', 'utf8'),
  });
  await page.getByRole('button', { name: '첨부만 확정', exact: true }).click();
  await page.getByRole('checkbox', { name: 'settings.py 교체 승인', exact: true }).check();
  await page.getByRole('button', { name: '첨부 변경 확정', exact: true }).click();
  await expect(page.getByRole('checkbox', { name: 'settings.py 교체 승인', exact: true })).toHaveCount(0);
  const replaced = expectSourceAnswer(
    await ask(page, '교체한 settings.py의 RETRY_LIMIT 값은 얼마야? 첨부 파일만 근거로 답해줘.'),
    7, 'RETRY_LIMIT = 7\n',
  );
  const originalSource = original.sources[original.citations[0].source_id];
  const replacedSource = replaced.sources[replaced.citations[0].source_id];
  expect(replacedSource.snapshot.snapshot_id).not.toBe(originalSource.snapshot.snapshot_id);
  expect(replacedSource.snapshot.content_hash).not.toBe(originalSource.snapshot.content_hash);
  expect(replaced.citations[0].source_id).not.toBe(original.citations[0].source_id);
  await inspectCapturedSource(page, 1, replaced, 'RETRY_LIMIT = 7\n');
  await inspectCapturedSource(page, 0, original, 'RETRY_LIMIT = 3\n');

  const turns = page.getByRole('region', { name: '대화 기록', exact: true });
  await expect(turns).toHaveCount(2);
  const displayed = await turns.locator('.answer-body').allTextContents();
  await page.reload();
  await expect(turns).toHaveCount(2);
  await expect(turns.locator('.answer-body')).toHaveText(displayed);
  await inspectCapturedSource(page, 0, original, 'RETRY_LIMIT = 3\n');
  await inspectCapturedSource(page, 1, replaced, 'RETRY_LIMIT = 7\n');
});

function expectSourceAnswer(result: AgentResponse, value: number, sourceText: string): AnswerResponse {
  expectCompleted(result);
  expect(answerText(result)).toMatch(new RegExp(`(?:^|\\D)${value}(?:\\D|$)`));
  const response = result.response!;
  expect(response.citations.length).toBeGreaterThan(0);
  for (const citation of response.citations) {
    const source = response.sources[citation.source_id];
    expect(source.snapshot.source_type).toBe('upload');
    expect(source.snapshot.title).toBe('settings.py');
    expect(source.element.text).toBe(sourceText);
    expect(Array.from(source.element.text).slice(citation.selection.start ?? 0, citation.selection.end ?? undefined).join(''))
      .toContain(sourceText.trimEnd());
  }
  return response;
}

async function inspectCapturedSource(page: Page, turnIndex: number, response: AnswerResponse, sourceText: string): Promise<void> {
  const turn = page.getByRole('region', { name: '대화 기록', exact: true }).nth(turnIndex);
  for (const citation of response.citations) {
    await turn.getByRole('button', { name: `근거 [${citation.number}] settings.py`, exact: true }).click();
    const panel = page.getByRole('dialog', { name: 'settings.py', exact: true });
    await expect(panel.getByRole('group', { name: '출처 요약', exact: true })).toContainText('업로드 자료');
    expect(await panel.getByRole('group', { name: '원문 1행', exact: true }).locator('code').textContent())
      .toBe(sourceText.trimEnd());
    await panel.getByText('전체 원문 보기', { exact: true }).click();
    expect(await panel.locator('.evidence-raw-source code').textContent()).toBe(sourceText);
    await panel.getByRole('button', { name: '근거 닫기', exact: true }).click();
  }
}
