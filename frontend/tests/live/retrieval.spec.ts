import { expect, test, type Page } from '@playwright/test';
import type { AnswerResponse } from '../../src/api/types';
import { answerText, answerUnitTexts, ask, expectCompleted, ready, saveAnswer } from './helpers';

const invoiceSource = [
  'def discounted_total(price, count):',
  '    subtotal = price * count',
  '    return round(subtotal * 0.9, 2)',
  '',
  'example = discounted_total(100, 3)',
  '',
].join('\n');

test('official sorted reverse guidance keeps two sentences, source evidence, and the saved answer after reload', async ({ page }) => {
  await ready(page);
  const result = await ask(page, 'Python 공식 문서를 근거로 sorted()의 reverse=True가 어떤 효과인지 한국어 두 문장으로 설명해줘. 코드 예시는 넣지 마.');
  expectCompleted(result);
  const response = result.response!;
  const body = answerText(result);
  const sentences = [...new Intl.Segmenter('ko', { granularity: 'sentence' }).segment(body)]
    .filter((item) => item.segment.trim());
  expect(sentences).toHaveLength(2);
  expect(response.content.blocks.some((block) => block.type === 'code')).toBe(false);
  await expect(page.locator('.answer-body pre')).toHaveCount(0);
  expectSemanticPass(response);
  expect(response.citations.length).toBeGreaterThan(0);
  const captured = Object.values(response.sources).map((source) => source.element.text).join('\n');
  expect(captured).toContain('sorted');
  expect(captured).toContain('reverse');
  await inspectSources(page, response, 'official');
  const displayed = await page.locator('.answer-body').textContent();

  const saved = await saveAnswer(page);
  for (const text of answerUnitTexts(response)) {
    expect(saved.bytes.toString('utf8')).toContain(text);
  }
  await page.reload();
  await expect(page.locator('.answer-body')).toBeVisible();
  expect(await page.locator('.answer-body').textContent()).toBe(displayed);
  await expect(page.getByRole('link', { name: 'TXT 다운로드', exact: true })).toHaveJSProperty('href', saved.href);
  await inspectSources(page, response, 'official');
});

test('the uploaded invoice function supports the ten percent discount and the independently calculated total', async ({ page }) => {
  await ready(page);
  const attachmentDetails = page.locator('details.sidebar-attachments');
  if (!await attachmentDetails.evaluate((element) => (element as HTMLDetailsElement).open)) {
    await attachmentDetails.getByText('첨부 자료', { exact: true }).click();
  }
  await page.getByLabel('파일 첨부', { exact: true }).setInputFiles({
    name: 'invoice.py', mimeType: 'text/x-python', buffer: Buffer.from(invoiceSource, 'utf8'),
  });
  await page.getByRole('button', { name: '첨부만 확정', exact: true }).click();
  await expect(page.getByRole('button', { name: 'invoice.py 첨부 해제', exact: true })).toBeVisible();
  const result = await ask(page, '첨부한 invoice.py의 discounted_total 함수가 적용하는 할인율과 example의 계산 결과를 알려줘. 첨부 코드만 근거로 짧게 설명해줘.');
  expectCompleted(result);
  const response = result.response!;
  const body = answerText(result);
  // These numeric expectations come from the supplied example: 100 × 3 × 0.9.
  expect(body).toMatch(/(?:^|\D)270(?:\.0+)?(?:\D|$)/);
  expect(body).toMatch(/10(?:\.0+)?\s*(?:%|퍼센트)|(?:^|\D)0\.10?(?:\D|$)/);
  expectSemanticPass(response);
  expect(response.citations.length).toBeGreaterThan(0);
  for (const source of Object.values(response.sources)) {
    expect(source.snapshot.source_type).toBe('upload');
    expect(source.element.text).toBe(invoiceSource);
  }
  await inspectSources(page, response, 'upload');
  const displayed = await page.locator('.answer-body').textContent();

  const saved = await saveAnswer(page);
  expect(saved.bytes.toString('utf8')).toContain('270');
  expect(saved.bytes.toString('utf8')).toContain('invoice.py');
  await page.reload();
  await expect(page.locator('.answer-body')).toBeVisible();
  expect(await page.locator('.answer-body').textContent()).toBe(displayed);
  await expect(page.getByRole('link', { name: 'TXT 다운로드', exact: true })).toHaveJSProperty('href', saved.href);
  await inspectSources(page, response, 'upload');
});

function expectSemanticPass(response: AnswerResponse): void {
  const validation = response.semantic_validation;
  expect(validation).not.toBeNull();
  expect(validation!.content_hash).toBe(response.content_hash);
  const requestCheck = validation!.checks.find((check) => check.name === 'request_fulfilled');
  expect(requestCheck?.status).toBe('evaluated');
  expect(requestCheck?.probability).toBeGreaterThanOrEqual(validation!.threshold);
  const grounded = validation!.checks.filter((check) => check.name === 'grounded');
  expect(grounded.length).toBeGreaterThan(0);
  for (const check of grounded) {
    expect(check.status).toBe('evaluated');
    expect(check.probability).toBeGreaterThanOrEqual(validation!.threshold);
  }
}

async function inspectSources(page: Page, response: AnswerResponse, sourceType: 'official' | 'upload'): Promise<void> {
  for (const citation of response.citations) {
    const source = response.sources[citation.source_id];
    expect(source.snapshot.source_type).toBe(sourceType);
    const title = source.snapshot.title || source.snapshot.source_uri;
    await page.getByRole('button', { name: `근거 [${citation.number}] ${title}`, exact: true }).click();
    const panel = page.getByRole('dialog', { name: title, exact: true });
    await expect(panel.getByRole('group', { name: '출처 요약', exact: true }))
      .toContainText(sourceType === 'official' ? '공식 문서' : '업로드 자료');
    if (sourceType === 'official') {
      expect(new URL(source.snapshot.source_uri).hostname).toBe('docs.python.org');
      await expect(panel.getByRole('link', { name: '현재 출처 페이지 열기', exact: true }))
        .toHaveAttribute('href', source.snapshot.source_uri);
    }
    await panel.getByText(sourceType === 'official' ? '수집한 원문 그대로 보기' : '전체 원문 보기', { exact: true }).click();
    expect(await panel.locator('.evidence-raw-source code').textContent()).toBe(source.element.text);
    await panel.getByRole('button', { name: '근거 닫기', exact: true }).click();
  }
}
