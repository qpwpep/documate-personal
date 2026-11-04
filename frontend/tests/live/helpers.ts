import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { expect, test, type Page } from '@playwright/test';
import type { AgentResponse, AnswerExportResult, AnswerResponse } from '../../src/api/types';

type StoredResponse = Omit<AgentResponse, 'response'> & {
  response: (Omit<AnswerResponse, 'sources'> & { sourceIds: string[] }) | null;
};

export async function ready(page: Page): Promise<void> {
  await page.route('**/agent/stream', async (route) => {
    if (route.request().method() !== 'POST') { await route.continue(); return; }
    const payload = route.request().postDataJSON() as Record<string, unknown>;
    await route.continue({ postData: JSON.stringify({ ...payload, include_debug: true }) });
  });
  await page.goto('./');
  await expect(page.getByText('질문할 준비가 되었습니다.', { exact: true })).toHaveCount(1);
}

export async function ask(page: Page, query: string): Promise<AgentResponse> {
  await page.getByRole('textbox', { name: '질문', exact: true }).fill(query);
  const submit = page.getByRole('button', { name: '질문 보내기', exact: true });
  await expect(submit).toBeEnabled();
  const streamed = page.waitForResponse((response) => response.url().endsWith('/agent/stream')
    && response.request().method() === 'POST', { timeout: 180_000 });
  await submit.click();
  const transport = await streamed;
  expect(transport.status()).toBe(200);
  const executionId = transport.headers()['x-execution-id'];
  expect(executionId).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  // The UI closes SSE after its validated final response. Read that actual
  // response from the UI snapshot, restoring its shared sources without
  // changing the body, status, diagnostics, or transport result.
  const captured = await page.waitForFunction(({ executionId }) => {
    const saved = sessionStorage.getItem('documate.react.conversations');
    if (saved === null) return null;
    const snapshot = JSON.parse(saved) as {
      sources: AnswerResponse['sources']; conversations: { turns: { result: StoredResponse | null }[] }[];
    };
    const result = snapshot.conversations.flatMap((conversation) => conversation.turns)
      .find((turn) => turn.result?.execution_id === executionId)?.result;
    if (!result) return null;
    if (result.response === null) return result;
    const { sourceIds, ...answer } = result.response;
    const sources = Object.fromEntries(sourceIds.map((id) => [id, snapshot.sources[id]]));
    return { ...result, response: { ...answer, sources } };
  }, { executionId }, { timeout: 180_000 });
  const result = await captured.jsonValue() as AgentResponse;
  expect(result.execution_id).toBe(executionId);
  await test.info().attach(`turn-${executionId}`, {
    body: JSON.stringify({ query, result }, null, 2), contentType: 'application/json',
  });
  return result;
}

export function expectCompleted(result: AgentResponse): void {
  expect(result.status, result.problem?.message ?? result.message).toBe('completed');
  expect(result.problem).toBeNull();
  expect(result.response).not.toBeNull();
  expect(result.answer_id).not.toBeNull();
}

export function answerText(result: AgentResponse): string {
  if (result.response === null) throw new Error(`Expected an answer, got ${result.status}: ${result.message}`);
  return answerUnitTexts(result.response).join('\n');
}

export function answerUnitTexts(response: AnswerResponse): string[] {
  return response.content.blocks.flatMap((block) => {
    switch (block.type) {
      case 'paragraph': return block.content.map((unit) => unit.text);
      case 'heading': case 'code': return [block.content.text];
      case 'list': return block.items.map((unit) => unit.text);
      case 'table': return [block.columns, ...block.rows].flatMap((row) => row.map((unit) => unit.text));
    }
  });
}

export async function saveAnswer(page: Page, index = -1): Promise<{
  result: AnswerExportResult; bytes: Buffer; href: string;
}> {
  const turn = page.getByRole('region', { name: '대화 기록', exact: true }).nth(index);
  const button = turn.locator('.answer-delivery').getByRole('button');
  await expect(button).toBeEnabled();
  const saved = page.waitForResponse((response) => response.request().method() === 'POST'
    && /\/answers\/[^/]+\/save$/.test(new URL(response.url()).pathname), { timeout: 180_000 });
  await button.click();
  const transport = await saved;
  expect(transport.status()).toBe(200);
  const result = await transport.json() as AnswerExportResult;
  expect(result.status, result.message).toBe('completed');
  expect(result.execution_id).toBe(transport.headers()['x-execution-id']);
  expect(result.receipt.status).toBe('success');
  expect(result.receipt.verification).toBe('verified');
  const artifact = result.receipt.artifact;
  expect(artifact).not.toBeNull();
  const link = turn.getByRole('link', { name: 'TXT 다운로드', exact: true });
  await expect(link).toBeVisible();
  const href = new URL((await link.getAttribute('href'))!, page.url()).href;
  const download = page.waitForEvent('download');
  await link.click();
  const path = await (await download).path();
  expect(path).not.toBeNull();
  const bytes = await readFile(path!);
  const fetched = await page.request.get(href);
  expect(fetched.status()).toBe(200);
  expect(bytes).toEqual(await fetched.body());
  expect(bytes.byteLength).toBe(artifact!.byte_count);
  expect(createHash('sha256').update(bytes).digest('hex')).toBe(artifact!.sha256);
  await test.info().attach(`save-${result.execution_id}`, {
    body: JSON.stringify(result, null, 2), contentType: 'application/json',
  });
  return { result, bytes, href };
}
