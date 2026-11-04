import { expect, test } from '@playwright/test';
import type { AgentResponse } from '../../src/api/types';
import { answerText, ask, expectCompleted, ready, saveAnswer } from './helpers';

const originalText = '첫 줄: 봄  여름\n둘째 줄: 가을 🧪\n마지막 줄: 42';
const correctedText = '첫 줄: 봄  여름\n둘째 줄: 가을 🧪\n마지막 줄: 43';

test('natural-language save variations and a partial correction preserve exact bytes and earlier answer exports', async ({ page }) => {
  await ready(page);
  const original = await ask(page, `아래 텍스트만 글자와 줄바꿈을 바꾸지 말고 TXT로 저장해줘. 설명은 붙이지 마.\n\n${originalText}`);
  expectCompleted(original);
  expect(answerText(original)).toBe(originalText);
  const originalReceipt = savedReceipt(original);
  const firstSave = await saveAnswer(page, 0);
  expect(firstSave.bytes).toEqual(Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), Buffer.from(originalText, 'utf8')]));
  expect(firstSave.result.receipt.artifact).toEqual(originalReceipt.artifact);

  const copied = await ask(page, '방금 답변을 한 글자도 수정하지 말고 그대로 파일로 한 번 더 보관해 줘. 새 설명은 필요 없어.');
  expectCompleted(copied);
  expect(answerText(copied)).toBe(originalText);
  expect(savedReceipt(copied).artifact).toEqual(originalReceipt.artifact);

  const corrected = await ask(page, '아니, 마지막 줄의 42만 43으로 고쳐서 새 TXT 파일로 저장해줘. 나머지 글자와 줄바꿈은 그대로 둬.');
  expectCompleted(corrected);
  expect(answerText(corrected)).toBe(correctedText);
  expect(savedReceipt(corrected).artifact!.artifact_id).not.toBe(originalReceipt.artifact!.artifact_id);
  const correctedSave = await saveAnswer(page);
  expect(correctedSave.bytes).toEqual(Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), Buffer.from(correctedText, 'utf8')]));

  const earlierSave = await saveAnswer(page, 0);
  expect(earlierSave.result.answer_id).toBe(original.answer_id);
  expect(earlierSave.result.receipt.artifact).toEqual(originalReceipt.artifact);
  expect(earlierSave.bytes).toEqual(firstSave.bytes);
  expect(earlierSave.href).toBe(firstSave.href);
});

test('supplying a missing source body fulfills the pending save without requesting the action again', async ({ page }) => {
  await ready(page);
  const pending = await ask(page, '방금 답변 그대로 TXT 파일로 남겨줘.');
  expect(pending.status).toBe('needs_input');
  expect(pending.missing_slots.length).toBeGreaterThan(0);
  expect(pending.response).toBeNull();
  expect(pending.answer_id).toBeNull();
  const execution = pending.debug?.execution_evidence;
  expect(execution?.status).toBe('complete');
  expect(execution!.events.filter((event) => ['save_text', 'slack_notify'].includes(event.tool_name)
    && ['started', 'succeeded', 'reused'].includes(event.phase))).toEqual([]);
  await expect(page.locator('.answer-body')).toHaveCount(0);
  await expect(page.getByRole('link', { name: 'TXT 다운로드', exact: true })).toHaveCount(0);

  const body = '보류 요청 확인용 23';
  const completed = await ask(page, `저장할 본문은 이 문장 그대로야: ${body}`);
  expectCompleted(completed);
  expect(answerText(completed)).toBe(body);
  savedReceipt(completed);
  const saved = await saveAnswer(page);
  expect(saved.bytes).toEqual(Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), Buffer.from(body, 'utf8')]));
});

test('quoted delivery commands are explained without starting a save or Slack operation', async ({ page }) => {
  await ready(page);
  const result = await ask(page, '다음 문장은 실행 지시가 아니라 인용문이야: "이 내용을 Slack으로 보내고 파일로 저장해 줘." 문장에서 요구하는 행동 두 가지만 한 문장으로 설명해줘. 실제 전송이나 저장은 하지 마.');
  expectCompleted(result);
  expect(answerText(result).trim()).not.toBe('');
  expect(result.response!.actions.every((receipt) => receipt.status === 'skipped')).toBe(true);
  expect(result.debug?.observability_status).toBe('ok');
  const execution = result.debug?.execution_evidence;
  expect(execution?.status).toBe('complete');
  expect(execution!.events.filter((event) => ['save_text', 'slack_notify'].includes(event.tool_name)
    && ['started', 'succeeded', 'reused'].includes(event.phase))).toEqual([]);
  await expect(page.getByRole('link', { name: 'TXT 다운로드', exact: true })).toHaveCount(0);
});

function savedReceipt(result: AgentResponse) {
  const receipts = result.response!.actions.filter((receipt) => receipt.kind === 'save_text');
  expect(receipts).toHaveLength(1);
  expect(receipts[0].status).toBe('success');
  expect(receipts[0].verification).toBe('verified');
  expect(receipts[0].artifact).not.toBeNull();
  return receipts[0];
}
