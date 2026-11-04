import { expect, test, type Page } from '@playwright/test';
import type { components } from '../../src/api/generated';
import type { AgentResponse, AnswerResponse } from '../../src/api/types';
import { answerText, ask, expectCompleted, ready } from './helpers';

const branchSource = [
  'import numpy as np',
  '',
  'if use_service:',
  '    np = service',
  'else:',
  '    result = np.array([1, 2])',
  '',
].join('\n');

const captureSource = [
  'import numpy as np',
  '',
  'match service:',
  '    case {"client": np}:',
  '        result = np.array([3, 4])',
  '',
].join('\n');

const notebookCells = [
  'import numpy as np\n',
  'if use_service:\n    np = service\n',
  'result = np.array([7, 8])\n',
];
const notebookSource = JSON.stringify({
  nbformat: 4,
  nbformat_minor: 5,
  metadata: {},
  cells: notebookCells.map((source, index) => ({
    id: `cell-${index}`, cell_type: 'code', metadata: {}, execution_count: null, outputs: [], source,
  })),
});

// Keep the originally observed source and question pairs: changing the requested
// symbols would bypass the distinction between canonical and raw name lookup.
test('the else branch preserves the actual NumPy call and its original sixth line', async ({ page }) => {
  await uploadSource(page, 'branch.py', 'text/x-python', branchSource);
  const result = await ask(page, '첨부한 branch.py에서 실제 NumPy의 numpy.array를 호출하는 문장만 원문 그대로 인용하고 줄 번호를 알려줘. 다른 객체의 array 호출은 포함하지 마.');
  const response = expectVerifiedAnswer(result);
  expect(answerText(result)).toContain('result = np.array([1, 2])');
  expect(result.debug!.retrieval_diagnostics.some((item) => item.answerability === 'covered')).toBe(true);
  expectBindingContext(result, 'result = np.array([1, 2])', ['import numpy as np', 'if use_service:']);
  const citation = citationContaining(response, branchSource, '    result = np.array([1, 2])');
  await inspectCallSource(page, response, citation, {
    title: 'branch.py', source: branchSource, line: 6, code: '    result = np.array([1, 2])',
  });
});

test('a match capture retains its binding context instead of treating the raw np.array name as NumPy', async ({ page }) => {
  await uploadSource(page, 'capture.py', 'text/x-python', captureSource);
  const result = await ask(page, '첨부 capture.py의 마지막 np.array 호출을 NumPy API 사용 사례로 볼 수 있어? match 문이 np에 어떤 영향을 주는지 코드 원문을 근거로 설명해줘.');
  const response = expectVerifiedAnswer(result);
  expectUnverifiedBinding(result);
  expectBindingContext(result, 'result = np.array([3, 4])', ['case {"client": np}:']);
  const citation = citationContaining(response, captureSource, '        result = np.array([3, 4])');
  expect(response.citations.some((item) => {
    const source = response.sources[item.source_id];
    return selectedText(source.element.text, item.selection).includes('case {"client": np}:');
  })).toBe(true);
  await inspectCallSource(page, response, citation, {
    title: 'capture.py', source: captureSource, line: 5, code: '        result = np.array([3, 4])',
  });
});

test('a conditional binding in an earlier notebook cell stays uncertain and locates the call in cell three', async ({ page }) => {
  await uploadSource(page, 'branch.ipynb', 'application/x-ipynb+json', notebookSource);
  const result = await ask(page, '이 노트북에서 numpy.array에 해당할 수 있는 호출을 찾아줘. 앞 셀에서 np를 바꿀 수 있다면 확정하지 말고, 호출이 있는 셀과 원문을 보여줘.');
  const response = expectVerifiedAnswer(result);
  expect(answerText(result)).toContain('result = np.array([7, 8])');
  expectUnverifiedBinding(result);
  expectBindingContext(result, 'result = np.array([7, 8])', ['import numpy as np', 'if use_service:\n    np = service']);
  const citation = citationContaining(response, notebookCells[2], 'result = np.array([7, 8])');
  expect(response.sources[citation.source_id].element.anchors).toEqual(expect.arrayContaining([
    expect.objectContaining({ kind: 'notebook', cell_index: 2, cell_id: 'cell-2', line_start: 1 }),
  ]));
  expect(response.citations.some((item) => {
    const source = response.sources[item.source_id];
    return source.element.text === notebookCells[1]
      && selectedText(source.element.text, item.selection).includes('np = service');
  })).toBe(true);
  await inspectCallSource(page, response, citation, {
    title: 'branch.ipynb', source: notebookCells[2], line: 1, cell: 3, code: 'result = np.array([7, 8])',
  });
});

async function uploadSource(page: Page, name: string, mimeType: string, source: string): Promise<void> {
  await ready(page);
  const attachments = page.locator('details.sidebar-attachments');
  if (!await attachments.evaluate((element) => (element as HTMLDetailsElement).open)) {
    await attachments.getByText('첨부 자료', { exact: true }).click();
  }
  await page.getByLabel('파일 첨부', { exact: true }).setInputFiles({
    name, mimeType, buffer: Buffer.from(source, 'utf8'),
  });
  await page.getByRole('button', { name: '첨부만 확정', exact: true }).click();
  await expect(page.getByRole('button', { name: `${name} 첨부 해제`, exact: true })).toBeVisible();
}

function expectVerifiedAnswer(result: AgentResponse): AnswerResponse {
  expectCompleted(result);
  const response = result.response!;
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
  expect(result.debug).not.toBeNull();
  return response;
}

function expectUnverifiedBinding(result: AgentResponse): void {
  expect(result.debug!.retrieval_diagnostics.some((item) => item.answerability === 'unknown'
    && item.warnings.includes('source_symbol_binding_unverified'))).toBe(true);
}

function expectBindingContext(result: AgentResponse, call: string, context: string[]): void {
  const hits = result.debug!.observed_hits;
  expect(hits.some((hit) => {
    if (hit.role !== 'candidate' || !selectedText(hit.evidence.element.text, hit.evidence.selection).includes(call)) return false;
    const supporting = hits.filter((item) => hit.supporting_evidence_ids.includes(item.evidence.id));
    const excerpts = [hit, ...supporting].map((item) => selectedText(item.evidence.element.text, item.evidence.selection));
    return context.every((snippet) => excerpts.some((excerpt) => excerpt.includes(snippet)));
  })).toBe(true);
}

function selectedText(text: string, selection: components['schemas']['SourceSelection']): string {
  return Array.from(text).slice(selection.start ?? 0, selection.end ?? undefined).join('');
}

function citationContaining(response: AnswerResponse, sourceText: string, call: string): AnswerResponse['citations'][number] {
  const citation = response.citations.find((item) => {
    const source = response.sources[item.source_id];
    return source.snapshot.source_type === 'upload' && source.element.text === sourceText
      && selectedText(source.element.text, item.selection).includes(call);
  });
  expect(citation, 'The cited original range must include the requested call.').toBeDefined();
  return citation!;
}

async function inspectCallSource(page: Page, response: AnswerResponse, citation: AnswerResponse['citations'][number], expected: {
  title: string; source: string; line: number; cell?: number; code: string;
}): Promise<void> {
  expect(response.sources[citation.source_id].snapshot.title).toBe(expected.title);
  await page.getByRole('button', { name: `근거 [${citation.number}] ${expected.title}`, exact: true }).click();
  const panel = page.getByRole('dialog', { name: expected.title, exact: true });
  const summary = panel.getByRole('group', { name: '출처 요약', exact: true });
  await expect(summary).toContainText('업로드 자료');
  if (expected.cell !== undefined) await expect(summary).toContainText(`셀 ${expected.cell}의`);
  expect(await panel.getByRole('group', { name: `원문 ${expected.line}행`, exact: true }).locator('code').textContent())
    .toBe(expected.code);
  await panel.getByText('전체 원문 보기', { exact: true }).click();
  expect(await panel.locator('.evidence-raw-source code').textContent()).toBe(expected.source);
  await panel.getByRole('button', { name: '근거 닫기', exact: true }).click();
}
