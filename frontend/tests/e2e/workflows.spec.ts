import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';

const fixtureUrl = 'http://127.0.0.1:18000';
const answer = '실제 FastAPI와 Python 런타임을 거친 한글 답변입니다. 🧪';

async function ask(page: Page, query: string) {
  await page.getByRole('textbox', { name: '질문', exact: true }).fill(query);
  const submit = page.getByRole('button', { name: '질문 보내기', exact: true });
  await expect(submit).toBeEnabled();
  await submit.click();
}

async function received(request: APIRequestContext) {
  return await (await request.get(`${fixtureUrl}/_fixture/requests`)).json() as {
    queries: string[];
    submissions: { query: string; session_id: string; slack_recipient: unknown }[];
    slack: { path: string; payload: { text?: string; channel?: string } }[];
  };
}

test('files keep their original evidence after replacement and download the selected answer', async ({ page, request }) => {
  await page.goto('./');
  await page.getByLabel('파일 첨부', { exact: true }).setInputFiles({
    name: 'alpha.py', mimeType: 'text/x-python', buffer: Buffer.from('alpha = 1\r\n'),
  });
  await page.getByRole('button', { name: '첨부만 확정', exact: true }).click();
  await expect(page.getByRole('button', { name: 'alpha.py 첨부 해제', exact: true })).toBeVisible();

  const query = '첨부 alpha 첫 번째 코드 확인';
  await ask(page, query);
  await expect(page.getByText('alpha = 1', { exact: true }).first()).toBeVisible();
  const originalSource = page.getByRole('button', { name: '근거 [1] alpha.py', exact: true }).first();
  await originalSource.click();
  const panel = page.getByRole('dialog', { name: 'alpha.py', exact: true });
  await expect(panel.getByText('alpha = 1', { exact: true }).first()).toBeVisible();
  await panel.getByRole('button', { name: '근거 닫기', exact: true }).click();
  await expect(originalSource).toBeFocused();

  const initialSave = page.waitForResponse((response) => response.request().method() === 'POST' && response.url().endsWith('/save'));
  await page.getByRole('button', { name: 'TXT 파일 만들기', exact: true }).click();
  const originalSaveUrl = (await initialSave).url();
  const downloadLink = page.getByRole('link', { name: 'TXT 다운로드', exact: true });
  await expect(downloadLink).toBeVisible();
  const url = new URL((await downloadLink.getAttribute('href')) ?? '', page.url()).href;
  const downloadPromise = page.waitForEvent('download');
  await downloadLink.click();
  const download = await downloadPromise;
  const path = await download.path();
  expect(path).not.toBeNull();
  const bytes = await readFile(path as string);
  expect(bytes).toEqual(await (await request.get(url)).body());
  expect(bytes.subarray(0, 3)).toEqual(Buffer.from([0xef, 0xbb, 0xbf]));
  expect(bytes.toString('utf8')).toContain('alpha = 1');
  expect(bytes.toString('utf8')).toContain('alpha.py');

  await page.getByLabel('파일 첨부', { exact: true }).setInputFiles({
    name: 'alpha.py', mimeType: 'text/x-python', buffer: Buffer.from('alpha = 2\n'),
  });
  await page.getByRole('button', { name: '첨부만 확정', exact: true }).click();
  await page.getByRole('checkbox', { name: 'alpha.py 교체 승인', exact: true }).check();
  await page.getByRole('button', { name: '첨부 변경 확정', exact: true }).click();
  await expect(page.getByRole('checkbox', { name: 'alpha.py 교체 승인', exact: true })).toHaveCount(0);
  await originalSource.click();
  await expect(panel.getByText('alpha = 1', { exact: true }).first()).toBeVisible();
  await expect(panel.getByText('alpha = 2', { exact: true })).toHaveCount(0);
  await panel.getByRole('button', { name: '근거 닫기', exact: true }).click();
  await ask(page, '첨부 alpha 교체된 코드 확인');
  await expect(page.getByText('alpha = 2', { exact: true }).first()).toBeVisible();
  const repeatedSave = page.waitForResponse((response) => response.request().method() === 'POST' && response.url().endsWith('/save'));
  await page.getByRole('button', { name: '저장 파일 다시 확인', exact: true }).click();
  const saveResponse = await repeatedSave;
  expect(saveResponse.status()).toBe(200);
  expect(saveResponse.url()).toBe(originalSaveUrl);
  await expect(downloadLink).toHaveAttribute('href', new URL(url).pathname);
  expect(await (await request.get(url)).body()).toEqual(bytes);

  const requests = await received(request);
  expect(requests.queries.filter((value) => value === query)).toEqual([query]);
  expect(requests.queries.filter((value) => value === '첨부 alpha 교체된 코드 확인')).toHaveLength(1);
  await page.getByRole('button', { name: 'alpha.py 첨부 해제', exact: true }).click();
  await expect(page.getByRole('button', { name: 'alpha.py 첨부 해제', exact: true })).toHaveCount(0);
  await originalSource.click();
  await expect(panel.getByText('alpha = 1', { exact: true }).first()).toBeVisible();
});

test('an interrupted attachment retries explicitly and does not replay its held question', async ({ page, request }) => {
  let firstUpload = true;
  await page.route('**/uploads/staging', async (route) => {
    if (firstUpload) {
      firstUpload = false;
      await route.abort('failed');
    } else {
      await route.continue();
    }
  });
  await page.goto('./');
  await page.getByLabel('파일 첨부', { exact: true }).setInputFiles({
    name: 'beta.py', mimeType: 'text/x-python', buffer: Buffer.from('beta = 3\n'),
  });
  const query = '첨부 재시도 후 명시적으로 질문';
  await ask(page, query);
  await expect(page.getByRole('button', { name: '첨부 다시 시도', exact: true })).toBeVisible();
  expect((await received(request)).queries).not.toContain(query);
  await page.getByRole('button', { name: '첨부 다시 시도', exact: true }).click();
  const resume = page.getByRole('button', { name: '보류한 질문 보내기', exact: true });
  await expect(resume).toBeVisible();
  expect((await received(request)).queries).not.toContain(query);
  await resume.click();
  await expect(page.getByText('beta = 3', { exact: true }).first()).toBeVisible();
  expect((await received(request)).queries.filter((value) => value === query)).toEqual([query]);
});

test('conversations preserve distinct drafts and restore without replaying completed questions', async ({ page, request }) => {
  await page.goto('./');
  const first = '첫 대화 복원 검증';
  await ask(page, first);
  await expect(page.getByText(answer, { exact: true })).toBeVisible();
  const draft = '  첫 대화의 미전송 초안\n둘째 줄  ';
  await page.getByRole('textbox', { name: '질문', exact: true }).fill(draft);
  await page.getByRole('button', { name: '새 채팅', exact: true }).click();
  const second = '두 번째 대화 검증';
  await ask(page, second);
  await expect(page.getByText(answer, { exact: true })).toBeVisible();
  await page.getByRole('textbox', { name: '질문', exact: true }).fill('다른 대화의 초안');
  await page.getByRole('button', { name: `대화 선택: ${first}`, exact: true }).click();
  await expect(page.getByRole('textbox', { name: '질문', exact: true })).toHaveValue(draft);
  await page.reload();
  await expect(page.getByText(answer, { exact: true })).toBeVisible();
  await expect(page.getByRole('textbox', { name: '질문', exact: true })).toHaveValue(draft);
  await page.getByRole('button', { name: `대화 선택: ${second}`, exact: true }).click();
  await expect(page.getByRole('textbox', { name: '질문', exact: true })).toHaveValue('다른 대화의 초안');
  const records = await received(request);
  expect(records.queries.filter((value) => value === first)).toEqual([first]);
  expect(records.queries.filter((value) => value === second)).toEqual([second]);
  expect(records.queries).not.toContain(draft);
});

test('reload during a held response preserves the submitted question and next draft without replay', async ({ page, request }) => {
  const query = '새로고침 중단 원문 보존 검증';
  await request.post(`${fixtureUrl}/_fixture/hold`, { data: { query } });
  await page.goto('./');
  await ask(page, query);
  await expect(page.getByRole('button', { name: '응답 수신 중단', exact: true })).toBeVisible();
  await expect.poll(async () => (await received(request)).queries.filter((value) => value === query).length).toBe(1);
  await page.getByRole('textbox', { name: '질문', exact: true }).fill('  다음에 보낼 초안\n');
  await page.reload({ waitUntil: 'domcontentloaded' });
  await request.post(`${fixtureUrl}/_fixture/release`, { data: { query } });
  await expect(page.getByRole('textbox', { name: '질문', exact: true })).toHaveValue('  다음에 보낼 초안\n');
  await expect(page.getByRole('region', { name: '대화 기록', exact: true }).getByText(query, { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: '질문 보내기', exact: true })).toBeEnabled();
  expect((await received(request)).queries.filter((value) => value === query)).toEqual([query]);
});

test('a changed server session epoch retains its answer and draft as read-only records', async ({ page, request }) => {
  const query = '만료될 대화 검증';
  await page.goto('./');
  await ask(page, query);
  await expect(page.getByText(answer, { exact: true })).toBeVisible();
  await page.getByRole('textbox', { name: '질문', exact: true }).fill('만료 후 보존할 초안');
  const submission = (await received(request)).submissions.find((item) => item.query === query);
  expect(submission).toBeDefined();
  await request.post(`${fixtureUrl}/_fixture/sessions/${submission?.session_id}/expire`);
  await page.reload();
  await expect(page.getByText(answer, { exact: true })).toBeVisible();
  await expect(page.getByText('서버 세션이 만료되어 읽기 전용입니다. 새 채팅에서 계속할 수 있습니다.', { exact: true })).toBeVisible();
  await expect(page.getByRole('textbox', { name: '질문', exact: true })).toHaveValue('만료 후 보존할 초안');
  await expect(page.getByRole('textbox', { name: '질문', exact: true })).toHaveAttribute('readonly');
  await expect(page.getByRole('button', { name: '질문 보내기', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'TXT 파일 만들기', exact: true })).toBeDisabled();
  expect((await received(request)).queries.filter((value) => value === query)).toEqual([query]);
});

test('notebook cell numbers and CRLF physical lines survive the real upload and evidence pipeline', async ({ page }) => {
  await page.goto('./');
  const notebook = {
    nbformat: 4, nbformat_minor: 5, metadata: {}, cells: [
      { cell_type: 'markdown', id: 'empty-introduction', metadata: {}, source: [] },
      { cell_type: 'code', id: 'alpha-cell', metadata: {}, execution_count: null, outputs: [], source: ['alpha = 1\r\n', 'beta = alpha + 2\r\n'] },
    ],
  };
  await page.getByLabel('파일 첨부', { exact: true }).setInputFiles({
    name: 'physical.ipynb', mimeType: 'application/x-ipynb+json', buffer: Buffer.from(JSON.stringify(notebook)),
  });
  await page.getByRole('button', { name: '첨부만 확정', exact: true }).click();
  await ask(page, '노트북 원문 셀과 줄 확인');
  await page.getByRole('button', { name: '근거 [1] physical.ipynb', exact: true }).click();
  const panel = page.getByRole('dialog', { name: 'physical.ipynb', exact: true });
  await expect(panel.getByRole('group', { name: '원문 1행', exact: true })).toContainText('alpha = 1');
  await expect(panel.getByRole('group', { name: '원문 2행', exact: true })).toContainText('beta = alpha + 2');
  await expect(panel.getByText(/셀 2의/)).toBeVisible();
  await panel.getByText('출처와 수집 정보', { exact: true }).click();
  await expect(panel.getByText(/셀 ID alpha-cell/)).toBeVisible();
});

test('Slack settings preserve invalid input for normal chat and send only on an explicit request', async ({ page, request }) => {
  const initialSends = (await received(request)).slack.filter((item) => item.path === '/chat.postMessage').length;
  await page.goto('./');
  await page.getByText('Slack 수신자 설정', { exact: true }).click();
  await expect(page.getByRole('status').filter({ hasText: '기본 수신자:' })).toContainText('UDEFAULT');
  await page.getByLabel('수신자 유형', { exact: true }).selectOption('channel');
  await page.getByLabel('Slack 수신자', { exact: true }).fill('invalid-channel');
  const ordinary = '잘못된 수신자를 입력한 일반 질문';
  await ask(page, ordinary);
  await expect(page.getByText(answer, { exact: true })).toBeVisible();
  const before = await received(request);
  expect(before.submissions.find((item) => item.query === ordinary)?.slack_recipient).toEqual({ kind: 'channel', value: 'invalid-channel' });
  const sendsBefore = before.slack.filter((item) => item.path === '/chat.postMessage').length;
  expect(sendsBefore).toBe(initialSends);
  await page.getByLabel('Slack 수신자', { exact: true }).fill('CINTENDED');
  const query = '확인 결과를 Slack으로 보내줘';
  await ask(page, query);
  await expect.poll(async () => (await received(request)).slack.filter((item) => item.path === '/chat.postMessage').length).toBe(sendsBefore + 1);
  const records = await received(request);
  const sent = records.slack.filter((item) => item.path === '/chat.postMessage').at(-1);
  expect(sent?.payload.channel).toBe('CINTENDED');
  expect(sent?.payload.text).toContain(answer);
  expect(records.queries.filter((value) => value === query)).toEqual([query]);
  await expect(page.getByRole('region', { name: '전달 결과', exact: true })).toContainText('전송 대상: CINTENDED · 메시지 식별자: 1.0');
});
