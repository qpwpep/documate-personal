import { expect, test, type APIRequestContext, type Route } from '@playwright/test';
import { readFile } from 'node:fs/promises';

const fixtureUrl = 'http://127.0.0.1:18000';

async function submittedQueries(request: APIRequestContext): Promise<string[]> {
  const response = await request.get(`${fixtureUrl}/_fixture/requests`);
  const payload = await response.json() as { queries: string[] };
  return payload.queries;
}

test('confirmed browser uploads release staging and allow repeated near-limit replacements', async ({ page, request }) => {
  await page.goto('./');
  const config = await (await request.get(`${fixtureUrl}/client-config`)).json() as {
    upload: { max_file_bytes: number };
  };
  const revisions = 7;
  for (let revision = 0; revision < revisions; revision += 1) {
    const notebook = {
      nbformat: 4, nbformat_minor: 5, metadata: { padding: '' }, cells: [{
        cell_type: 'code', id: 'replacement-cell', metadata: {}, execution_count: null,
        outputs: [], source: [`value = ${revision}\n`],
      }],
    };
    notebook.metadata.padding = 'x'.repeat(config.upload.max_file_bytes - Buffer.byteLength(JSON.stringify(notebook)));
    const bytes = Buffer.from(JSON.stringify(notebook));
    expect(bytes.byteLength).toBe(config.upload.max_file_bytes);
    await page.getByLabel('파일 첨부', { exact: true }).setInputFiles({
      name: 'capacity.ipynb', mimeType: 'application/x-ipynb+json', buffer: bytes,
    });
    const sync = page.waitForResponse((response) => response.request().method() === 'POST'
      && response.url().endsWith('/uploads/sync'));
    await page.getByRole('button', { name: '첨부만 확정', exact: true }).click();
    if (revision > 0) {
      await page.getByRole('checkbox', { name: 'capacity.ipynb 교체 승인', exact: true }).check();
      await page.getByRole('button', { name: '첨부 변경 확정', exact: true }).click();
    }
    const response = await sync;
    expect(response.status()).toBe(200);
    const manifest = await response.json() as { revision: number; files: { name: string; size_bytes: number }[] };
    expect(manifest.revision).toBe(revision + 1);
    expect(manifest.files).toHaveLength(1);
    expect(manifest.files[0]).toMatchObject({ name: 'capacity.ipynb', size_bytes: bytes.byteLength });
    await expect(page.getByRole('button', { name: 'capacity.ipynb 첨부 해제', exact: true })).toBeEnabled();
    const sessionId = new URL(response.url()).pathname.split('/')[2];
    expect(await (await request.get(`${fixtureUrl}/_fixture/sessions/${sessionId}/staging`)).json())
      .toEqual({ files: 0, bytes: 0 });
    // A lost success response can be retried even after its input bytes are reclaimed.
    const retry = await request.post(response.url(), { data: response.request().postDataJSON() });
    expect(retry.status()).toBe(200);
    expect(await retry.json()).toEqual(manifest);
    if (revision === 0 || revision === revisions - 1) {
      await page.getByLabel('파일 첨부', { exact: true }).setInputFiles({
        name: 'capacity.ipynb', mimeType: 'application/x-ipynb+json', buffer: bytes,
      });
      const duplicateSync = page.waitForResponse((received) => received.request().method() === 'POST'
        && received.url().endsWith('/uploads/sync'));
      await page.getByRole('button', { name: '첨부만 확정', exact: true }).click();
      const duplicate = await duplicateSync;
      expect(duplicate.status()).toBe(200);
      expect(await duplicate.json()).toEqual(manifest);
      await expect(page.getByRole('button', { name: 'capacity.ipynb 첨부 해제', exact: true })).toBeEnabled();
      expect(await (await request.get(`${fixtureUrl}/_fixture/sessions/${sessionId}/staging`)).json())
        .toEqual({ files: 0, bytes: 0 });
    }
  }
});

test('stopping the preflight manifest check keeps the draft and reconnects without submitting', async ({ page, request }) => {
  const query = '  연결 확인 중 중단한 질문 원문\n둘째 줄  ';
  await page.goto('./');
  const input = page.getByRole('textbox', { name: '질문', exact: true });
  const submit = page.getByRole('button', { name: '질문 보내기', exact: true });
  await input.fill(query);
  await expect(submit).toBeEnabled();
  let intercept: (route: Route) => void = () => { throw new Error('The route resolver is not ready'); };
  const intercepted = new Promise<Route>((resolve) => { intercept = resolve; });
  await page.route(/\/sessions\/[^/]+\/uploads$/, (route) => { intercept(route); }, { times: 1 });
  await submit.click();
  const held = await intercepted;
  try {
    expect(await submittedQueries(request)).not.toContain(query);
    await page.getByRole('button', { name: '응답 수신 중단', exact: true }).click();
    await expect(page.getByRole('button', { name: '연결 다시 확인', exact: true })).toBeEnabled();
    await expect(input).toHaveValue(query);
    expect(await submittedQueries(request)).not.toContain(query);
  } finally {
    await held.abort();
  }
  await page.getByRole('button', { name: '연결 다시 확인', exact: true }).click();
  await expect(submit).toBeEnabled();
  await expect(input).toHaveValue(query);
  expect(await submittedQueries(request)).not.toContain(query);
  await submit.click();
  await expect(page.getByText('답변 받음', { exact: true })).toBeVisible();
  expect((await submittedQueries(request)).filter((item) => item === query)).toEqual([query]);
});

test('copying Markdown through the real input-copy path retains block structure and exact downloaded text', async ({ page, request }) => {
  const original = '# 복사한 Markdown 제목\n\n'
    + '3. 첫 번째 항목\n4. 두 번째 항목\n\n'
    + '| 이름 | 값 |\n| --- | --- |\n| alpha | 7 |\n\n'
    + '```python\nif True:\n    print("원문 코드")\n```\n';
  const query = `다음 Markdown 원문을 그대로 복사해줘.\n\n${original}`;
  const consoleErrors: string[] = [];
  page.on('pageerror', (error) => consoleErrors.push(error.message));
  await page.goto('./');
  await page.getByRole('textbox', { name: '질문', exact: true }).fill(query);
  await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
  await expect(page.getByText('답변 받음', { exact: true })).toBeVisible();
  const body = page.locator('.answer-body');
  await expect(body.getByRole('heading', { name: '복사한 Markdown 제목', level: 1, exact: true })).toBeVisible();
  await expect(body.locator('ol')).toHaveAttribute('start', '3');
  await expect(body.getByRole('listitem')).toHaveText(['첫 번째 항목', '두 번째 항목']);
  await expect(body.getByRole('table')).toHaveCount(1);
  await expect(body.getByRole('columnheader')).toHaveText(['이름', '값']);
  await expect(body.getByRole('cell')).toHaveText(['alpha', '7']);
  expect(await body.locator('pre code').textContent()).toBe('if True:\n    print("원문 코드")\n');
  await page.getByRole('button', { name: 'TXT 파일 만들기', exact: true }).click();
  const downloadLink = page.getByRole('link', { name: 'TXT 다운로드', exact: true });
  await expect(downloadLink).toBeVisible();
  const url = new URL((await downloadLink.getAttribute('href')) ?? '', page.url()).href;
  const downloadPromise = page.waitForEvent('download');
  await downloadLink.click();
  const download = await downloadPromise;
  const path = await download.path();
  expect(path).not.toBeNull();
  const downloaded = await readFile(path as string);
  expect(downloaded).toEqual(Buffer.from(`\ufeff${original}`));
  expect(downloaded).toEqual(await (await request.get(url)).body());
  expect((await submittedQueries(request)).filter((item) => item === query)).toEqual([query]);
  expect(consoleErrors).toEqual([]);
});
