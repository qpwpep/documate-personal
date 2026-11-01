import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';

// Resolve a non-trustworthy HTTP origin locally without changing browser APIs.
test.use({ launchOptions: { args: ['--host-resolver-rules=MAP documate.test 127.0.0.1', '--no-proxy-server'] } });

test('the production bundle serves chat from the API origin without the Vite client', async ({ page }) => {
  const assetRequests: string[] = [];
  const errors: string[] = [];
  page.on('request', (request) => assetRequests.push(request.url()));
  page.on('pageerror', (error) => errors.push(error.message));
  const document = await page.goto('./');
  expect(document?.status(), 'Run npm --prefix frontend run build before the production smoke').toBe(200);
  await page.getByRole('textbox', { name: '질문', exact: true }).fill('정적 빌드에서 질문 연결 검증');
  const submit = page.getByRole('button', { name: '질문 보내기', exact: true });
  await expect(submit).toBeEnabled();
  await submit.click();
  await expect(page.getByText('실제 FastAPI와 Python 런타임을 거친 한글 답변입니다. 🧪', { exact: true })).toBeVisible();
  expect(assetRequests.some((url) => url.includes('/app/assets/') && url.endsWith('.js'))).toBe(true);
  expect(assetRequests.some((url) => url.includes('/app/assets/') && url.endsWith('.css'))).toBe(true);
  expect(assetRequests.some((url) => url.includes('/@vite/') || url.includes('/src/main.tsx'))).toBe(false);
  expect(assetRequests.every((url) => new URL(url).origin === 'http://127.0.0.1:18000')).toBe(true);
  expect(errors).toEqual([]);
});

test.describe('plain HTTP outside localhost', () => {
  test.use({
    baseURL: 'http://documate.test:18000/app/',
  });

  test('creates conversations, uploads, asks, downloads and restores without randomUUID', async ({ page, request }) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto('./');
    expect(await page.evaluate(() => window.isSecureContext)).toBe(false);
    expect(await page.evaluate(() => typeof crypto.randomUUID)).toBe('undefined');
    expect(errors).toEqual([]);
    await expect(page.getByText('질문할 준비가 되었습니다.', { exact: true })).toBeVisible();
    await page.getByText('첨부 자료', { exact: true }).click();

    await page.getByLabel('파일 첨부', { exact: true }).setInputFiles({
      name: 'alpha.py', mimeType: 'text/x-python', buffer: Buffer.from('alpha = 1\r\n'),
    });
    const upload = page.waitForResponse((response) => response.request().method() === 'POST'
      && response.url().endsWith('/uploads/sync'));
    await page.getByRole('button', { name: '첨부만 확정', exact: true }).click();
    expect((await upload).status()).toBe(200);
    const attachment = page.getByRole('button', { name: 'alpha.py 첨부 해제', exact: true });
    await expect(attachment).toBeEnabled();

    const query = 'HTTP LAN 첨부 alpha 코드 확인';
    const input = page.getByRole('textbox', { name: '질문', exact: true });
    await input.fill(query);
    const firstSubmission = page.waitForRequest((submitted) => submitted.method() === 'POST'
      && submitted.url().endsWith('/agent/stream'));
    await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
    const firstSession = (await firstSubmission).postDataJSON().session_id as string;
    await expect(page.getByText('답변 받음', { exact: true })).toBeVisible();
    await expect(page.getByText('alpha = 1', { exact: true }).first()).toBeVisible();

    await page.getByRole('button', { name: 'TXT 파일 만들기', exact: true }).click();
    const downloadLink = page.getByRole('link', { name: 'TXT 다운로드', exact: true });
    await expect(downloadLink).toBeVisible();
    const downloadUrl = new URL((await downloadLink.getAttribute('href')) ?? '', page.url());
    const downloading = page.waitForEvent('download');
    await downloadLink.click();
    const download = await downloading;
    const path = await download.path();
    expect(path).not.toBeNull();
    const bytes = await readFile(path as string);
    expect(bytes.toString('utf8')).toContain('alpha = 1');
    expect(bytes.toString('utf8')).toContain('alpha.py');
    expect(bytes).toEqual(await (await request.get(`http://127.0.0.1:18000${downloadUrl.pathname}`)).body());

    const draft = '  새로고침 뒤 남길 초안\n둘째 줄  ';
    await input.fill(draft);
    await page.reload();
    await expect(page.getByText('질문할 준비가 되었습니다.', { exact: true })).toBeVisible();
    await page.getByText('첨부 자료', { exact: true }).click();
    await expect(input).toHaveValue(draft);
    await expect(page.getByText('alpha = 1', { exact: true }).first()).toBeVisible();
    await expect(attachment).toBeEnabled();
    await expect(downloadLink).toBeVisible();

    await page.getByRole('button', { name: '새 채팅', exact: true }).click();
    await expect(page.getByText('질문할 준비가 되었습니다.', { exact: true })).toBeVisible();
    await expect(input).toHaveValue('');
    await expect(attachment).toHaveCount(0);
    const nextQuery = 'HTTP LAN 새 대화 질문';
    await input.fill(nextQuery);
    const nextSubmission = page.waitForRequest((submitted) => submitted.method() === 'POST'
      && submitted.url().endsWith('/agent/stream'));
    await page.getByRole('button', { name: '질문 보내기', exact: true }).click();
    expect((await nextSubmission).postDataJSON().session_id).not.toBe(firstSession);
    await expect(page.getByText('실제 FastAPI와 Python 런타임을 거친 한글 답변입니다. 🧪', { exact: true })).toBeVisible();

    await page.getByRole('button', { name: `대화 선택: ${query}`, exact: true }).click();
    await expect(input).toHaveValue(draft);
    await expect(attachment).toBeEnabled();
    await expect(page.getByText('alpha = 1', { exact: true }).first()).toBeVisible();
    const received = await (await request.get('http://127.0.0.1:18000/_fixture/requests')).json() as { queries: string[] };
    expect(received.queries.filter((value) => value === query)).toEqual([query]);
    expect(received.queries.filter((value) => value === nextQuery)).toEqual([nextQuery]);
    expect(errors).toEqual([]);
  });
});
