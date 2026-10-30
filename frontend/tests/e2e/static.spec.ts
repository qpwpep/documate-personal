import { expect, test } from '@playwright/test';

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
