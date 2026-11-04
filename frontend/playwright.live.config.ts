import { defineConfig, devices } from '@playwright/test';

if (process.env.RUN_LIVE_BROWSER !== '1') {
  throw new Error('Live browser tests call paid APIs. Set RUN_LIVE_BROWSER=1 explicitly to run them.');
}
const baseURL = process.env.DOCUMATE_LIVE_BASE_URL;
if (!baseURL) throw new Error('Set DOCUMATE_LIVE_BASE_URL to an already running application URL, including /app/.');
const endpoint = new URL(baseURL);
if (!['http:', 'https:'].includes(endpoint.protocol) || !endpoint.pathname.endsWith('/app/')) {
  throw new Error('DOCUMATE_LIVE_BASE_URL must be an HTTP(S) application URL ending in /app/.');
}

export default defineConfig({
  testDir: './tests/live',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: true,
  timeout: 300_000,
  expect: { timeout: 20_000 },
  outputDir: '../output/react-live/artifacts',
  reporter: [['list'], ['json', { outputFile: '../output/react-live/results.json' }]],
  use: {
    ...devices['Desktop Chrome'],
    baseURL,
    timezoneId: 'Asia/Seoul',
    locale: 'ko-KR',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'live-chromium' }],
});
