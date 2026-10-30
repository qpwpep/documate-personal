import { defineConfig, devices } from '@playwright/test';
import { fileURLToPath } from 'node:url';

const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 30_000,
  expect: { timeout: 10_000 },
  outputDir: '../output/react-e2e',
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:15173/app/',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    {
      command: 'uv run --locked python -X utf8 -m tests.browser.react_api_fixture --port 18000',
      cwd: repositoryRoot,
      url: 'http://127.0.0.1:18000/_fixture/ready',
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      command: 'npm run dev -- --port 15173',
      url: 'http://127.0.0.1:15173/app/',
      env: { DOCUMATE_API_PROXY_TARGET: 'http://127.0.0.1:18000' },
      reuseExistingServer: false,
      timeout: 30_000,
    },
  ],
});
