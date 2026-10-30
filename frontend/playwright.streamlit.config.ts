import { defineConfig, devices } from '@playwright/test';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  testDir: './tests/streamlit',
  workers: 1,
  retries: 0,
  timeout: 45_000,
  expect: { timeout: 15_000 },
  outputDir: '../output/streamlit-e2e',
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:18502',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: 'uv run --locked python -X utf8 -m streamlit run tests/browser/streamlit_fixture.py --server.address 127.0.0.1 --server.port 18502 --server.headless true --browser.gatherUsageStats false',
    cwd: fileURLToPath(new URL('../', import.meta.url)),
    url: 'http://127.0.0.1:18502/_stcore/health',
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
