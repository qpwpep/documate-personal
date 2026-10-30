import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

const target = process.env.DOCUMATE_API_PROXY_TARGET ?? 'http://127.0.0.1:8000';

export default defineConfig({
  base: '/app/',
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: Object.fromEntries(
      ['/agent', '/sessions', '/client-config', '/slack', '/download'].map((path) => [
        path,
        { target },
      ]),
    ),
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    clearMocks: true,
  },
});
