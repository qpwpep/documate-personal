import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, beforeEach } from 'vitest';

let originalClipboard: PropertyDescriptor | undefined;

// user-event cannot register clipboard cleanup while Vitest globals are disabled.
beforeEach(() => {
  if (typeof window !== 'undefined') {
    originalClipboard = Object.getOwnPropertyDescriptor(window.navigator, 'clipboard');
  }
});

afterEach(() => {
  cleanup();
  if (typeof window === 'undefined') return;
  if (originalClipboard) Object.defineProperty(window.navigator, 'clipboard', originalClipboard);
  else Reflect.deleteProperty(window.navigator, 'clipboard');
});
