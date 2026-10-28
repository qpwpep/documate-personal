/** Real-browser regressions for the responsive sidebar, run through CUA.
 * AppTest does not execute keyboard navigation or the rendered overlay.
 */
function assert(condition, message) {
  if (!condition) throw new Error(message);
}

/** Check ownership immediately after opening; the native CSS transition may
 * still be moving the focused control into view. Visibility is a separate
 * assertion after the caller has observed the settled rendered sidebar.
 */
export async function mobileSidebarOpened(tab) {
  const state = await tab.playwright.evaluate(() => {
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    const active = document.activeElement;
    const controlled = [...sidebar.querySelectorAll('[aria-controls]')]
      .flatMap(el => el.getAttribute('aria-controls').split(/\s+/))
      .map(id => document.getElementById(id)).filter(Boolean);
    const rect = active.getBoundingClientRect();
    return {
      open: sidebar.getAttribute('data-collapsed') === 'false',
      inside: sidebar.contains(active) || controlled.some(el => el.contains(active)),
      label: active.getAttribute('aria-label') || active.textContent?.slice(0, 80),
      tag: active.tagName,
      visible: getComputedStyle(active).visibility === 'visible'
        && rect.width > 0 && rect.height > 0 && rect.bottom > 0
        && rect.top < innerHeight && rect.right > 0 && rect.left < innerWidth,
    };
  });
  assert(state.open, `Sidebar is not open: ${JSON.stringify(state)}`);
  assert(state.inside,
    `Mobile sidebar focus escaped into hidden/background content: ${JSON.stringify(state)}`);
  return { check: 'mobile sidebar immediately owns focus', ...state };
}

/** Run after observing the settled sidebar, or during ordinary Tab traversal. */
export async function mobileSidebarFocus(tab) {
  const state = await mobileSidebarOpened(tab);
  assert(state.visible, `Sidebar focus is not visible: ${JSON.stringify(state)}`);
  return { ...state, check: 'mobile sidebar owns visible focus' };
}

/** Call after the open transition, without replacing the immediate check above. */
export async function mobileSidebarTabCycle(tab, { steps = 24, backwards = false } = {}) {
  const visited = [await mobileSidebarFocus(tab)];
  for (let index = 0; index < steps; index += 1) {
    await tab.pressKey(null, backwards ? 'shift+Tab' : 'Tab');
    visited.push(await mobileSidebarFocus(tab));
  }
  return { check: 'mobile sidebar Tab containment', backwards,
    visited: visited.map(item => ({ tag: item.tag, label: item.label })) };
}

/** Run after Escape or the visible close button. */
export async function sidebarClosedFocus(tab) {
  const state = await tab.playwright.evaluate(() => ({
    closed: document.querySelector('[data-testid="stSidebar"]')?.getAttribute('data-collapsed') === 'true',
    openerFocused: document.activeElement?.getAttribute('data-testid') === 'stExpandSidebarButton',
    mainBlocked: Boolean(document.querySelector('#stMain')?.closest('[inert]')),
  }));
  assert(state.closed && state.openerFocused && !state.mainBlocked,
    `Closing the sidebar did not restore normal navigation: ${JSON.stringify(state)}`);
  return { check: 'sidebar closes and returns focus', ...state };
}

/** With an open desktop sidebar, Tab through the native header into main. */
export async function desktopMainFocused(tab) {
  const state = await tab.playwright.evaluate(() => ({
    open: document.querySelector('[data-testid="stSidebar"]')?.getAttribute('data-collapsed') === 'false',
    inMain: Boolean(document.activeElement?.closest('#stMain')),
    mainBlocked: Boolean(document.querySelector('#stMain')?.closest('[inert]')),
    label: document.activeElement?.getAttribute('aria-label') || document.activeElement?.textContent?.slice(0, 80),
  }));
  assert(state.open && state.inMain && !state.mainBlocked,
    `Desktop sidebar prevented access to main content: ${JSON.stringify(state)}`);
  return { check: 'desktop keeps sidebar and main available', ...state };
}
