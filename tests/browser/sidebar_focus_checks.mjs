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
    sidebarInert: Boolean(document.querySelector('[data-testid="stSidebar"]')?.inert),
    openerFocused: document.activeElement?.getAttribute('data-testid') === 'stExpandSidebarButton',
    mainBlocked: Boolean(document.querySelector('#stMain')?.closest('[inert]')),
  }));
  assert(state.closed && state.sidebarInert && state.openerFocused && !state.mainBlocked,
    `Closing the sidebar did not restore normal navigation: ${JSON.stringify(state)}`);
  return { check: 'sidebar closes and returns focus', ...state };
}

/** Run after choosing another conversation on a narrow screen. */
export async function mobileConversationSelected(tab, expectedTitle) {
  const state = await sidebarClosedFocus(tab);
  const conversation = await tab.playwright.evaluate(() => {
    const current = document.querySelector('.st-key-documate_recent_conversations button:disabled');
    const title = document.querySelector('.dm-conversation-title');
    const titleRect = title?.getBoundingClientRect();
    const mainRect = document.querySelector('#stMain').getBoundingClientRect();
    const visibleTop = Math.max(mainRect.top,
      document.querySelector('[data-testid="stHeader"]')?.getBoundingClientRect().bottom || 0);
    return {
      current: current?.textContent || "", pending: Boolean(document.querySelector('textarea:disabled')),
      scriptState: document.querySelector('[data-testid="stApp"]')?.getAttribute('data-test-script-state'),
      title: title?.textContent || "", titleTop: titleRect?.top, titleBottom: titleRect?.bottom,
      titleVisible: Boolean(titleRect && titleRect.width > 0 && titleRect.height > 0
        && titleRect.top >= visibleTop - 1 && titleRect.bottom <= Math.min(mainRect.bottom, innerHeight) + 1),
    };
  });
  assert(conversation.current.includes(expectedTitle),
    `The selected conversation is not identified: ${JSON.stringify(conversation)}`);
  assert(conversation.scriptState === "notRunning",
    `The selected conversation is still rendering: ${JSON.stringify(conversation)}`);
  assert(conversation.title.includes(expectedTitle) && conversation.titleVisible,
    `The selected conversation title is outside the reading area: ${JSON.stringify(conversation)}`);
  assert(!conversation.pending, "Changing conversations started a request");
  return { check: "mobile conversation selection returns to the conversation", ...state, ...conversation };
}

/** Check localized controls after the composer is restored. */
export async function koreanControlLabels(tab) {
  const labels = await tab.playwright.evaluate(() => ({
    upload: document.querySelector('[data-testid="stChatInput"] button[aria-label="파일 첨부"]')?.getAttribute('aria-label'),
    send: document.querySelector('[data-testid="stChatInputSubmitButton"]')?.getAttribute('aria-label'),
    open: document.querySelector('[data-testid="stExpandSidebarButton"]')?.getAttribute('aria-label'),
  }));
  assert(labels.upload === "파일 첨부" && labels.send === "질문 보내기" && labels.open === "대화 메뉴 열기",
    `Composer and navigation labels are not in Korean: ${JSON.stringify(labels)}`);
  return { check: "Korean composer and sidebar labels", ...labels };
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
