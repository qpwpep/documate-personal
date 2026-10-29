import assert from "node:assert/strict";
import test from "node:test";

import browserState from "../../src/app/web/streamlit_browser_state.js";

const conversationId = "f0579919-4ccc-4934-90b5-59102fe3c371";
const storageKey = "documate.conversations.v1";
const previousMessages = [
  { role: "user", content: "이전 질문" },
  { role: "assistant", result: null },
];

// Only browser/Streamlit boundaries are replaced. Input handlers, snapshot
// merging and persistence all execute the production component module.
function browserBoundary(t) {
  const storage = new Map();
  const listeners = new Map();
  const windowListeners = new Map();
  const events = [];
  let notifyMutation = () => {};
  const selector = `.st-key-conversation-${conversationId}-chat_input textarea`;
  const send = {
    disabled: false,
    closest: value => value === '[data-testid="stChatInputSubmitButton"]' ? send : null,
  };
  const composer = {
    querySelector: value => value === '[data-testid="stChatInputSubmitButton"]' ? send : null,
  };
  const input = {
    value: "", disabled: false,
    matches: value => value === selector,
    closest: value => value === '[data-testid="stChatInput"]' ? composer : null,
    focus() {},
  };
  const document = {
    body: {}, activeElement: input,
    querySelector: value => value === selector ? input : null,
    querySelectorAll: () => [],
    getElementById: () => null,
    addEventListener(type, listener) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(listener);
    },
    removeEventListener(type, listener) { listeners.get(type)?.delete(listener); },
  };
  const replacements = {
    document,
    window: {
      addEventListener(type, listener) {
        if (!windowListeners.has(type)) windowListeners.set(type, new Set());
        windowListeners.get(type).add(listener);
      },
      removeEventListener(type, listener) { windowListeners.get(type)?.delete(listener); },
    },
    sessionStorage: { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) },
    MutationObserver: class {
      constructor(callback) { notifyMutation = callback; }
      observe() {}
      disconnect() {}
    },
    getComputedStyle: element => ({ visibility: element.visibility || "visible" }),
    cancelAnimationFrame() {},
  };
  const originalDescriptors = Object.keys(replacements).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]);
  for (const [key, value] of Object.entries(replacements)) {
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  const parentElement = {};
  let cleanup;
  t.after(() => {
    cleanup?.();
    for (const [key, descriptor] of originalDescriptors) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  });
  const setStateValue = (name, value) => events.push({ name, value });
  return {
    events,
    document,
    notifyMutation: () => notifyMutation(),
    dispatch(type, event) {
      for (const listener of listeners.get(type) || []) listener(event);
    },
    resize() {
      for (const listener of windowListeners.get("resize") || []) listener();
    },
    render({ revision = 1, pending = null, saved = null, messages = previousMessages, focus = true } = {}) {
      const snapshot = {
        version: 1, active_conversation_id: conversationId,
        conversations: [{ id: conversationId, draft: { text: "", revision }, messages,
          pending_prompt: pending, saved_prompt: saved }],
      };
      cleanup = browserState({ parentElement, setStateValue, data: {
        conversation_id: conversationId, revision, text: "", focus_input: focus,
        pending_chat: pending !== null, message_count: messages.length,
        snapshot: JSON.stringify(snapshot),
      } });
    },
    submit(prompt, method) {
      input.value = prompt;
      const type = method === "Enter" ? "keydown" : "click";
      const event = { type, key: "Enter", target: method === "Enter" ? input : send };
      for (const listener of listeners.get(type) || []) listener(event);
      // Native Streamlit clears its composer after dispatching the submission.
      input.value = "";
    },
    stored() { return JSON.parse(storage.get(storageKey)).conversations[0]; },
    reload() {
      browserState({ parentElement, setStateValue, data: { restore: true } });
      return JSON.parse(events.at(-1).value.snapshot).conversations[0];
    },
  };
}

function sourceScene(browser, initialLayout) {
  const { document } = browser;
  let layout = initialLayout;
  let opened = null;
  let viewport = { top: 64, bottom: 780, composerTop: 623 };
  const main = {
    scrollTop: 500,
    getBoundingClientRect: () => ({ top: viewport.top, bottom: viewport.bottom }),
  };
  const header = { getBoundingClientRect: () => ({ top: 0, bottom: 60 }) };
  const bottom = { getBoundingClientRect: () => ({ top: viewport.composerTop, bottom: viewport.bottom }) };
  const event = target => ({ target, preventDefault() {}, stopImmediatePropagation() {} });
  // Streamlit retains both help-tooltip trigger copies in the same popover.
  // Responsive CSS hides one copy without disconnecting either DOM node.
  function popover() {
    const scope = {
      isConnected: true, offsetTop: main.scrollTop + 120 - viewport.top,
      querySelectorAll: selector => selector === '[data-testid="stPopoverButton"]' ? scope.buttons : [],
      querySelector(selector) { return this.querySelectorAll(selector)[0] || null; },
    };
    scope.buttons = ["desktop", "mobile"].map(mode => ({
      isConnected: true, disabled: false, textContent: "근거 [1]", mode,
      getClientRects: () => layout === mode ? [{}] : [],
      getBoundingClientRect() {
        const top = viewport.top + scope.offsetTop - main.scrollTop;
        return { top, bottom: top + 40, height: 40 };
      },
      // The browser centers within the full scroller, including the region
      // covered by the fixed composer. Do not fake an always-visible result.
      scrollIntoView() {
        main.scrollTop = scope.offsetTop + 20 - (viewport.bottom - viewport.top) / 2;
      },
      getAttribute: name => name === "aria-expanded" ? String(opened === scope) : null,
      closest(selector) {
        if (selector === '[data-testid="stPopoverButton"]') return this;
        return selector === '[data-testid="stPopover"]' ? scope : null;
      },
      focus() { if (this.getClientRects().length) document.activeElement = this; },
      click() {
        browser.dispatch("click", event(this));
        opened = opened === scope ? null : scope;
        // Removing the focused panel leaves BODY until focus is restored.
        if (!opened) document.activeElement = document.body;
        browser.notifyMutation();
      },
    }));
    const title = { focus() { document.activeElement = this; } };
    scope.panel = {
      querySelector: selector => selector === ".dm-source-header" ? {} : selector === ".dm-source-title" ? title : null,
      getClientRects: () => opened === scope ? [{}] : [],
    };
    return scope;
  }
  const other = popover();
  const selected = popover();
  const originalQuery = document.querySelector;
  document.querySelectorAll = selector => {
    if (selector === "main") return [main];
    if (selector === '[data-testid="stHeader"]') return [header];
    if (selector === '[data-testid="stBottom"]') return [bottom];
    if (selector === '[data-testid="stPopoverBody"]') return opened ? [opened.panel] : [];
    if (selector === '[data-testid="stPopoverButton"]') return [...other.buttons, ...selected.buttons];
    if (selector === '[data-testid="stPopoverButton"][aria-expanded="true"]') return opened?.buttons || [];
    return [];
  };
  document.querySelector = selector => document.querySelectorAll(selector)[0] || originalQuery(selector);
  document.getElementById = id => id === "stMain" ? main : null;
  return {
    other, selected, main, bottom,
    get opened() { return opened; },
    open() { selected.buttons.find(button => button.mode === layout).click(); },
    resize(nextLayout, top = selected.buttons[0].getBoundingClientRect().top, nextViewport = viewport) {
      layout = nextLayout;
      viewport = nextViewport;
      selected.offsetTop = main.scrollTop + top - viewport.top;
      browser.resize();
      browser.notifyMutation();
    },
    close(method) {
      if (method === "Escape") {
        browser.dispatch("keydown", { ...event(document.activeElement), key: "Escape" });
      } else {
        const closeButton = { closest: selector => selector === ".dm-source-close" ? closeButton : null };
        browser.dispatch("click", event(closeButton));
      }
    },
  };
}

for (const method of ["Enter", "send button"]) {
  test(`${method}: reload retains the submitted question before Python consumes it`, t => {
    const browser = browserBoundary(t);
    const prompt = "  첨부를 검토해줘\n질문 원문 유지  ";
    browser.render();
    browser.submit(prompt, method);
    assert.equal(browser.stored().pending_prompt, prompt);

    // A completed answer restored focus. The next Python run clears that
    // flag before chat_input consumes the new submission and its revision.
    browser.render({ focus: false });
    assert.deepEqual(browser.stored().draft, { text: "", revision: 2 });
    assert.equal(browser.stored().pending_prompt, prompt);
    const restored = browser.reload();
    assert.equal(restored.pending_prompt, prompt);
    assert.deepEqual(restored.messages, previousMessages);
    assert.deepEqual(browser.events.map(event => event.name), ["restore"]);
  });
}

for (const outcome of ["completed response", "held attachment question"]) {
  test(`${outcome}: acknowledged server state replaces the submission marker`, t => {
    const browser = browserBoundary(t);
    const prompt = "다음 질문";
    browser.render();
    browser.submit(prompt, "Enter");
    browser.render({ revision: 2, pending: prompt, focus: false });
    assert.equal(browser.stored().pending_prompt, prompt);

    // Completion and attachment review must each clear the uncertainty marker;
    // otherwise reload would append the same question again as interrupted.
    const saved = outcome === "held attachment question" ? prompt : null;
    const messages = saved ? previousMessages : [...previousMessages,
      { role: "user", content: prompt }, { role: "assistant", result: null }];
    browser.render({ revision: 2, saved, messages });
    const restored = browser.reload();
    assert.equal(restored.pending_prompt, null);
    assert.equal(restored.saved_prompt, saved);
    assert.deepEqual(restored.messages, messages);
  });
}

for (const [from, to] of [["desktop", "desktop"], ["desktop", "mobile"], ["mobile", "desktop"]]) {
  for (const method of ["Escape", "close button"]) {
    test(`${from} to ${to}, ${method}: focus returns to the visible trigger of the opened source`, t => {
      const browser = browserBoundary(t);
      const source = sourceScene(browser, from);
      browser.render({ focus: false });
      source.open();
      const originalTrigger = source.selected.buttons.find(button => button.mode === from);
      assert.equal(source.opened, source.selected);

      source.resize(to);
      assert.equal(originalTrigger.isConnected, true);
      assert.equal(originalTrigger.getClientRects().length, from === to ? 1 : 0);
      source.close(method);

      const visibleTrigger = source.selected.buttons.find(button => button.mode === to);
      assert.equal(source.opened, null);
      assert.equal(visibleTrigger.getAttribute("aria-expanded"), "false");
      assert.equal(visibleTrigger.getClientRects().length, 1);
      assert.equal(browser.document.activeElement, visibleTrigger);
      assert.equal(visibleTrigger.getBoundingClientRect().top, 120, "An already visible trigger changed reading position");
      assert(!source.other.buttons.includes(browser.document.activeElement), "Focus moved to another source with the same label");
    });
  }
}

for (const [position, top] of [["above the reading area", 20], ["behind the composer", 661]]) {
  test(`resized source ${position}: closing returns focus inside the unobscured reading area`, t => {
    const browser = browserBoundary(t);
    const source = sourceScene(browser, "desktop");
    browser.render({ focus: false });
    source.open();
    source.resize("mobile", top);
    source.close("Escape");

    const trigger = source.selected.buttons.find(button => button.mode === "mobile");
    assert.equal(source.opened, null);
    assert.equal(browser.document.activeElement, trigger);
    const rect = trigger.getBoundingClientRect();
    assert(rect.top >= source.main.getBoundingClientRect().top,
      `Focused trigger is above the reading area: ${rect.top}`);
    assert(rect.bottom <= source.bottom.getBoundingClientRect().top,
      `Focused trigger is behind the composer: ${rect.bottom}`);
  });
}

for (const method of ["Escape", "close button"]) {
  for (const [position, top] of [["above the reading area", 20], ["behind the composer", 190], ["already visible", 100]]) {
    test(`short viewport, ${position}, ${method}: the focused source remains fully readable`, t => {
      const browser = browserBoundary(t);
      const source = sourceScene(browser, "desktop");
      browser.render({ focus: false });
      source.open();
      // Matches the 390×360 browser reproduction: the main scroller extends
      // below a fixed composer that starts at about y=203.
      source.resize("mobile", top, { top: 60, bottom: 360, composerTop: 203.5 });
      const previousScrollTop = source.main.scrollTop;
      source.close(method);

      const trigger = source.selected.buttons.find(button => button.mode === "mobile");
      assert.equal(source.opened, null);
      assert.equal(browser.document.activeElement, trigger);
      assert(!source.other.buttons.includes(browser.document.activeElement), "Focus moved to another source with the same label");
      const rect = trigger.getBoundingClientRect();
      assert(rect.top >= 60, `Focused trigger is above the reading area: ${rect.top}`);
      assert(rect.bottom <= 203.5, `Focused trigger is behind the composer: ${rect.bottom}`);
      if (position === "already visible") {
        assert.equal(rect.top, 100, "An already visible trigger changed reading position");
        assert.equal(source.main.scrollTop, previousScrollTop);
      }
    });
  }
}

for (const method of ["Escape", "close button"]) {
  for (const [position, top] of [["behind the header", 30], ["already visible", 100]]) {
    test(`mobile to desktop, ${position}, ${method}: the header does not cover the focused source`, t => {
      const browser = browserBoundary(t);
      const source = sourceScene(browser, "mobile");
      browser.render({ focus: false });
      source.open();
      // On desktop the main scroller starts at y=0, but the fixed toolbar
      // still intercepts interaction with content in the first 60 pixels.
      source.resize("desktop", top, { top: 0, bottom: 780, composerTop: 623 });
      const previousScrollTop = source.main.scrollTop;
      source.close(method);

      const trigger = source.selected.buttons.find(button => button.mode === "desktop");
      assert.equal(source.opened, null);
      assert.equal(browser.document.activeElement, trigger);
      assert(!source.other.buttons.includes(browser.document.activeElement), "Focus moved to another source with the same label");
      const rect = trigger.getBoundingClientRect();
      assert(rect.top >= 60, `Focused trigger is behind the header: ${rect.top}`);
      assert(rect.bottom <= 623, `Focused trigger is behind the composer: ${rect.bottom}`);
      if (position === "already visible") {
        assert.equal(rect.top, 100, "An already visible trigger changed reading position");
        assert.equal(source.main.scrollTop, previousScrollTop);
      }
    });
  }
}
