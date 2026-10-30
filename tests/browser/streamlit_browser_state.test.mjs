import assert from "node:assert/strict";
import test from "node:test";

import browserState from "../../src/app/web/streamlit_browser_state.js";

const conversationId = "f0579919-4ccc-4934-90b5-59102fe3c371";
const storageKey = "documate.conversations.v1";
const previousMessages = [
  { role: "user", content: "이전 질문" },
  { role: "assistant", result: null },
];

function attributeElement(initial = {}) {
  const attributes = new Map(Object.entries(initial));
  return {
    dataset: {}, disabled: false,
    getAttribute: name => attributes.get(name) ?? null,
    hasAttribute: name => attributes.has(name),
    setAttribute: (name, value) => attributes.set(name, value),
    removeAttribute: name => attributes.delete(name),
  };
}

test("hot-reloaded modules retire the previous bridge before changing busy state", async t => {
  const browser = browserBoundary(t);
  const reloaded = (await import("../../src/app/web/streamlit_browser_state.js?hot-reload-regression")).default;
  const labels = [];
  const setAttribute = browser.send.setAttribute;
  browser.send.setAttribute = (name, value) => {
    setAttribute(name, value);
    if (name === "aria-label") labels.push(value);
    browser.queueMutation();
  };
  browser.render();
  assert.equal(browser.drainMutations().pending, 0);

  // Streamlit imports changed JS as a new module without calling the old
  // entry point's cleanup first. Both modules see the same native composer.
  browser.render({ component: reloaded, chatInProgress: true });
  const settled = browser.drainMutations();
  assert.equal(settled.pending, 0,
    `Hot reload kept changing the send label: ${labels.slice(-6).join(" -> ")}`);
  assert.equal(browser.send.getAttribute("aria-label"), "답변을 기다리는 중 — 초안 작성 가능");
  assert.equal(browser.send.disabled, true);
});

test("late cleanup from a replaced component cannot detach its successor", async t => {
  const browser = browserBoundary(t);
  const reloaded = (await import("../../src/app/web/streamlit_browser_state.js?replacement-regression")).default;
  const previousCleanup = browser.render();
  const nextParent = {};
  browser.render({ component: reloaded, parent: nextParent, chatInProgress: true });
  assert.equal(browser.activeObserverCount(), 1);
  previousCleanup();
  // A subsequent rerun must still find and retire the successor's observer.
  browser.render({ component: reloaded, parent: nextParent, chatInProgress: true });
  // A late unmount belongs to the previous host, not the currently visible one.
  browser.input.value = "다음 질문을 보존해줘";
  browser.dispatch("input", { target: browser.input });
  assert.equal(browser.stored().draft.text, "다음 질문을 보존해줘");
  assert.equal(browser.send.getAttribute("aria-label"), "답변을 기다리는 중 — 초안 작성 가능");
  assert.equal(browser.activeObserverCount(), 1);
});

test("an active answer retains a typed next question without submitting it", t => {
  const browser = browserBoundary(t);
  Object.assign(browser.send, attributeElement());
  browser.render({ pending: "실행 중인 질문", chatInProgress: true });
  browser.input.value = "완료 후 보낼 다음 질문";
  let prevented = false;
  browser.dispatch("keydown", {
    target: browser.input, key: "Enter",
    preventDefault() { prevented = true; }, stopImmediatePropagation() {},
  });
  assert.equal(prevented, true);
  assert.equal(browser.stored().pending_prompt, "실행 중인 질문");
  assert.equal(browser.stored().draft.text, "완료 후 보낼 다음 질문");
  assert.equal(browser.input.disabled, false);
  assert.equal(browser.send.disabled, true);
});

test("a pending file changes suggestions and the send action without a server rerun", t => {
  const browser = browserBoundary(t);
  Object.assign(browser.send, attributeElement());
  const normal = { hidden: false };
  const selected = { hidden: true, dataset: {} };
  let chips = [];
  browser.composer.querySelectorAll = selector => selector === '[data-testid="stFileChip"]' ? chips : [];
  const originalQuery = browser.document.querySelector;
  browser.document.querySelector = selector => {
    if (selector === '[class*="-selected_upload_prompts"]') return selected;
    if (selector === '[class*="-default_prompts"]') return normal;
    return originalQuery(selector);
  };
  browser.render();
  chips = [{ getAttribute: () => null, querySelector: () => null }];
  browser.notifyMutation();
  assert.equal(browser.send.getAttribute("aria-label"), "파일 보내기");
  assert.equal(selected.hidden, false);
  assert.equal(normal.hidden, true);
  browser.input.value = "이 파일의 핵심을 알려줘";
  browser.dispatch("input", { target: browser.input });
  assert.equal(browser.send.getAttribute("aria-label"), "질문과 파일 보내기");
  chips = [];
  browser.notifyMutation();
  assert.equal(browser.send.getAttribute("aria-label"), "질문 보내기");
  assert.equal(selected.hidden, true);
  assert.equal(normal.hidden, false);
});

test("rejected attachment feedback stays with the composer and clears without losing the draft", t => {
  const browser = browserBoundary(t);
  let chips = [{ getAttribute: () => "true", querySelector: () => null }];
  browser.composer.querySelectorAll = selector => selector === '[data-testid="stFileChip"]' ? chips : [];
  browser.render();
  browser.input.value = "작성 중이던 질문";
  browser.dispatch("input", { target: browser.input });
  const feedback = browser.document.getElementById("documate-upload-selection");
  assert.ok(feedback, "The fixed composer must contain attachment feedback");
  assert.equal(feedback.parentElement, browser.composer);
  assert.match(feedback.textContent, /선택 오류 1개/);
  assert.match(feedback.textContent, /제거/);
  assert.equal(feedback.getAttribute("role"), "status");
  assert.equal(browser.input.getAttribute("aria-describedby"), feedback.id);
  assert.equal(browser.send.disabled, true);
  browser.notifyMutation();
  assert.equal(browser.composer.children.length, 1, "Repeated updates must not duplicate feedback");

  chips = [];
  browser.notifyMutation();
  assert.equal(feedback.textContent, "");
  assert.equal(browser.input.getAttribute("aria-describedby"), null);
  assert.equal(browser.input.value, "작성 중이던 질문");
  assert.equal(browser.send.disabled, false);
  browser.render();
  assert.equal(browser.composer.children.length, 1, "A rerun must retire its previous feedback");
});

test("Slack recipient errors are associated with the input and clear after correction", t => {
  const browser = browserBoundary(t);
  const recipient = attributeElement({ "aria-describedby": "existing-help" });
  const feedback = { dataset: { slackInvalid: "true" } };
  const originalQuery = browser.document.querySelector;
  browser.document.querySelector = selector => selector === 'input[aria-label="Slack 수신자"]' ? recipient : originalQuery(selector);
  browser.document.getElementById = id => id === "documate-slack-recipient-help" ? feedback : null;
  browser.render();
  assert.equal(recipient.getAttribute("aria-invalid"), "true");
  assert.equal(recipient.getAttribute("aria-describedby"), "existing-help documate-slack-recipient-help");
  feedback.dataset.slackInvalid = "false";
  browser.notifyMutation();
  assert.equal(recipient.getAttribute("aria-invalid"), "false");
  assert.equal(recipient.getAttribute("aria-describedby"), "existing-help documate-slack-recipient-help");
});

test("a collapsed sidebar hides its full subtree and restores access when reopened", t => {
  const browser = browserBoundary(t);
  const sidebar = {
    ...attributeElement({ "data-collapsed": "true" }), inert: false,
    contains: element => element === child,
  };
  const child = {};
  const opener = { focus() { browser.document.activeElement = this; } };
  const originalQuery = browser.document.querySelector;
  browser.document.querySelector = selector => {
    if (selector === '[data-testid="stSidebar"]') return sidebar;
    if (selector === '[data-testid="stExpandSidebarButton"]') return opener;
    return originalQuery(selector);
  };
  browser.render({ focus: false });
  assert.equal(sidebar.inert, true);
  sidebar.setAttribute("data-collapsed", "false");
  browser.notifyMutation();
  assert.equal(sidebar.inert, false);
  browser.document.activeElement = child;
  sidebar.setAttribute("data-collapsed", "true");
  browser.notifyMutation();
  assert.equal(sidebar.inert, true);
  assert.equal(browser.document.activeElement, opener);
});

test("an inline citation opens its own source and returns focus to the citation", t => {
  const browser = browserBoundary(t);
  const source = sourceScene(browser, "desktop");
  const anchor = {
    isConnected: true,
    getAttribute: name => name === "href" ? "#dm-citation-answer-1" : null,
    closest: selector => selector === 'a[href^="#dm-citation-"]' ? anchor : null,
    getClientRects: () => [{}],
    getBoundingClientRect: () => ({ top: 150, bottom: 175 }),
    focus() { browser.document.activeElement = this; },
  };
  const originalQuery = browser.document.querySelector;
  browser.document.querySelector = selector => selector === ".st-key-dm-citation-answer-1"
    ? source.selected : originalQuery(selector);
  browser.render({ focus: false });
  let prevented = false;
  let stopped = false;
  browser.dispatch("click", {
    target: anchor,
    preventDefault() { prevented = true; },
    stopImmediatePropagation() { stopped = true; },
  });
  // Native popovers dismiss when the original link click reaches their
  // outside-click listener after the bridge opens the trigger.
  if (!stopped && source.opened) source.close("close button");
  assert.equal(prevented, true);
  assert.equal(source.opened, source.selected);
  source.close("Escape");
  assert.equal(source.opened, null);
  assert.equal(browser.document.activeElement, anchor);
});

test("a new attachment failure receives focus once without interrupting later work", t => {
  const browser = browserBoundary(t);
  const error = {
    dataset: { dmUploadError: "attempt-1" },
    focus() { browser.document.activeElement = this; },
    scrollIntoView() {},
  };
  const originalQuery = browser.document.querySelector;
  browser.document.querySelector = selector => selector === "[data-dm-upload-error]"
    ? error : originalQuery(selector);
  browser.render();
  assert.equal(browser.document.activeElement, error);
  browser.document.activeElement = browser.input;
  browser.notifyMutation();
  browser.render();
  assert.equal(browser.document.activeElement, browser.input);
  error.dataset.dmUploadError = "attempt-2";
  browser.notifyMutation();
  assert.equal(browser.document.activeElement, error);
});

test("attachment recovery begins below the fixed header on a short mobile screen", t => {
  const browser = browserBoundary(t);
  const main = {
    scrollTop: 200,
    getBoundingClientRect: () => ({ top: 0 }),
    dispatchEvent() {},
  };
  const error = {
    dataset: { dmUploadError: "short-screen-attempt" },
    getBoundingClientRect: () => ({ top: 700 - main.scrollTop }),
    focus() { browser.document.activeElement = this; },
    scrollIntoView({ block }) { main.scrollTop = block === "center" ? 520 : 700; },
  };
  const originalQuery = browser.document.querySelector;
  browser.document.querySelector = selector => {
    if (selector === "[data-dm-upload-error]") return error;
    if (selector === '[data-testid="stHeader"]') return { getBoundingClientRect: () => ({ bottom: 60 }) };
    return originalQuery(selector);
  };
  browser.document.getElementById = id => id === "stMain" ? main : null;
  browser.render();
  assert.equal(browser.document.activeElement, error);
  assert.equal(error.getBoundingClientRect().top, 76,
    "The recovery title needs a 16px gap after the fixed 60px header");
});

test("source search reruns return focus to the same occurrence of an inline citation", t => {
  const browser = browserBoundary(t);
  const source = sourceScene(browser, "desktop");
  function citation() {
    const anchor = {
      ...attributeElement({ href: "#dm-citation-answer-1" }),
      textContent: "[1]", isConnected: true,
      closest: selector => selector === 'a[href^="#dm-citation-"]' ? anchor : null,
      getClientRects: () => [{}],
      getBoundingClientRect: () => ({ top: 150, bottom: 175 }),
      focus() { browser.document.activeElement = this; },
    };
    return anchor;
  }
  let citations = [citation(), citation()];
  const originalQuery = browser.document.querySelector;
  const originalQueryAll = browser.document.querySelectorAll;
  browser.document.querySelector = selector => selector === ".st-key-dm-citation-answer-1"
    ? source.selected : originalQuery(selector);
  browser.document.querySelectorAll = selector => selector === 'a[href^="#dm-citation-"]'
    ? citations : originalQueryAll(selector);
  browser.render({ focus: false });
  browser.dispatch("click", {
    target: citations[1], preventDefault() {}, stopImmediatePropagation() {},
  });
  assert.equal(source.opened, source.selected);
  // Streamlit replaces rendered links while processing the source-search input.
  for (const anchor of citations) anchor.isConnected = false;
  citations = [citation(), citation()];
  browser.render({ focus: false });
  source.close("Escape");
  assert.equal(browser.document.activeElement, citations[1]);
});

test("reloading a completed conversation reveals its latest answer above the composer", t => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const browser = browserBoundary(t);
  const main = {
    scrollTop: 900,
    getBoundingClientRect: () => ({ top: 60 }),
    dispatchEvent() {},
    scrollTo({ top }) { this.scrollTop = top; },
  };
  const answer = { getBoundingClientRect: () => ({ top: 460 - main.scrollTop }) };
  const originalQuery = browser.document.querySelector;
  browser.document.querySelector = selector => selector === '[data-testid="stApp"]'
    ? { getAttribute: () => "notRunning" } : originalQuery(selector);
  browser.document.querySelectorAll = selector => selector === '[data-testid="stChatMessage"]'
    ? [{}, answer] : [];
  browser.document.getElementById = id => id === "stMain" ? main : null;
  browser.render();
  browser.reload();
  browser.render();
  t.mock.timers.tick(200);
  assert.equal(main.scrollTop, 384, "The latest answer must begin 16px below the visible top");
});

// Only browser/Streamlit boundaries are replaced. Input handlers, snapshot
// merging and persistence all execute the production component module.
function browserBoundary(t) {
  const storage = new Map();
  const listeners = new Map();
  const windowListeners = new Map();
  const events = [];
  const observers = new Set();
  const pendingMutations = new Set();
  const selector = `.st-key-conversation-${conversationId}-chat_input textarea`;
  const send = {
    ...attributeElement(),
    closest: value => value === '[data-testid="stChatInputSubmitButton"]' ? send : null,
  };
  const composer = {
    children: [],
    prepend(element) { element.parentElement = this; this.children.unshift(element); },
    querySelector: value => value === '[data-testid="stChatInputSubmitButton"]' ? send : null,
    querySelectorAll: () => [],
  };
  const input = {
    ...attributeElement(),
    value: "", disabled: false,
    matches: value => value === selector,
    closest: value => value === '[data-testid="stChatInput"]' ? composer : null,
    focus() {},
  };
  const document = {
    body: {}, activeElement: input,
    querySelector: value => value === selector ? input : null,
    querySelectorAll: () => [],
    getElementById: id => composer.children.find(element => element.id === id) || null,
    createElement() {
      return { ...attributeElement(), textContent: "", parentElement: null,
        remove() {
          if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(child => child !== this);
          this.parentElement = null;
        },
      };
    },
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
      constructor(callback) { this.callback = callback; }
      observe() { observers.add(this); }
      disconnect() { observers.delete(this); pendingMutations.delete(this); }
    },
    getComputedStyle: element => ({ visibility: element.visibility || "visible" }),
    requestAnimationFrame(callback) { callback(); return 1; },
    cancelAnimationFrame() {},
    WheelEvent: class {},
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
    input, send, composer,
    notifyMutation() {
      for (const observer of [...observers]) observer.callback();
    },
    queueMutation() {
      for (const observer of observers) pendingMutations.add(observer);
    },
    drainMutations(limit = 50) {
      let deliveries = 0;
      // Model MutationObserver's queued delivery, with a hard bound so a
      // microtask loop fails this test instead of hanging the test process.
      while (pendingMutations.size && deliveries < limit) {
        const observer = pendingMutations.values().next().value;
        pendingMutations.delete(observer);
        if (observers.has(observer)) observer.callback();
        deliveries += 1;
      }
      return { pending: pendingMutations.size, deliveries };
    },
    activeObserverCount: () => observers.size,
    dispatch(type, event) {
      for (const listener of listeners.get(type) || []) listener(event);
    },
    resize() {
      for (const listener of windowListeners.get("resize") || []) listener();
    },
    render({ revision = 1, pending = null, saved = null, messages = previousMessages, focus = true, chatInProgress = false,
      component = browserState, parent = parentElement } = {}) {
      const snapshot = {
        version: 1, active_conversation_id: conversationId,
        conversations: [{ id: conversationId, draft: { text: "", revision }, messages,
          pending_prompt: pending, saved_prompt: saved }],
      };
      cleanup = component({ parentElement: parent, setStateValue, data: {
        conversation_id: conversationId, revision, text: "", focus_input: focus,
        chat_in_progress: chatInProgress,
        pending_chat: pending !== null, message_count: messages.length,
        snapshot: JSON.stringify(snapshot),
      } });
      return cleanup;
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
