// Keep Streamlit's own composer, upload handling, and public prefill mechanism.
// Streamlit imports changed JS as a fresh module without first calling the
// previous module's cleanup. This bridge owns document-wide listeners, so its
// owner must survive module replacement as well as ordinary data reruns.
const bridgeCleanupKey = Symbol.for("documate.browser-state.cleanup");
// Native chat widgets discard their local File list when a conversation is
// unmounted. Keep the original bytes in this page, not in sessionStorage.
const conversationFiles = new Map();
const restoredComposers = new WeakMap();
const restoredPages = new WeakSet();
const sourceOrigins = new WeakMap();

export default function ({ data, setStateValue, parentElement }) {
  // Streamlit reruns this entry point for new data; it only calls its returned
  // cleanup on unmount. Retire the previous listeners before replacing them.
  document[bridgeCleanupKey]?.();
  const storageKey = "documate.conversations.v1";
  const readingKey = "documate.reading-position.v1";
  const uploadErrorKey = "documate.upload-error-focus.v1";
  const storageError = "브라우저에 대화를 저장하지 못했습니다. 새로고침하면 최근 대화와 초안을 잃을 수 있습니다.";
  if (data.restore) {
    restoredPages.add(parentElement);
    try {
      setStateValue("restore", { snapshot: sessionStorage.getItem(storageKey), error: null });
    } catch {
      setStateValue("restore", { snapshot: null, error: storageError });
    }
    return;
  }

  const snapshot = JSON.parse(data.snapshot);
  let closeAfterNavigation = false;
  try {
    const previous = JSON.parse(sessionStorage.getItem(storageKey) || "null");
    if (previous?.version === 1 && Array.isArray(previous.conversations)) {
      closeAfterNavigation = previous.active_conversation_id !== snapshot.active_conversation_id;
      for (const conversation of snapshot.conversations) {
        const stored = previous.conversations.find(item => item.id === conversation.id);
        const draft = stored?.draft;
        // Python advances the revision on submit/edit. An unchanged revision
        // must not overwrite text just typed in the native browser input.
        if (typeof draft?.text === "string" && Number.isInteger(draft.revision)
            && draft.revision >= conversation.draft.revision) {
          // A submit clears the browser draft before Python consumes it. Keep
          // its question until a server snapshot acknowledges that revision.
          if (draft.revision > conversation.draft.revision) {
            conversation.pending_prompt = stored.pending_prompt ?? null;
          }
          conversation.draft = draft;
        }
      }
    }
  } catch {
    // Bootstrap validates stored JSON. A later unreadable value is replaced
    // with the current server-rendered display state, if storage allows it.
  }
  const storedConversation = snapshot.conversations.find(item => item.id === data.conversation_id);
  for (const conversationId of conversationFiles.keys()) {
    if (!snapshot.conversations.some(item => item.id === conversationId)) conversationFiles.delete(conversationId);
  }
  const selector = `.st-key-conversation-${data.conversation_id}-chat_input textarea`;
  let restoringFiles = false;
  let lastSent = data.text;
  let submitting = false;
  let closeRequested = false;
  let sidebarHadFocus = false;
  let sidebarOverlayActive = false;
  let sourcePanel = null;
  let sourceOpener = null;
  let sourceLink = null;
  let sourceFrame;
  let interruptRequested = false;
  const sidebarBackground = new Map();
  let focusInput = data.focus_input;
  let focusedUploadError = null;
  try {
    focusedUploadError = sessionStorage.getItem(uploadErrorKey);
  } catch {
    // The conversation bridge already reports unavailable browser storage.
  }
  let previousReading = null;
  try {
    const stored = JSON.parse(sessionStorage.getItem(readingKey) || "null");
    if (stored?.conversation_id === data.conversation_id && Number.isInteger(stored.message_count) && stored.message_count >= 0
        && typeof stored.pending === "boolean"
        && (stored.top === null || (Number.isFinite(stored.top) && stored.top >= 0))) {
      previousReading = stored;
    }
  } catch {
    // Conversation persistence already reports unavailable browser storage.
  }
  // A fast response can reach the browser without displaying the busy run.
  const newAnswer = Boolean(previousReading && !data.pending_chat
    && data.message_count > previousReading.message_count
    && storedConversation.messages.at(-1)?.role === "assistant");
  const reading = {
    conversation_id: data.conversation_id,
    message_count: newAnswer ? previousReading.message_count : data.message_count,
    pending: data.pending_chat,
    top: previousReading?.pending || newAnswer ? previousReading.top : null,
    // Draft and navigation events can cause separate reruns before scrolling.
    navigation: closeAfterNavigation || previousReading?.navigation === true,
  };
  let observeUserScroll = reading.top !== null;
  // A restored composer makes Streamlit jump past the answer to the footer.
  // On reload, show the latest answer while keeping live reading rules intact.
  const restoredAnswer = restoredPages.has(parentElement)
    && storedConversation.messages.at(-1)?.role === "assistant";
  let answerScrollPending = newAnswer || restoredAnswer;
  let navigationScrollPending = reading.navigation;
  let scrollTimer;
  let readingFrame;

  function persistReading() {
    try {
      sessionStorage.setItem(readingKey, JSON.stringify(reading));
    } catch {
      // Keep the current view usable when browser storage is unavailable.
    }
  }

  function recordReadingPosition() {
    const main = document.getElementById("stMain");
    if (!reading.pending || !observeUserScroll || !main) return;
    // Returning to the latest content opts back into seeing the new answer.
    reading.top = main.scrollHeight - main.clientHeight - main.scrollTop > 80 ? main.scrollTop : null;
    persistReading();
  }

  function onReadingIntent(event) {
    if (!event.isTrusted) return;
    if (!event.target.closest?.("#stMain") || event.target.closest?.('[data-testid="stChatInput"]')) return;
    if (event.type === "keydown"
        && (!["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(event.key)
          || event.target.closest?.('input, textarea, select, [contenteditable="true"]'))) return;
    observeUserScroll = true;
    // A gesture after completion takes priority over an automatic adjustment.
    if (answerScrollPending || navigationScrollPending) {
      reading.message_count = data.message_count;
      reading.navigation = false;
      persistReading();
    }
    answerScrollPending = false;
    navigationScrollPending = false;
    restoredPages.delete(parentElement);
    clearTimeout(scrollTimer);
    cancelAnimationFrame(readingFrame);
    readingFrame = requestAnimationFrame(recordReadingPosition);
  }

  function settleReadingScroll() {
    if (!answerScrollPending && !navigationScrollPending) return;
    clearTimeout(scrollTimer);
    const input = document.querySelector(selector);
    const messages = document.querySelectorAll('[data-testid="stChatMessage"]');
    if (messages.length !== data.message_count) return;
    if (answerScrollPending && (!input || input.disabled)) return;
    if (navigationScrollPending && !document.querySelector('.dm-conversation-title')) return;
    // Message rows can arrive before the restored composer. Wait for the run
    // to finish so its later mount cannot restart native scrolling afterward.
    if (document.querySelector('[data-testid="stApp"]')
        ?.getAttribute("data-test-script-state") !== "notRunning") return;
    // Streamlit animates to the bottom when the composer returns. Wait for
    // both its scrolling and the final answer layout to settle before moving.
    scrollTimer = setTimeout(() => {
      const main = document.getElementById("stMain");
      if (!main) return;
      // The same main container survives conversation changes. A new view
      // starts at its title; ordinary reruns retain the current reading policy.
      if (navigationScrollPending) {
        navigationScrollPending = false;
        moveReadingPosition(main, 0);
        return;
      }
      const answer = document.querySelectorAll('[data-testid="stChatMessage"]')[data.message_count - 1];
      if (!answer) return;
      const visibleTop = Math.max(main.getBoundingClientRect().top,
        document.querySelector('[data-testid="stHeader"]')?.getBoundingClientRect().bottom || 0);
      const top = reading.top ?? main.scrollTop + answer.getBoundingClientRect().top - visibleTop - 16;
      answerScrollPending = false;
      moveReadingPosition(main, top);
    }, 160);
  }

  function moveReadingPosition(main, top) {
    // Streamlit's animation can overwrite scrollTo. Its main-container
    // wheel listener cancels that animation; a zero-delta event targets
    // no widget and performs no scrolling or input action of its own.
    main.dispatchEvent(new WheelEvent("wheel"));
    readingFrame = requestAnimationFrame(() => {
      reading.navigation = false;
      reading.message_count = data.message_count;
      restoredPages.delete(parentElement);
      persistReading();
      main.scrollTo({ top: Math.max(0, top), behavior: "instant" });
    });
  }

  function onMainScroll(event) {
    if (event.target.id !== "stMain") return;
    recordReadingPosition();
    settleReadingScroll();
  }

  function persist() {
    let error = null;
    try {
      sessionStorage.setItem(storageKey, JSON.stringify(snapshot));
    } catch {
      error = storageError;
    }
    if (error !== (data.storage_error || null)) setStateValue("storage_error", error);
  }

  function storeDraft() {
    const input = document.querySelector(selector);
    if (submitting || !input || input.disabled) return;
    storedConversation.draft.text = input.value;
    persist();
  }

  function beginSubmission() {
    const prompt = document.querySelector(selector)?.value;
    submitting = true;
    conversationFiles.delete(data.conversation_id);
    // Keep the question before the server's queued-request snapshot arrives.
    // Reloading this marker only restores an uncertain outcome, never resends.
    if (prompt?.trim()) storedConversation.pending_prompt = prompt;
    storedConversation.draft = { text: "", revision: data.revision + 1 };
    persist();
  }

  function flushDraft() {
    const input = document.querySelector(selector);
    if (submitting || !input || input.disabled || input.value === lastSent) return;
    lastSent = input.value;
    setStateValue("draft", {
      conversation_id: data.conversation_id,
      revision: data.revision,
      text: input.value,
    });
  }

  function onKeyDown(event) {
    // Capture keyboard navigation before leaving the composer. DOM removal
    // after submit also emits blur, so blur must never request a rerun.
    if (event.key === "Tab" && event.target.matches?.(selector)) flushDraft();
    if (event.key === "Enter" && !event.shiftKey && !event.ctrlKey && !event.metaKey && !event.isComposing
        && event.target.matches?.(selector)) {
      if (blockUnavailableSubmission(event)) return;
      const send = event.target.closest('[data-testid="stChatInput"]')
        ?.querySelector('[data-testid="stChatInputSubmitButton"]');
      if (send && !send.disabled) beginSubmission();
    }
  }

  function onPointerDown(event) {
    const send = event.target.closest?.('[data-testid="stChatInputSubmitButton"]');
    if (send && blockUnavailableSubmission(event)) return;
    // Flush before a navigation trigger; that trigger then carries this widget
    // state even if Streamlit coalesces the two rerun requests.
    if (!event.target.closest?.('[data-testid="stChatInput"]')) flushDraft();
    if (event.target.closest?.('[data-testid="stSidebarCollapseButton"]')) {
      closeRequested = true;
    }
  }

  function onClick(event) {
    const citation = event.target.closest?.('a[href^="#dm-citation-"]');
    if (citation) {
      const target = citation.getAttribute("href").slice(1);
      if (!/^dm-citation-[\w-]+$/.test(target)) return;
      const buttons = document.querySelector(`.st-key-${target}`)
        ?.querySelectorAll('[data-testid="stPopoverButton"]') || [];
      const button = [...buttons].find(element => !element.disabled && element.getClientRects().length
        && getComputedStyle(element).visibility === "visible");
      if (button) {
        event.preventDefault();
        event.stopImmediatePropagation();
        // Finish the link event before opening the native popover. Otherwise
        // its outside-click handler can immediately dismiss the new panel.
        cancelAnimationFrame(sourceFrame);
        sourceFrame = requestAnimationFrame(() => {
          if (!button.isConnected) return;
          button.click();
          sourceLink = citation;
          const href = citation.getAttribute("href");
          const links = [...document.querySelectorAll('a[href^="#dm-citation-"]')]
            .filter(link => link.getAttribute("href") === href);
          sourceOrigins.set(parentElement, { href, index: links.indexOf(citation) });
        });
      }
      return;
    }
    const headerButton = event.target.closest?.('[data-testid="stHeader"] button');
    if (data.operation_id && headerButton
        && (headerButton.textContent.trim() === "Stop" || headerButton.dataset.documateInterrupt)) {
      // Streamlit's native Stop prevents the rerun that restores this UI.
      // A component event interrupts the receiver while allowing a normal
      // rerun. It does not cancel the independent server request.
      event.preventDefault();
      event.stopImmediatePropagation();
      if (!interruptRequested) {
        interruptRequested = true;
        localizeControls();
        setStateValue("interrupt", {
          operation_id: data.operation_id,
          conversation_id: data.conversation_id,
          nonce: Date.now(),
        });
      }
      return;
    }
    if (event.target.closest?.('.dm-source-close')) {
      event.preventDefault();
      closeSourcePanel();
      return;
    }
    const sourceButton = event.target.closest?.('[data-testid="stPopoverButton"]');
    if (sourceButton) {
      if (sourceButton.getAttribute("aria-expanded") !== "true") {
        sourceLink = null;
        sourceOrigins.delete(parentElement);
      }
      sourceOpener = sourceButton;
    }
    const send = event.target.closest?.('[data-testid="stChatInputSubmitButton"]');
    if (send && blockUnavailableSubmission(event)) return;
    // Keyboard activation of the send button has no pointerdown event.
    if (send && !send.disabled && !submitting) beginSubmission();
    // The component's draft event and a Streamlit button callback can arrive
    // separately. Protect the visible draft before a suggestion can replace it.
    const suggestion = event.target.closest?.('[class*="-quick_prompt_"] button');
    if (suggestion && document.querySelector(selector)?.value) {
      flushDraft();
      event.preventDefault();
      event.stopImmediatePropagation();
    }
  }

  function onInput(event) {
    if (event.target.matches?.(selector)) {
      storeDraft();
      renderUploadSelection();
    }
    // Re-enable examples as soon as the user clears a draft, without rerunning
    // on every keystroke. The submit latch excludes submission-driven clearing.
    if (event.target.matches?.(selector) && event.target.value === "") flushDraft();
  }

  function rememberSelectedFiles(event) {
    if (restoringFiles || submitting) return;
    const composer = document.querySelector(selector)?.closest('[data-testid="stChatInput"]');
    if (!composer?.contains(event.target)) return;
    let files;
    if (event.type === "change") {
      if (!event.target.matches?.('input[type="file"]')) return;
      files = event.target.files;
    } else if (event.type === "drop") {
      files = event.dataTransfer?.files;
    } else {
      files = event.clipboardData?.files;
    }
    if (!files?.length) return;
    if (data.chat_in_progress) {
      event.preventDefault();
      event.stopImmediatePropagation();
      return;
    }
    conversationFiles.set(data.conversation_id, [
      ...(conversationFiles.get(data.conversation_id) || []), ...files,
    ]);
  }

  function forgetRemovedFile(event) {
    const remove = event.target.closest?.('[data-testid="stFileChipDeleteBtn"]');
    const composer = document.querySelector(selector)?.closest('[data-testid="stChatInput"]');
    if (!remove || !composer?.contains(remove)) return;
    const chip = remove.closest('[data-testid="stFileChip"]');
    const filename = chip.querySelector('[data-testid="stFileChipName"]')?.title;
    const files = conversationFiles.get(data.conversation_id) || [];
    // Native validation renders rejected files before accepted uploads. Its
    // chip exposes the name and rounded size, but no file ID. Size-limit status
    // distinguishes files on either side of the limit even when sizes round
    // to the same label. Equal signatures retain their selection order.
    const label = chip.getAttribute("aria-label");
    const tooLarge = fileChipTooLarge(chip);
    const matchingChips = [...composer.querySelectorAll('[data-testid="stFileChip"]')]
      .filter(item => item.getAttribute("aria-label") === label && fileChipTooLarge(item) === tooLarge);
    const matchingFiles = files.flatMap((file, index) =>
      file.name === filename && `${file.name}, ${nativeFileSize(file.size)}` === label
        && (file.size > data.max_file_bytes) === tooLarge ? [index] : []);
    const index = matchingFiles[matchingChips.indexOf(chip)] ?? -1;
    if (index < 0) return;
    files.splice(index, 1);
    if (!files.length) conversationFiles.delete(data.conversation_id);
  }

  function fileChipTooLarge(chip) {
    const error = chip.querySelector('[role="alert"]')?.textContent || "";
    return error.includes(" or smaller.") || error.includes("File is too large.");
  }

  function nativeFileSize(bytes) {
    // Streamlit's visible chip sizes use binary units on Windows and decimal
    // units elsewhere, advancing units at half the base with one decimal.
    const base = /^Win/i.test(navigator.platform) ? 1024 : 1000;
    const units = ["B", "KB", "MB", "GB"];
    let unit = 0;
    while (bytes > base / 2 && unit < units.length - 1) {
      bytes /= base;
      unit += 1;
    }
    return `${bytes.toFixed(1)}${units[unit]}`;
  }

  function restoreSelectedFiles() {
    const composer = document.querySelector(selector)?.closest('[data-testid="stChatInput"]');
    const input = composer?.querySelector('[data-testid="stChatInputFileUploadButton"] input[type="file"]');
    if (!input || input.disabled || restoredComposers.get(composer) === data.conversation_id) return;
    // Expanding a multiline composer replaces its file input while preserving
    // its pending native upload. Restore once per composer, not per input.
    restoredComposers.set(composer, data.conversation_id);
    const files = conversationFiles.get(data.conversation_id);
    if (!files?.length || composer.querySelector('[data-testid="stFileChip"]')) return;
    const selection = new DataTransfer();
    for (const file of files) selection.items.add(file);
    // Re-enter Streamlit's normal validation/upload path for the new widget.
    // Restoring a draft only selects files; it never submits the question.
    restoringFiles = true;
    try {
      input.files = selection.files;
      input.dispatchEvent(new Event("change", { bubbles: true }));
    } finally {
      restoringFiles = false;
    }
  }

  function onFocus(event) {
    sidebarHadFocus = Boolean(event.target.closest?.('[data-testid="stSidebar"]'));
    if (sidebarOverlayActive && !sidebarFocusRoots().some(root => root.contains(event.target))) {
      sidebarFocusableElements()[0]?.focus();
    }
  }

  function recoverSourceFocus() {
    const panel = [...document.querySelectorAll('[data-testid="stPopoverBody"]')]
      .find(element => element.querySelector('.dm-source-header') && element.getClientRects().length);
    if (panel === sourcePanel) return;
    if (!panel) {
      const opener = currentSourceLink() || currentSourceOpener();
      if (sourcePanel && opener) {
        opener.focus({ preventScroll: true });
        const main = document.getElementById('stMain');
        const bounds = opener.getBoundingClientRect();
        const visibleTop = Math.max(main?.getBoundingClientRect().top ?? 0,
          document.querySelector('[data-testid="stHeader"]')?.getBoundingClientRect().bottom ?? 0);
        const visibleBottom = document.querySelector('[data-testid="stBottom"]')?.getBoundingClientRect().top
          ?? window.innerHeight;
        // Fixed headers and composers overlap the scroller. Reveal only the
        // hidden edge, with room for the focus outline.
        if (main && bounds.top < visibleTop) {
          main.scrollTop += bounds.top - visibleTop - 8;
        } else if (main && bounds.bottom > visibleBottom) {
          main.scrollTop += bounds.bottom - visibleBottom + 8;
        }
      }
      sourcePanel = null;
      sourceLink = null;
      sourceOrigins.delete(parentElement);
      return;
    }
    sourcePanel = panel;
    if (!sourceOpener?.isConnected || sourceOpener.getAttribute('aria-expanded') !== 'true') {
      sourceOpener = null;
    }
    sourceOpener = currentSourceOpener();
    focusInput = false;
    panel.querySelector('.dm-source-title')?.focus({ preventScroll: true });
  }

  function closeSourcePanel() {
    const opener = currentSourceOpener();
    if (opener?.getAttribute('aria-expanded') === 'true') {
      opener.click();
    }
  }

  function currentSourceLink() {
    if (sourceLink?.isConnected) return sourceLink;
    // Source search reruns replace the original DOM link. Preserve its exact
    // occurrence because several sentences can cite the same source.
    const origin = sourceOrigins.get(parentElement);
    if (!origin) return null;
    return [...document.querySelectorAll('a[href^="#dm-citation-"]')]
      .filter(link => link.getAttribute("href") === origin.href)[origin.index] || null;
  }

  function currentSourceOpener() {
    // Streamlit keeps separate desktop/mobile buttons inside one popover.
    // Resizing hides the original without disconnecting it. Resolve its visible
    // sibling within the same popover: different answers can share a label.
    const popover = sourceOpener?.closest('[data-testid="stPopover"]');
    const candidates = popover
      ? popover.querySelectorAll('[data-testid="stPopoverButton"]')
      : document.querySelectorAll('[data-testid="stPopoverButton"][aria-expanded="true"]');
    return [...candidates].find(button => !button.disabled && button.getClientRects().length
      && getComputedStyle(button).visibility === 'visible');
  }

  function onSourceKeyDown(event) {
    if (!sourcePanel) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopImmediatePropagation();
      closeSourcePanel();
      return;
    }
    if (event.key !== 'Tab') return;
    const focusable = [...sourcePanel.querySelectorAll('a[href], button, input, select, textarea, summary, [tabindex]')]
      .filter(element => {
        // Closed details can retain layout boxes in Chromium; only their own
        // summary participates in keyboard navigation.
        const closed = element.closest('details:not([open])');
        if (closed && (!element.matches('summary') || element.parentElement !== closed
            || closed.parentElement.closest('details:not([open])'))) return false;
        return !element.disabled && element.tabIndex >= 0 && element.getClientRects().length
          && getComputedStyle(element).visibility === 'visible';
      });
    const first = focusable[0];
    const last = focusable.at(-1);
    const current = document.activeElement;
    if (!focusable.includes(current) || (event.shiftKey ? current === first : current === last)) {
      event.preventDefault();
      event.stopImmediatePropagation();
      (event.shiftKey ? last : first)?.focus();
    }
  }

  function recoverSidebarFocus() {
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    const main = document.getElementById("stMain");
    const open = sidebar?.getAttribute("data-collapsed") === "false";
    if (open) sidebar.inert = false;
    // Streamlit takes main out of the flex row in its mobile layout. Use that
    // layout mode: the sidebar's animated transform still looks closed at the
    // moment data-collapsed changes, so its current rectangle is not reliable.
    const overlay = open && main && getComputedStyle(main).position === "absolute";
    if (closeAfterNavigation) {
      closeAfterNavigation = false;
      if (overlay) {
        closeRequested = true;
        document.querySelector('[data-testid="stSidebarCollapseButton"] button')?.click();
        return;
      }
    }
    if (overlay) {
      sidebarOverlayActive = true;
      for (const element of [main, document.querySelector('[data-testid="stHeader"]'),
        document.querySelector('a[href="#stMain"]')].filter(Boolean)) {
        if (!sidebarBackground.has(element)) sidebarBackground.set(element, element.inert);
        element.inert = true;
      }
      if (!sidebarFocusRoots().some(root => root.contains(document.activeElement))) {
        sidebarFocusableElements()[0]?.focus();
      }
      // Opening a navigation menu takes precedence over composer restoration.
      focusInput = false;
      return;
    }
    releaseSidebarBackground();
    if (sidebar?.getAttribute("data-collapsed") === "true") {
      const focusedInside = sidebar.contains(document.activeElement);
      if (closeRequested || sidebarOverlayActive || focusedInside
          || (sidebarHadFocus && document.activeElement === document.body)) {
        document.querySelector('[data-testid="stExpandSidebarButton"]')?.focus();
        focusInput = false;
      }
      // CSS visibility can be overridden by descendant styles. Inert removes
      // the entire closed subtree from focus navigation and the AX tree.
      sidebar.inert = true;
      closeRequested = false;
      sidebarHadFocus = false;
    }
    sidebarOverlayActive = false;
    if (focusInput) {
      const input = document.querySelector(selector);
      if (input && !input.disabled) {
        input.focus({ preventScroll: true });
        focusInput = false;
      }
    }
  }

  function releaseSidebarBackground() {
    for (const [element, wasInert] of sidebarBackground) element.inert = wasInert;
    sidebarBackground.clear();
  }

  function sidebarFocusRoots() {
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (!sidebar) return [];
    // Selectboxes render their listboxes in a portal outside the sidebar.
    // Keep those owned popups usable without admitting background controls.
    const popups = [...sidebar.querySelectorAll('[aria-controls]')]
      .flatMap(element => element.getAttribute("aria-controls").split(/\s+/))
      .map(id => document.getElementById(id))
      .filter(element => element && !sidebar.contains(element));
    return [sidebar, ...popups];
  }

  function sidebarFocusableElements() {
    return sidebarFocusRoots().flatMap(root => [...root.querySelectorAll(
      'a[href], button, input, select, textarea, summary, [tabindex]'
    )]).filter(element => {
      if (element.disabled || element.tabIndex < 0 || element.closest('[inert]')
          || !element.getClientRects().length || getComputedStyle(element).visibility !== "visible") return false;
      if (element.matches('input[type="radio"]') && element.name && !element.checked) {
        return ![...element.form?.elements || document.querySelectorAll('input[type="radio"]')]
          .some(other => other.name === element.name && other.checked);
      }
      return true;
    });
  }

  function onSidebarKeyDown(event) {
    if (!sidebarOverlayActive) return;
    const roots = sidebarFocusRoots();
    // Escape first dismisses a selectbox's own popup. A second Escape closes
    // the sidebar, preserving the standard nested-widget keyboard behavior.
    if (event.key === "Escape" && roots.length === 1) {
      event.preventDefault();
      event.stopImmediatePropagation();
      closeRequested = true;
      document.querySelector('[data-testid="stSidebarCollapseButton"] button')?.click();
      return;
    }
    if (event.key !== "Tab") return;
    const focusable = sidebarFocusableElements();
    const first = focusable[0];
    const last = focusable.at(-1);
    const inside = roots.some(root => root.contains(document.activeElement));
    if (!inside || (event.shiftKey ? document.activeElement === first : document.activeElement === last)) {
      event.preventDefault();
      (event.shiftKey ? last : first)?.focus();
    }
  }

  let uploadSelection = null;

  function renderUploadSelection() {
    // These chips belong to the native picker, not the confirmed server
    // manifest. Observe its rendered selection without taking over uploads.
    const composer = document.querySelector(selector)?.closest('[data-testid="stChatInput"]');
    const chips = [...(composer?.querySelectorAll('[data-testid="stFileChip"]') || [])];
    const rejected = chips.filter(chip => chip.getAttribute("aria-invalid") === "true").length;
    const uploading = chips.filter(chip => chip.querySelector('[data-testid="stFileChipIconSpinner"]')).length;
    const waiting = chips.length - rejected - uploading;
    const send = composer?.querySelector('[data-testid="stChatInputSubmitButton"]');
    const input = document.querySelector(selector);
    if (send) {
      const label = data.chat_in_progress ? "답변을 기다리는 중 — 초안 작성 가능"
        : waiting ? (input.value.trim() ? "질문과 파일 보내기" : "파일 보내기") : "질문 보내기";
      if (send.getAttribute("aria-label") !== label) send.setAttribute("aria-label", label);
      if (send.getAttribute("title") !== label) send.setAttribute("title", label);
    }
    if (send && data.chat_in_progress) {
      send.dataset.documateChatBlocked = "true";
      if (!send.disabled) send.disabled = true;
    } else if (send?.dataset.documateChatBlocked) {
      delete send.dataset.documateChatBlocked;
      send.disabled = input.disabled || rejected > 0 || uploading > 0 || (input.value === "" && waiting === 0);
    }
    for (const control of composer?.querySelectorAll('input[type="file"], [data-testid="stChatInputFileUploadButton"] button') || []) {
      if (data.chat_in_progress) {
        control.dataset.documateChatBlocked = "true";
        if (!control.disabled) control.disabled = true;
      } else if (control.dataset.documateChatBlocked) {
        delete control.dataset.documateChatBlocked;
        control.disabled = input.disabled;
      }
    }
    if (send && rejected) {
      send.dataset.documateUploadBlocked = "true";
      if (!send.disabled) send.disabled = true;
    } else if (send?.dataset.documateUploadBlocked) {
      delete send.dataset.documateUploadBlocked;
      // Return control to the native picker after the user removes its errors.
      send.disabled = Boolean(data.chat_in_progress) || input.disabled || uploading > 0 || (input.value === "" && waiting === 0);
    }
    const selectedPrompts = document.querySelector('[class*="-selected_upload_prompts"]');
    const defaultPrompts = document.querySelector('[class*="-default_prompts"]');
    if (selectedPrompts && defaultPrompts) {
      const hasSelection = waiting > 0 || uploading > 0;
      if (selectedPrompts.dataset.selectedFiles !== String(hasSelection)) selectedPrompts.dataset.selectedFiles = String(hasSelection);
      if (selectedPrompts.hidden === hasSelection) selectedPrompts.hidden = !hasSelection;
      if (defaultPrompts.hidden !== hasSelection) defaultPrompts.hidden = hasSelection;
    }
    if (!composer) return;
    // The native composer lives in Streamlit's fixed bottom area. Own a small
    // sibling of its content, rather than moving a React-managed HTML element
    // out of the scrolling message area.
    if (uploadSelection?.parentElement !== composer) {
      uploadSelection?.remove();
      uploadSelection = document.createElement("p");
      uploadSelection.id = "documate-upload-selection";
      uploadSelection.setAttribute("role", "status");
      uploadSelection.setAttribute("aria-live", "polite");
      uploadSelection.setAttribute("aria-atomic", "true");
      composer.prepend(uploadSelection);
    }
    const states = [];
    if (rejected) states.push(`선택 오류 ${rejected}개 · 오류가 있는 파일을 제거하고 지원 형식·크기로 다시 선택하세요. 질문은 유지됩니다. 오류를 해결하면 전송할 수 있습니다.`);
    if (waiting) states.push(`전송 대기 ${waiting}개`);
    if (uploading) states.push(`파일 업로드 중 ${uploading}개`);
    const text = chips.length
      ? states.join(" · ") + (waiting || uploading ? " 아직 검색에 사용되지 않습니다." : "") + (waiting ? " 전송하면 자료를 준비합니다." : "")
      : "";
    if (uploadSelection.textContent !== text) uploadSelection.textContent = text;
    uploadSelection.dataset.invalid = String(rejected > 0);
    const describedBy = new Set((input.getAttribute("aria-describedby") || "").split(/\s+/).filter(Boolean));
    if (text) describedBy.add(uploadSelection.id);
    else describedBy.delete(uploadSelection.id);
    const description = [...describedBy].join(" ");
    if (description && input.getAttribute("aria-describedby") !== description) input.setAttribute("aria-describedby", description);
    else if (!description && input.hasAttribute("aria-describedby")) input.removeAttribute("aria-describedby");
  }

  function localizeControls() {
    for (const link of document.querySelectorAll('a[href^="#dm-citation-"]')) {
      const label = `근거 ${link.textContent.trim()} 열기`;
      if (link.getAttribute("aria-label") !== label) link.setAttribute("aria-label", label);
    }
    if (data.operation_id) {
      for (const button of document.querySelectorAll('[data-testid="stHeader"] button')) {
        if (button.textContent.trim() !== 'Stop' && !button.dataset.documateInterrupt) continue;
        button.dataset.documateInterrupt = 'true';
        const label = interruptRequested ? '수신 중단 중…' : '응답 수신 중단';
        if (button.textContent !== label) button.textContent = label;
        if (button.getAttribute('aria-label') !== label) button.setAttribute('aria-label', label);
        const hint = '화면의 응답 수신만 중단합니다. 서버 작업은 계속될 수 있습니다.';
        if (button.title !== hint) button.title = hint;
        if (interruptRequested && !button.disabled) button.disabled = true;
      }
    }
    const controls = [
      ['[data-testid="stChatInput"] input[type="file"]', "첨부할 파일 선택"],
      ['[data-testid="stChatInput"] button:is([aria-label="Upload files"], [aria-label="파일 첨부"])', "파일 첨부"],
      ['[data-testid="stExpandSidebarButton"]', "대화 메뉴 열기"],
      ['[data-testid="stSidebarCollapseButton"] button', "대화 메뉴 닫기"],
    ];
    for (const [selector, label] of controls) {
      for (const control of document.querySelectorAll(selector)) {
        if (control.getAttribute("aria-label") !== label) control.setAttribute("aria-label", label);
        if (control.getAttribute("title") !== label) control.setAttribute("title", label);
      }
    }
  }

  function connectRecipientFeedback() {
    const input = document.querySelector('input[aria-label="Slack 수신자"]');
    if (!input) return;
    const feedbackId = "documate-slack-recipient-help";
    const feedback = document.getElementById(feedbackId);
    const describedBy = new Set((input.getAttribute("aria-describedby") || "").split(/\s+/).filter(Boolean));
    if (feedback) describedBy.add(feedbackId);
    else describedBy.delete(feedbackId);
    const description = [...describedBy].join(" ");
    if (description && input.getAttribute("aria-describedby") !== description) input.setAttribute("aria-describedby", description);
    else if (!description) input.removeAttribute("aria-describedby");
    if (feedback) {
      const invalid = feedback.dataset.slackInvalid;
      if (input.getAttribute("aria-invalid") !== invalid) input.setAttribute("aria-invalid", invalid);
    } else input.removeAttribute("aria-invalid");
  }

  function blockUnavailableSubmission(event) {
    const composer = document.querySelector(selector)?.closest('[data-testid="stChatInput"]');
    if (!data.chat_in_progress && !composer?.querySelector('[data-testid="stFileChip"][aria-invalid="true"]')) return false;
    // Streamlit omits rejected files from its submit payload. Stop before its
    // handler clears the picker and draft, including keyboard submission.
    event.preventDefault();
    event.stopImmediatePropagation();
    storeDraft();
    renderUploadSelection();
    return true;
  }

  function revealUploadError() {
    const error = document.querySelector("[data-dm-upload-error]");
    if (!error || error.dataset.dmUploadError === focusedUploadError) return;
    const app = document.querySelector('[data-testid="stApp"]');
    if (app && app.getAttribute("data-test-script-state") !== "notRunning") return;
    focusInput = false;
    answerScrollPending = false;
    navigationScrollPending = false;
    clearTimeout(scrollTimer);
    const main = document.getElementById("stMain");
    main?.dispatchEvent(new WheelEvent("wheel"));
    error.focus({ preventScroll: true });
    if (main) {
      const visibleTop = Math.max(main.getBoundingClientRect().top,
        document.querySelector('[data-testid="stHeader"]')?.getBoundingClientRect().bottom || 0);
      main.scrollTop += error.getBoundingClientRect().top - visibleTop - 16;
    } else {
      error.scrollIntoView({ block: "start", behavior: "instant" });
    }
    focusedUploadError = error.dataset.dmUploadError;
    try {
      sessionStorage.setItem(uploadErrorKey, focusedUploadError);
    } catch {
      // Keep the once-per-run guard even when browser storage is unavailable.
    }
  }

  document.addEventListener("keydown", onKeyDown, true);
  document.addEventListener("pointerdown", onPointerDown, true);
  document.addEventListener("click", onClick, true);
  document.addEventListener("input", onInput, true);
  document.addEventListener("change", rememberSelectedFiles, true);
  document.addEventListener("drop", rememberSelectedFiles, true);
  document.addEventListener("paste", rememberSelectedFiles, true);
  document.addEventListener("click", forgetRemovedFile, true);
  document.addEventListener("focusin", onFocus, true);
  document.addEventListener("keydown", onSidebarKeyDown, true);
  document.addEventListener("keydown", onSourceKeyDown, true);
  window.addEventListener("resize", recoverSidebarFocus);
  document.addEventListener("wheel", onReadingIntent, { capture: true, passive: true });
  document.addEventListener("touchmove", onReadingIntent, { capture: true, passive: true });
  document.addEventListener("pointerdown", onReadingIntent, true);
  document.addEventListener("keydown", onReadingIntent, true);
  document.addEventListener("scroll", onMainScroll, true);
  window.addEventListener("pagehide", storeDraft);
  const observer = new MutationObserver(() => {
    localizeControls();
    connectRecipientFeedback();
    recoverSidebarFocus();
    recoverSourceFocus();
    restoreSelectedFiles();
    renderUploadSelection();
    revealUploadError();
    settleReadingScroll();
  });
  observer.observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ["data-collapsed", "aria-expanded", "aria-invalid", "data-slack-invalid", "disabled", "aria-label", "title", "data-test-script-state"] });
  localizeControls();
  connectRecipientFeedback();
  recoverSidebarFocus();
  recoverSourceFocus();
  restoreSelectedFiles();
  renderUploadSelection();
  revealUploadError();
  persist();
  persistReading();
  settleReadingScroll();

  let disposed = false;
  const cleanup = () => {
    // Streamlit can later unmount an already-replaced component. That stale
    // cleanup must not release state belonging to the current owner.
    if (disposed) return;
    disposed = true;
    document.removeEventListener("keydown", onKeyDown, true);
    document.removeEventListener("pointerdown", onPointerDown, true);
    document.removeEventListener("click", onClick, true);
    document.removeEventListener("input", onInput, true);
    document.removeEventListener("change", rememberSelectedFiles, true);
    document.removeEventListener("drop", rememberSelectedFiles, true);
    document.removeEventListener("paste", rememberSelectedFiles, true);
    document.removeEventListener("click", forgetRemovedFile, true);
    document.removeEventListener("focusin", onFocus, true);
    document.removeEventListener("keydown", onSidebarKeyDown, true);
    document.removeEventListener("keydown", onSourceKeyDown, true);
    window.removeEventListener("resize", recoverSidebarFocus);
    releaseSidebarBackground();
    document.removeEventListener("wheel", onReadingIntent, true);
    document.removeEventListener("touchmove", onReadingIntent, true);
    document.removeEventListener("pointerdown", onReadingIntent, true);
    document.removeEventListener("keydown", onReadingIntent, true);
    document.removeEventListener("scroll", onMainScroll, true);
    window.removeEventListener("pagehide", storeDraft);
    observer.disconnect();
    uploadSelection?.remove();
    clearTimeout(scrollTimer);
    cancelAnimationFrame(readingFrame);
    cancelAnimationFrame(sourceFrame);
    if (document[bridgeCleanupKey] === cleanup) delete document[bridgeCleanupKey];
  };
  document[bridgeCleanupKey] = cleanup;
  return cleanup;
}
