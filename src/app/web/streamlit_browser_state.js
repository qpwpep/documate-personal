// Keep Streamlit's own composer, upload handling, and public prefill mechanism.
export default function ({ data, setStateValue }) {
  const storageKey = "documate.conversations.v1";
  const storageError = "브라우저에 대화를 저장하지 못했습니다. 새로고침하면 최근 대화와 초안을 잃을 수 있습니다.";
  if (data.restore) {
    try {
      setStateValue("restore", { snapshot: sessionStorage.getItem(storageKey), error: null });
    } catch {
      setStateValue("restore", { snapshot: null, error: storageError });
    }
    return;
  }

  const snapshot = JSON.parse(data.snapshot);
  try {
    const previous = JSON.parse(sessionStorage.getItem(storageKey) || "null");
    if (previous?.version === 1 && Array.isArray(previous.conversations)) {
      for (const conversation of snapshot.conversations) {
        const draft = previous.conversations.find(item => item.id === conversation.id)?.draft;
        // Python advances the revision on submit/edit. An unchanged revision
        // must not overwrite text just typed in the native browser input.
        if (typeof draft?.text === "string" && Number.isInteger(draft.revision)
            && draft.revision >= conversation.draft.revision) {
          conversation.draft = draft;
        }
      }
    }
  } catch {
    // Bootstrap validates stored JSON. A later unreadable value is replaced
    // with the current server-rendered display state, if storage allows it.
  }
  const storedConversation = snapshot.conversations.find(item => item.id === data.conversation_id);
  const selector = `.st-key-conversation-${data.conversation_id}-chat_input textarea`;
  let lastSent = data.text;
  let submitting = false;
  let closeRequested = false;
  let sidebarHadFocus = false;
  let focusInput = data.focus_input;

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
    submitting = true;
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
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing
        && event.target.matches?.(selector) && event.target.value.trim()) {
      beginSubmission();
    }
  }

  function onPointerDown(event) {
    const send = event.target.closest?.('[data-testid="stChatInputSubmitButton"]');
    if (send && !send.disabled) beginSubmission();
    // Flush before a navigation trigger; that trigger then carries this widget
    // state even if Streamlit coalesces the two rerun requests.
    if (!event.target.closest?.('[data-testid="stChatInput"]')) flushDraft();
    if (event.target.closest?.('[data-testid="stSidebarCollapseButton"]')) {
      closeRequested = true;
    }
  }

  function onClick(event) {
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
    if (event.target.matches?.(selector)) storeDraft();
    // Re-enable examples as soon as the user clears a draft, without rerunning
    // on every keystroke. The submit latch excludes submission-driven clearing.
    if (event.target.matches?.(selector) && event.target.value === "") flushDraft();
  }

  function onFocus(event) {
    sidebarHadFocus = Boolean(event.target.closest?.('[data-testid="stSidebar"]'));
  }

  function recoverSidebarFocus() {
    if (focusInput) {
      const input = document.querySelector(selector);
      if (input && !input.disabled) {
        input.focus();
        focusInput = false;
      }
    }
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (sidebar?.getAttribute("data-collapsed") !== "true") return;
    const focusedInside = sidebar.contains(document.activeElement);
    if (closeRequested || focusedInside || (sidebarHadFocus && document.activeElement === document.body)) {
      document.querySelector('[data-testid="stExpandSidebarButton"]')?.focus();
    }
    closeRequested = false;
    sidebarHadFocus = false;
  }

  function renderUploadSelection() {
    const status = document.getElementById("documate-upload-selection");
    if (!status) return;
    // These chips belong to the native picker, not the confirmed server
    // manifest. Observe its rendered selection without taking over uploads.
    const composer = document.querySelector(selector)?.closest('[data-testid="stChatInput"]');
    const chips = [...(composer?.querySelectorAll('[data-testid="stFileChip"]') || [])];
    const rejected = chips.filter(chip => chip.getAttribute("aria-invalid") === "true").length;
    const uploading = chips.filter(chip => chip.querySelector('[data-testid="stFileChipIconSpinner"]')).length;
    const waiting = chips.length - rejected - uploading;
    const states = [];
    if (waiting) states.push(`전송 대기 ${waiting}개`);
    if (uploading) states.push(`파일 업로드 중 ${uploading}개`);
    if (rejected) states.push(`선택 오류 ${rejected}개 · 파일 옆의 오류를 확인하고 해당 파일을 제거하거나 다시 선택하세요.`);
    const text = chips.length
      ? states.join(" · ") + " 아직 검색에 사용되지 않습니다." + (waiting ? " 전송하면 자료를 준비합니다." : "")
      : "선택한 파일은 전송 후 검색에 사용됩니다. 파일만 먼저 전송할 수도 있습니다.";
    if (status.textContent !== text) status.textContent = text;
  }

  document.addEventListener("keydown", onKeyDown, true);
  document.addEventListener("pointerdown", onPointerDown, true);
  document.addEventListener("click", onClick, true);
  document.addEventListener("input", onInput, true);
  document.addEventListener("focusin", onFocus, true);
  window.addEventListener("pagehide", storeDraft);
  const observer = new MutationObserver(() => {
    recoverSidebarFocus();
    renderUploadSelection();
  });
  observer.observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ["data-collapsed", "aria-invalid"] });
  recoverSidebarFocus();
  renderUploadSelection();
  persist();

  return () => {
    document.removeEventListener("keydown", onKeyDown, true);
    document.removeEventListener("pointerdown", onPointerDown, true);
    document.removeEventListener("click", onClick, true);
    document.removeEventListener("input", onInput, true);
    document.removeEventListener("focusin", onFocus, true);
    window.removeEventListener("pagehide", storeDraft);
    observer.disconnect();
  };
}
