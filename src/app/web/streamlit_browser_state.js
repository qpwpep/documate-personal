// Keep Streamlit's own composer, upload handling, and public prefill mechanism.
export default function ({ data, setStateValue }) {
  const selector = `.st-key-conversation-${data.conversation_id}-chat_input textarea`;
  let lastSent = data.text;
  let submitting = false;
  let closeRequested = false;
  let sidebarHadFocus = false;
  let focusInput = data.focus_input;

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
      submitting = true;
    }
  }

  function onPointerDown(event) {
    const send = event.target.closest?.('[data-testid="stChatInputSubmitButton"]');
    if (send && !send.disabled) submitting = true;
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

  document.addEventListener("keydown", onKeyDown, true);
  document.addEventListener("pointerdown", onPointerDown, true);
  document.addEventListener("click", onClick, true);
  document.addEventListener("input", onInput, true);
  document.addEventListener("focusin", onFocus, true);
  const observer = new MutationObserver(recoverSidebarFocus);
  observer.observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ["data-collapsed"] });
  recoverSidebarFocus();

  return () => {
    document.removeEventListener("keydown", onKeyDown, true);
    document.removeEventListener("pointerdown", onPointerDown, true);
    document.removeEventListener("click", onClick, true);
    document.removeEventListener("input", onInput, true);
    document.removeEventListener("focusin", onFocus, true);
    observer.disconnect();
  };
}
