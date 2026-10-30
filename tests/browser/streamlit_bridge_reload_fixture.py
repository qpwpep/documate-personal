"""Exercise the real component hot-reload boundary with bounded native observers.

Run with ``uv run python -m streamlit run
tests/browser/streamlit_bridge_reload_fixture.py --server.port 8503``.
An optional DOCUMATE_BRIDGE_TEST_SOURCE points to a pre-fix JS copy. Reload the
browser between source versions. No model, upload API, or external service runs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
from streamlit.components.v2 import component


# Native observers deliver queued mutations, unlike the simple DOM boundary in
# most Node tests. Stop a runaway burst so the diagnostic tab remains readable.
DIAGNOSTICS = r"""
const diagnosticKey = Symbol.for("documate.bridge-reload-diagnostics");
const diagnostic = document[diagnosticKey] ??= {
  observers: new Set(), deliveries: 0, burst: 0, stopped: false, labels: [],
  report() {
    const target = document.getElementById("bridge-reload-report");
    if (!target) return;
    target.textContent = JSON.stringify({
      activeObservers: this.observers.size, deliveries: this.deliveries,
      runawayStopped: this.stopped, recentLabels: this.labels,
    }, null, 2);
  },
};
const NativeMutationObserver = window.MutationObserver;
class MutationObserver {
  constructor(callback) {
    this.native = new NativeMutationObserver((records, observer) => {
      diagnostic.deliveries += 1;
      if (diagnostic.burst++ === 0) setTimeout(() => { diagnostic.burst = 0; }, 0);
      const label = document.querySelector('[data-testid="stChatInputSubmitButton"]')?.getAttribute("aria-label");
      if (label !== diagnostic.labels.at(-1)) {
        diagnostic.labels.push(label);
        diagnostic.labels = diagnostic.labels.slice(-12);
      }
      if (diagnostic.burst > 250) {
        diagnostic.stopped = true;
        const activeBeforeStop = diagnostic.observers.size;
        for (const item of [...diagnostic.observers]) item.disconnect();
        console.error("BRIDGE_RELOAD_RUNAWAY", JSON.stringify({activeBeforeStop, deliveries: diagnostic.deliveries, labels: diagnostic.labels}));
        diagnostic.report();
        return;
      }
      callback(records, observer);
    });
  }
  observe(target, options) {
    if (diagnostic.stopped) return;
    diagnostic.observers.add(this);
    this.native.observe(target, options);
  }
  disconnect() {
    diagnostic.observers.delete(this);
    this.native.disconnect();
  }
}
if (!diagnostic.reportListener) {
  diagnostic.reportListener = event => {
    if (event.target.closest?.("#bridge-reload-report-button")) diagnostic.report();
  };
  document.addEventListener("click", diagnostic.reportListener, true);
}
"""


def main() -> None:
    st.set_page_config(page_title="브리지 코드 교체 재현", layout="centered")
    st.title("브리지 코드 교체 재현")
    st.caption("실제 Streamlit 컴포넌트와 전송 입력창을 사용합니다. 반복이 감지되면 감시자를 중단해 탭을 보호합니다.")
    st.session_state.setdefault("generation", 0)
    st.session_state.setdefault("busy", False)
    st.session_state.setdefault("submitted", "")
    if st.button("JS 코드 교체"):
        st.session_state.generation += 1
    if st.button("응답 완료로 전환"):
        st.session_state.busy = False
    st.write(f"코드 세대: {st.session_state.generation} · 답변 대기: {st.session_state.busy}")
    if st.session_state.submitted:
        st.info(f"수신한 질문: {st.session_state.submitted}")
    st.html('<button id="bridge-reload-report-button">감시자 진단 보기</button><pre id="bridge-reload-report">진단 대기</pre>')

    conversation_id = "bridge-reload-fixture"
    prompt = st.chat_input("재현용 질문", key=f"conversation-{conversation_id}-chat_input")
    if prompt:
        st.session_state.submitted = prompt
        st.session_state.busy = True
        st.rerun()

    source = Path(os.environ.get("DOCUMATE_BRIDGE_TEST_SOURCE", "src/app/web/streamlit_browser_state.js"))
    script = DIAGNOSTICS + source.read_text(encoding="utf-8") + f"\n// fixture generation {st.session_state.generation}\n"
    bridge = component("documate_bridge_reload_fixture", js=script)
    bridge(data={
        "conversation_id": conversation_id, "revision": 0, "text": "",
        "focus_input": False, "pending_chat": st.session_state.busy,
        "chat_in_progress": st.session_state.busy, "message_count": 0,
        "snapshot": json.dumps({"version": 1, "active_conversation_id": conversation_id,
                                "conversations": [{"id": conversation_id, "messages": [],
                                                   "draft": {"text": "", "revision": 0}}]}),
    }, key="reload-bridge", height=0)


if __name__ == "__main__":
    main()
