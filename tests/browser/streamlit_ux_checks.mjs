/** Browser assertions for the real UI (including streamlit_fixture.py).
 * Pass the browser skill's tab handle; all page evaluation is read-only.
 * These complement AppTest, which cannot execute CSS, focus, or unsent typing.
 */
function assert(condition, message) {
  if (!condition) throw new Error(message);
}

/** Observe the actual native file picker; no simulated upload state. */
export async function attachmentSelectionState(tab, { waiting, ready, rejected = 0 }) {
  const summary = await tab.playwright.evaluate(() => ({
    chips: document.querySelectorAll('[data-testid="stChatInput"] [data-testid="stFileChip"]').length,
    waiting: document.querySelector('[role="status"][id="documate-upload-selection"]')?.textContent || "",
    ready: document.querySelector('.dm-upload-summary')?.textContent || "",
  }));
  assert(summary.chips === waiting + rejected, `Unexpected native file selection: ${JSON.stringify(summary)}`);
  if (rejected) {
    assert(summary.waiting.includes(`선택 오류 ${rejected}개`),
      `Rejected file is presented as ready to send: ${JSON.stringify(summary)}`);
  }
  assert(summary.waiting.includes(waiting ? `전송 대기 ${waiting}개` : rejected ? "파일 옆의 오류" : "선택한 파일은 전송 후"),
    `Selected files lack a distinct waiting state: ${JSON.stringify(summary)}`);
  assert(summary.ready.includes(`검색 가능 ${ready}개`),
    `Server-ready files are not distinguished: ${JSON.stringify(summary)}`);
  return { check: "selected versus searchable attachments", ...summary };
}

export async function composerText(tab, expected) {
  const actual = await tab.playwright.getByRole("textbox", {
    name: "공식 문서나 업로드한 자료에 대해 질문하세요", exact: true,
  }).evaluate(el => el.value);
  assert(actual === expected, `Composer text mismatch: ${JSON.stringify({ expected, actual })}`);
  return { check: "composer text", value: actual };
}

/** Run after leaving and returning to a conversation with unsubmitted files. */
export async function conversationDraftSelection(tab, { draft, filenames, messages = 0 }) {
  await composerText(tab, draft);
  const selection = await tab.playwright.evaluate(() => ({
    filenames: [...document.querySelectorAll('[data-testid="stChatInput"] [data-testid="stFileChip"] button[aria-label^="Remove "]')]
      .map(button => button.getAttribute("aria-label").slice("Remove ".length)),
    messages: document.querySelectorAll('[data-testid="stChatMessage"]').length,
    sendDisabled: document.querySelector('[data-testid="stChatInputSubmitButton"]')?.disabled,
  }));
  assert(JSON.stringify(selection.filenames.toSorted()) === JSON.stringify(filenames.toSorted()),
    `Conversation lost or mixed its unsent files: ${JSON.stringify({ expected: filenames, actual: selection.filenames })}`);
  assert(selection.messages === messages, "Navigation submitted a question or changed the conversation history");
  assert(selection.sendDisabled === (draft === "" && filenames.length === 0),
    `Restored draft has the wrong submission state: ${JSON.stringify(selection)}`);
  return { check: "conversation retains exact question and selected files", ...selection };
}

/** Check visible size/status as well as names, including same-name selections. */
export async function attachmentSelectionDetails(tab, expected) {
  const actual = await tab.playwright.evaluate(() =>
    [...document.querySelectorAll('[data-testid="stChatInput"] [data-testid="stFileChip"]')].map(chip => {
      const name = chip.querySelector('[data-testid="stFileChipName"]').title;
      return {
        name,
        size: chip.getAttribute("aria-label").slice(name.length + 2),
        rejected: chip.getAttribute("aria-invalid") === "true",
      };
    }));
  assert(actual.length === expected.length && actual.every((file, index) =>
    file.name === expected[index].name && file.size === expected[index].size
      && file.rejected === expected[index].rejected),
    `Restored attachment metadata differs: ${JSON.stringify({ expected, actual })}`);
  return { check: "selected attachment names, sizes and validation status", files: actual };
}

/** Run with a real rejected file selected, before removing/replacing it. */
export async function rejectedAttachmentPreservesDraft(tab, draft) {
  const input = tab.playwright.getByRole("textbox", {
    name: "공식 문서나 업로드한 자료에 대해 질문하세요", exact: true,
  });
  await input.fill(draft);
  const messagesBefore = await tab.playwright.evaluate(() =>
    document.querySelectorAll('[data-testid="stChatMessage"]').length);
  await input.press("Enter");
  await tab.getAXState({ emit: false });
  await composerText(tab, draft);
  const state = await tab.playwright.evaluate(() => ({
    rejected: document.querySelectorAll('[data-testid="stChatInput"] [data-testid="stFileChip"][aria-invalid="true"]').length,
    messages: document.querySelectorAll('[data-testid="stChatMessage"]').length,
    sendDisabled: document.querySelector('[data-testid="stChatInputSubmitButton"]')?.disabled,
    notice: document.getElementById("documate-upload-selection")?.textContent || "",
  }));
  assert(state.rejected > 0, "Rejected attachment disappeared on attempted submission");
  assert(state.messages === messagesBefore, "Question was sent without its rejected attachment");
  assert(state.sendDisabled, "Send remains enabled with a rejected attachment");
  assert(state.notice.includes("전송할 수 없습니다"), "Rejected attachment lacks a recovery instruction");
  return { check: "rejected attachment blocks submission and preserves draft", ...state };
}

/** Remove an actual rejected selection and verify the same draft can continue. */
export async function rejectedAttachmentRecovery(tab, { filename, draft, remaining = 0 }) {
  await tab.playwright.getByRole("button", { name: `Remove ${filename}`, exact: true }).click();
  await tab.getAXState({ emit: false });
  await composerText(tab, draft);
  const chips = await tab.playwright.getByTestId("stChatInput").getByTestId("stFileChip").count();
  const enabled = await tab.playwright.getByRole("button", { name: "Send message", exact: true }).isEnabled();
  assert(chips === remaining, `Recovery lost or retained unexpected attachments: ${chips}`);
  assert(enabled === (draft !== "" || remaining > 0), "Send did not recover after removing the rejected file");
  return { check: "rejected attachment removal preserves draft and restores submission", chips, enabled };
}

export async function answerCompleted(tab, answerText) {
  const answer = tab.playwright.getByText(answerText, { exact: true });
  assert(await answer.isVisible(), `Final answer not displayed: ${answerText}`);
  const composer = tab.playwright.getByRole("textbox", {
    name: "공식 문서나 업로드한 자료에 대해 질문하세요", exact: true,
  });
  assert(await composer.isEnabled(), "Composer stayed busy after the answer");
  return { check: "answer completed", answerText };
}

/** Use after a real attachment has produced an answer. No blur before reload. */
export async function reloadPreservesConversation(tab, { draft, answerText, attachmentText }) {
  await answerCompleted(tab, answerText);
  const attachment = tab.playwright.getByText(attachmentText, { exact: false }).first();
  assert(await attachment.isVisible(), "Confirmed attachment is not displayed before reload");
  const messageCount = await tab.playwright.evaluate(() =>
    document.querySelectorAll('[data-testid="stChatMessage"]').length);
  await tab.playwright.getByRole("textbox", {
    name: "공식 문서나 업로드한 자료에 대해 질문하세요", exact: true,
  }).fill(draft);
  await tab.reload();
  await tab.playwright.getByText(answerText, { exact: true }).waitFor({ state: "visible", timeoutMs: 20000 });
  await tab.getAXState({ emit: false });
  await composerText(tab, draft);
  await answerCompleted(tab, answerText);
  assert(await attachment.isVisible(), "Confirmed attachment disappeared on reload");
  const restoredCount = await tab.playwright.evaluate(() =>
    document.querySelectorAll('[data-testid="stChatMessage"]').length);
  assert(restoredCount === messageCount, "Reload lost or duplicated chat messages");
  return { check: "reload preserves answer, attachment and exact unsent draft", messages: restoredCount };
}

/** Run on a new conversation. AppTest cannot reproduce just-typed browser text. */
export async function suggestionDraftProtection(tab, suggestionText) {
  const input = tab.playwright.getByRole("textbox", {
    name: "공식 문서나 업로드한 자료에 대해 질문하세요", exact: true,
  });
  const suggestion = tab.playwright.getByRole("button", { name: suggestionText, exact: true });
  const draft = "  작성 중인 내 질문\n두 번째 줄  ";
  await composerText(tab, "");
  await input.fill(draft);
  await suggestion.click();
  await tab.getAXState({ emit: false });
  await composerText(tab, draft);
  await input.fill("");
  await suggestion.click();
  await tab.getAXState({ emit: false });
  await composerText(tab, suggestionText);
  const messages = await tab.playwright.evaluate(() =>
    document.querySelectorAll('[data-testid="stChatMessage"]').length);
  assert(messages === 0, "Choosing an example submitted a question");
  return { check: "suggestion protects typed draft and works after clearing", messages };
}

export async function partialAnswer(tab, { bodyText, globalLimits, unitLimits }) {
  assert(await tab.playwright.getByText(bodyText, { exact: false }).first().isVisible(),
    "Partial answer body is not visible");
  assert(await tab.playwright.getByText("부분 답변 · 요청 전체를 완성하지 못했습니다.", { exact: false }).isVisible(),
    "Partial answer summary is not visible");

  for (const [label, limits] of [
    ["답변의 한계와 처리 정보", globalLimits],
    [`이 내용의 한계 (${unitLimits.length}개)`, unitLimits],
  ]) {
    const details = await tab.playwright.getByText(label, { exact: true }).evaluate(el => {
      const container = el.closest("details");
      return { exists: Boolean(container), open: container?.open, text: container?.textContent || "" };
    });
    assert(details.exists && !details.open, `Limit details should start collapsed: ${label}`);
    for (const limit of limits) {
      assert(details.text.includes(limit), `Limit detail was lost: ${limit}`);
    }
  }
  return { check: "partial answer hierarchy", bodyVisible: true, detailsCollapsed: true,
    preservedLimits: globalLimits.length + unitLimits.length };
}

export async function readableText(tab, text) {
  const measured = await tab.playwright.getByText(text, { exact: true }).first().evaluate(el => {
    const color = getComputedStyle(el).color;
    let background = "";
    for (let parent = el; parent; parent = parent.parentElement) {
      const candidate = getComputedStyle(parent).backgroundColor;
      if (candidate !== "rgba(0, 0, 0, 0)" && candidate !== "transparent") {
        background = candidate;
        break;
      }
    }
    function luminance(css) {
      return css.match(/[\d.]+/g).slice(0, 3).map(Number).map(v => {
        const s = v / 255;
        return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
      }).reduce((sum, v, i) => sum + v * [0.2126, 0.7152, 0.0722][i], 0);
    }
    const a = luminance(color), b = luminance(background);
    return { color, background, contrast: (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05) };
  });
  assert(measured.contrast >= 4.5, `Unreadable text ${text}: ${JSON.stringify(measured)}`);
  return { check: "text contrast", text, ...measured };
}

/** Measure rendered code text against its actual opaque block background. */
export async function codeContrast(tab) {
  const measured = await tab.playwright.getByTestId("stCode").evaluateAll(blocks => {
    function luminance(css) {
      return css.match(/[\d.]+/g).slice(0, 3).map(Number).map(v => {
        const s = v / 255;
        return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
      }).reduce((sum, v, i) => sum + v * [0.2126, 0.7152, 0.0722][i], 0);
    }
    return blocks.flatMap(block => Array.from(block.querySelectorAll("pre, code, span"))
      .filter(el => el.getClientRects().length > 0
        && Array.from(el.childNodes).some(node => node.nodeType === 3 && node.textContent.trim()))
      .map(el => {
        const color = getComputedStyle(el).color;
        let background = "";
        for (let parent = el; parent; parent = parent.parentElement) {
          const candidate = getComputedStyle(parent).backgroundColor;
          if (candidate !== "rgba(0, 0, 0, 0)" && candidate !== "transparent") {
            background = candidate;
            break;
          }
        }
        const a = luminance(color), b = luminance(background);
        return { color, background, contrast: (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05) };
      }));
  });
  assert(measured.length > 0, "No rendered code text to measure");
  const worst = measured.reduce((a, b) => a.contrast < b.contrast ? a : b);
  assert(worst.contrast >= 4.5, `Unreadable code: ${JSON.stringify(worst)}`);
  return { check: "rendered code contrast", samples: measured.length, minimum: worst.contrast,
    colors: [...new Set(measured.map(sample => sample.color))] };
}

export async function visibleFocus(tab) {
  const focused = await tab.playwright.evaluate(() => {
    const el = document.activeElement;
    const rect = el.getBoundingClientRect();
    return {
      tag: el.tagName, label: el.getAttribute("aria-label") || el.textContent?.slice(0, 80),
      visibility: getComputedStyle(el).visibility,
      inClosedSidebar: Boolean(el.closest('[data-testid="stSidebar"][data-collapsed="true"]')),
      onScreen: rect.right > 0 && rect.left < innerWidth && rect.bottom > 0 && rect.top < innerHeight,
    };
  });
  assert(!focused.inClosedSidebar, `Focus entered closed sidebar: ${JSON.stringify(focused)}`);
  assert(focused.tag === "BODY" || (focused.visibility === "visible" && focused.onScreen),
    `Invisible focus: ${JSON.stringify(focused)}`);
  return { check: "visible keyboard focus", ...focused };
}

export async function sidebarOpenerFocused(tab) {
  const focused = await tab.playwright.evaluate(() =>
    document.activeElement?.getAttribute("data-testid") === "stExpandSidebarButton");
  assert(focused, "Closing sidebar did not return focus to its opener");
  return { check: "sidebar focus return", passed: true };
}

/** The header must own space outside the scrollable reading area. */
export async function mobileToolbarLayout(tab) {
  const layout = await tab.playwright.evaluate(() => {
    const header = document.querySelector('[data-testid="stHeader"]');
    const main = document.querySelector('#stMain');
    return {
      headerBottom: header.getBoundingClientRect().bottom,
      mainTop: main.getBoundingClientRect().top,
      mainWidth: main.clientWidth, scrollWidth: main.scrollWidth,
    };
  });
  assert(layout.mainTop >= layout.headerBottom - 1,
    `Toolbar overlaps the reading area: ${JSON.stringify(layout)}`);
  assert(layout.scrollWidth <= layout.mainWidth + 1,
    `Reading area overflows horizontally: ${JSON.stringify(layout)}`);
  return { check: "mobile toolbar outside reading area", ...layout };
}

export async function answerStartVisible(tab, text) {
  const position = await tab.playwright.locator('[data-testid="stChatMessage"] p').evaluateAll((elements, prefix) => {
    const el = elements.filter(element => element.textContent.startsWith(prefix)).at(-1);
    const rect = el.getBoundingClientRect();
    const main = document.querySelector('#stMain').getBoundingClientRect();
    const composer = document.querySelector('[data-testid="stChatInput"]').getBoundingClientRect();
    return { top: rect.top, mainTop: main.top, bottom: composer.top };
  }, text);
  assert(position.top >= position.mainTop - 1 && position.top < position.bottom,
    `Completed answer starts outside the reading area: ${JSON.stringify(position)}`);
  return { check: "completed answer start visible", ...position };
}

export async function readingPosition(tab, previousTop) {
  const top = await tab.playwright.evaluate(() => document.querySelector('#stMain').scrollTop);
  if (previousTop !== undefined) {
    assert(Math.abs(top - previousTop) < 4, `Reading position changed from ${previousTop} to ${top}`);
  }
  return top;
}

export async function parameterEvidence(tab) {
  const details = await tab.playwright.evaluate(() => {
    const source = document.querySelector('.dm-source-markdown');
    return {
      width: source.clientWidth, scrollWidth: source.scrollWidth,
      emphasized: [...source.querySelectorAll('strong')].map(el => el.textContent),
      code: [...source.querySelectorAll('pre')].map(el => el.textContent),
    };
  });
  assert(details.emphasized.includes('order') && details.emphasized.includes('copy'),
    `Parameter definitions lost their formatting: ${JSON.stringify(details)}`);
  assert(!details.code.some(text => text.includes('**order**')),
    `Parameter prose is rendered as code: ${JSON.stringify(details)}`);
  assert(details.code.some(text => text.includes('values.reshape((2, 3))')),
    "Real code example was lost");
  assert(details.scrollWidth <= details.width + 1,
    `Evidence prose overflows horizontally: ${JSON.stringify(details)}`);
  return { check: "wrapped parameter definitions and preserved code", ...details };
}
