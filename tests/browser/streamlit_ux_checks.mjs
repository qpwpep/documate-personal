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
  assert(summary.waiting.includes(waiting ? `전송 대기 ${waiting}개` : rejected ? "파일 옆의 오류" : "새 파일은 +로 추가"),
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

/** Run with the code-context fixture's evidence dialog open. */
export async function literalCodeContext(tab, expectedCode) {
  const context = await tab.playwright.locator(".dm-source-answer-context").evaluate(el => ({
    code: [...el.querySelectorAll("code")].map(item => item.textContent),
    headings: el.querySelectorAll("h1,h2,h3,h4,h5,h6").length,
    codeBlocks: [...el.querySelectorAll("pre")].map(block => ({
      width: block.clientWidth, scrollWidth: block.scrollWidth,
    })),
  }));
  assert(JSON.stringify(context.code) === JSON.stringify(expectedCode),
    `Code context changed literal text: ${JSON.stringify(context)}`);
  assert(context.headings === 0, "Code comments became Markdown headings");
  assert(context.codeBlocks.every(block => block.scrollWidth <= block.width + 1),
    `Code context needs horizontal scrolling: ${JSON.stringify(context.codeBlocks)}`);
  return { check: "literal block and inline code context", ...context };
}

/** Run after an interrupted question is followed by a completed answer. */
export async function pastInterruptionGuidance(tab, completedAnswer) {
  const state = await tab.playwright.evaluate(() => ({
    text: document.querySelector("main")?.textContent || "",
    editButtons: [...document.querySelectorAll("button")]
      .filter(button => button.textContent.trim() === "질문 수정").length,
  }));
  assert(state.text.includes("응답 수신이 중단되었습니다."), "Interrupted history disappeared");
  assert(state.text.includes(completedAnswer), "The following answer has not completed");
  assert(!state.text.includes("아래 ‘질문 수정’") && state.editButtons === 0,
    "Past interruption still points to an unavailable edit action");
  return { check: "past interruption retains warning without stale recovery guidance", passed: true };
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
  const enabled = await tab.playwright.getByRole("button", { name: "질문 보내기", exact: true }).isEnabled();
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

/** Run after opening evidence at 320px and 390px, including a long source. */
export async function evidencePopoverLayout(tab) {
  const layout = await tab.playwright.getByRole("dialog").evaluate(el => {
    const rect = el.getBoundingClientRect();
    const title = el.querySelector('.dm-source-title').getBoundingClientRect();
    return {
      left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom,
      viewportWidth: innerWidth, viewportHeight: innerHeight,
      titleVisible: title.top >= rect.top && title.bottom <= rect.bottom,
      width: el.clientWidth, scrollWidth: el.scrollWidth,
      detailsCollapsed: [...el.querySelectorAll('details')].every(details => !details.open),
    };
  });
  assert(layout.left >= 0 && layout.right <= layout.viewportWidth
    && layout.top >= 0 && layout.bottom <= layout.viewportHeight && layout.titleVisible,
    `Evidence panel is clipped by the viewport: ${JSON.stringify(layout)}`);
  assert(layout.scrollWidth <= layout.width + 1,
    `Evidence panel overflows horizontally: ${JSON.stringify(layout)}`);
  assert(layout.detailsCollapsed, "Collection metadata or raw source should start collapsed");
  return { check: "evidence panel fits viewport with collapsed metadata", ...layout };
}

/** Run after opening a source with Enter and after Tab / Shift+Tab. */
export async function evidenceKeyboardFocus(tab, expectedFocus) {
  const state = await tab.playwright.getByRole("dialog").evaluate(panel => {
    const close = panel.querySelector('.dm-source-close');
    const rect = close?.getBoundingClientRect();
    return {
      focusInside: panel.contains(document.activeElement),
      focused: document.activeElement?.textContent?.trim().slice(0, 100),
      context: panel.querySelector('.dm-source-answer-context')?.textContent || "",
      closeLabel: close?.getAttribute('aria-label'),
      closeWidth: rect?.width, closeHeight: rect?.height,
    };
  });
  assert(state.focusInside, `Source keyboard focus escaped to the background: ${JSON.stringify(state)}`);
  if (expectedFocus) {
    assert(state.focused?.endsWith(expectedFocus),
      `Source keyboard focus did not reach ${expectedFocus}: ${JSON.stringify(state)}`);
  }
  assert(state.context.includes("이 근거가 연결된 답변"), "Source panel lost the associated answer context");
  assert(state.closeLabel === "근거 닫기" && state.closeWidth >= 44 && state.closeHeight >= 44,
    `Source close control is missing or too small: ${JSON.stringify(state)}`);
  return { check: "source context, close target and keyboard focus", ...state };
}

/** Closing by Escape or the close button returns to the actual citation opener. */
export async function evidenceClosedFocus(tab, expectedOpener, expectedPopoverIndex) {
  const state = await tab.playwright.evaluate(() => {
    const focused = document.activeElement;
    const rect = focused.getBoundingClientRect();
    const covering = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
    const readingTop = Math.max(document.getElementById('stMain').getBoundingClientRect().top,
      document.querySelector('[data-testid="stHeader"]')?.getBoundingClientRect().bottom ?? 0);
    const readingBottom = document.querySelector('[data-testid="stBottom"]')?.getBoundingClientRect().top
      ?? window.innerHeight;
    return {
      openPanels: [...document.querySelectorAll('[data-testid="stPopoverBody"]')]
        .filter(panel => panel.getClientRects().length).length,
      openerFocused: focused.getAttribute('data-testid') === 'stPopoverButton',
      expanded: focused.getAttribute('aria-expanded'),
      openerText: focused.querySelector('[data-testid="stMarkdownContainer"]')?.textContent?.trim(),
      openerVisible: Boolean(focused.getClientRects().length && getComputedStyle(focused).visibility === 'visible'),
      popoverIndex: [...document.querySelectorAll('[data-testid="stPopover"]')]
        .indexOf(focused.closest('[data-testid="stPopover"]')),
      openerUncovered: covering === focused || focused.contains(covering),
      openerWithinReadingArea: rect.top >= readingTop && rect.bottom <= readingBottom,
      verticalBounds: { top: rect.top, bottom: rect.bottom, readingTop, readingBottom },
    };
  });
  assert(state.openPanels === 0 && state.openerFocused && state.openerVisible && state.expanded === 'false',
    `Source close lost focus or left its panel open: ${JSON.stringify(state)}`);
  assert(state.openerUncovered, `Focused source button is obscured or outside the viewport: ${JSON.stringify(state)}`);
  assert(state.openerWithinReadingArea,
    `Focused source button is partially outside the reading area: ${JSON.stringify(state)}`);
  if (expectedOpener) {
    assert(state.openerText === expectedOpener, `Focus returned to a different citation: ${JSON.stringify(state)}`);
  }
  if (expectedPopoverIndex !== undefined) {
    assert(state.popoverIndex === expectedPopoverIndex,
      `Focus returned to a different answer's citation: ${JSON.stringify(state)}`);
  }
  return { check: "source closes and returns focus", ...state };
}

/** Theme alerts use opaque surfaces, so measure the actual browser colors. */
export async function statusContrast(tab) {
  const colors = await tab.playwright.locator('[data-testid="stAlertContainer"]').evaluateAll(elements =>
    elements.map(element => {
      const text = element.querySelector('p') || element;
      return { text: text.textContent, color: getComputedStyle(text).color,
        background: getComputedStyle(element).backgroundColor };
    }));
  assert(colors.length > 0, "No status feedback is visible for contrast checking");
  const luminance = rgb => rgb.map(value => value / 255)
    .map(value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4)
    .reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0);
  const results = colors.map(item => {
    const foreground = item.color.match(/[\d.]+/g).map(Number);
    const background = item.background.match(/[\d.]+/g).map(Number);
    assert(background.length === 3 || background[3] === 1, "Status surface must have an opaque theme color");
    const first = luminance(foreground.slice(0, 3));
    const second = luminance(background.slice(0, 3));
    return { ...item, ratio: (Math.max(first, second) + 0.05) / (Math.min(first, second) + 0.05) };
  });
  assert(results.every(item => item.ratio >= 4.5), `Low status text contrast: ${JSON.stringify(results)}`);
  return { check: "status text contrast >= 4.5", results };
}
