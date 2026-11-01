import { useState } from 'react';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { components } from '../api/generated';
import { EvidencePanel } from './EvidencePanel';

type Citation = components['schemas']['Citation'];

function citation(text: string, overrides: Partial<Citation['evidence']> = {}): Citation {
  return {
    number: 1,
    evidence: {
      id: 'ref:retained',
      snapshot: {
        snapshot_id: 'snapshot:retained', document_id: 'doc:retained', content_hash: `sha256:${'a'.repeat(64)}`,
        source_uri: 'https://docs.example.test/guide/page.html', title: '보관된 자료', media_type: 'text/markdown',
        source_type: 'official', parser: 'test', parser_version: '1', parser_config: {},
        capture_scope: 'provider_excerpt', quality_issues: ['freshness_unverified'],
      },
      element: { element_id: 'element:retained', kind: 'paragraph', text, anchors: [], heading_path: ['Guide', 'Example'] },
      selection: { start: 0, end: Array.from(text).length, cell_ids: [] },
      ...overrides,
    },
  };
}

function Harness({ source, context = [] }: {
  source: Citation;
  context?: { text: string; isCode?: boolean; inline?: boolean }[];
}) {
  const [opener, setOpener] = useState<HTMLButtonElement | null>(null);
  return <>
    <button type="button">다른 답변의 근거</button>
    <button type="button" onClick={(event) => setOpener(event.currentTarget)}>이 답변의 근거</button>
    {opener && <EvidencePanel citation={source} context={context} opener={opener} onClose={() => setOpener(null)} />}
  </>;
}

const originalShowModal = Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, 'showModal');
const originalClose = Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, 'close');
beforeEach(() => {
  // jsdom does not implement the native dialog boundary; browser tests exercise it.
  Object.defineProperties(HTMLDialogElement.prototype, {
    showModal: { configurable: true, value(this: HTMLDialogElement) { this.open = true; } },
    close: { configurable: true, value(this: HTMLDialogElement) { this.open = false; } },
  });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  for (const [name, original] of [['showModal', originalShowModal], ['close', originalClose]] as const) {
    if (original) Object.defineProperty(HTMLDialogElement.prototype, name, original);
    else Reflect.deleteProperty(HTMLDialogElement.prototype, name);
  }
});

describe('EvidencePanel', () => {
  it('shows the source identity and captured location before the excerpt', async () => {
    const user = userEvent.setup();
    render(<Harness source={citation('Retained source.')} />);
    await user.click(screen.getByRole('button', { name: '이 답변의 근거' }));
    const panel = screen.getByRole('dialog');
    const summary = within(panel).getByRole('group', { name: '출처 요약' });
    expect(within(summary).getByText('근거 1')).toBeVisible();
    expect(within(summary).getByText('공식 문서')).toBeVisible();
    expect(within(summary).getByText('사용 범위: 수집한 요소의 1행 (원문 줄 번호는 확인되지 않았습니다)')).toBeVisible();
    expect(within(summary).getByText('일부 수집 · 최신성 미확인')).toBeVisible();
    const excerptHeading = within(panel).getByRole('heading', { name: '답변에 사용한 부분' });
    expect(summary.compareDocumentPosition(excerptHeading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('keeps code match details collapsed while preserving the excerpt, source range and collection limits', async () => {
    const user = userEvent.setup();
    const text = 'alpha = 1\r\nbeta = 2\r\nalpha = 1\r\n';
    render(<Harness source={citation(text, {
      element: {
        element_id: 'code:retained', kind: 'code', text,
        anchors: [{ kind: 'code', start: 0, end: text.length, line_start: 17, line_end: 19, precision: 'exact' }],
      },
    })} context={[{ text: 'alpha = 1', isCode: true }]} />);
    await user.click(screen.getByRole('button', { name: '이 답변의 근거' }));
    const panel = screen.getByRole('dialog');
    const matchDetails = within(panel).getByText('답변의 코드와 일치하는 원문: 17행, 19행');
    const repeatedMatch = within(panel).getByText('일치하는 위치를 모두 표시했습니다.');
    expect(matchDetails).not.toBeVisible();
    expect(repeatedMatch).not.toBeVisible();
    expect(within(panel).getByText('사용 범위: 원문 17–19행')).toBeVisible();
    expect(within(panel).getByText('일부 수집 · 최신성 미확인')).toBeVisible();
    expect(within(panel).getByRole('group', { name: '원문 17행' })).toHaveTextContent('alpha = 1');
    expect(within(panel).getByRole('group', { name: '원문 18행' })).toHaveTextContent('beta = 2');
    expect(within(panel).getByRole('group', { name: '원문 19행' })).toHaveTextContent('alpha = 1');
    expect([...panel.querySelectorAll('.evidence-code-line mark')].map((mark) => mark.textContent)).toEqual(['alpha = 1', 'alpha = 1']);
    await user.click(within(panel).getByText('출처와 수집 정보'));
    expect(matchDetails).toBeVisible();
    expect(repeatedMatch).toBeVisible();
    expect(within(panel).getByText('현재 공식 페이지를 확인하지 못해 검색에 보관된 내용을 사용했습니다. 최신 문서와 다를 수 있습니다.')).toBeVisible();
  });

  it('copies only the selected excerpt with Unicode, CRLF, whitespace and Markdown intact', async () => {
    const user = userEvent.setup();
    const prefix = '제외 😀\r\n';
    const selected = '  **발췌** 😀\r\n\tsecond line\r\n';
    const text = `${prefix}${selected}제외한 끝`;
    render(<Harness source={citation(text, {
      selection: { start: Array.from(prefix).length, end: Array.from(prefix + selected).length, cell_ids: [] },
    })} />);
    await user.click(screen.getByRole('button', { name: '이 답변의 근거' }));
    const panel = screen.getByRole('dialog');
    await user.click(within(panel).getByRole('button', { name: '발췌 복사' }));
    expect(await navigator.clipboard.readText()).toBe(selected);
    expect(within(panel).getByText('발췌를 복사했습니다.')).toBeVisible();
    await user.click(within(panel).getByText('수집한 원문 그대로 보기'));
    await user.click(within(panel).getByRole('button', { name: '보관 원문 복사' }));
    expect(await navigator.clipboard.readText()).toBe(text);
    expect(within(panel).getByText('보관 원문을 복사했습니다.')).toBeVisible();
  });

  it('reports an unavailable clipboard without claiming the excerpt was copied', async () => {
    const user = userEvent.setup();
    vi.spyOn(navigator.clipboard, 'writeText').mockRejectedValue(new DOMException('Denied', 'NotAllowedError'));
    render(<Harness source={citation('Original excerpt.')} />);
    await user.click(screen.getByRole('button', { name: '이 답변의 근거' }));
    const panel = screen.getByRole('dialog');
    await user.click(within(panel).getByRole('button', { name: '발췌 복사' }));
    expect(within(panel).getByText('복사하지 못했습니다. 발췌를 선택해 복사해 주세요.')).toBeVisible();
    expect(within(panel).queryByText('발췌를 복사했습니다.')).not.toBeInTheDocument();
  });

  it('preserves the selected Unicode source, literal search, raw capture, and answer code', async () => {
    const user = userEvent.setup();
    const prefix = '제외된 😀 앞부분\n';
    const selected = `# Selected section\n\n${'Earlier unrelated paragraph.\n\n'.repeat(35)}A+B exact 문구.\n\n[현재 예제](../example.html)`;
    const text = `${prefix}${selected}\n제외된 뒷부분`;
    const source = citation(text, { selection: { start: Array.from(prefix).length, end: Array.from(prefix + selected).length, cell_ids: [] } });
    const original = JSON.stringify(source);
    const code = '__version__ = "1.0"\n# keep comment\nprint("<tag>")\n';
    render(<Harness source={source} context={[{ text: code, isCode: true }]} />);
    await user.click(screen.getByRole('button', { name: '이 답변의 근거' }));
    const panel = screen.getByRole('dialog', { name: '보관된 자료' });

    expect(within(panel).getByText('일부 수집 · 최신성 미확인')).toBeVisible();
    await user.type(within(panel).getByRole('searchbox', { name: '발췌에서 찾기' }), 'A+B');
    expect(within(panel).getByText('검색어와 일치하는 문단 1개 · 내용의 근거성 판정은 아닙니다.')).toBeVisible();
    expect(panel.querySelector('mark')?.textContent).toBe('A+B');
    await user.click(within(panel).getByText('답변에 사용한 전체 발췌'));
    expect(within(panel).getByRole('link', { name: '현재 예제' })).toHaveAttribute('href', 'https://docs.example.test/example.html');
    await user.click(within(panel).getByText('연결된 답변 보기'));
    expect(panel.querySelector('.evidence-answer-context pre code')?.textContent).toBe(code);
    await user.click(within(panel).getByText('수집한 원문 그대로 보기'));
    expect(panel.querySelector('.evidence-raw-source code')?.textContent).toBe(text);
    expect(within(panel).getByText(`자료 내용 해시: sha256:${'a'.repeat(64)}`)).toBeVisible();
    expect(JSON.stringify(source)).toBe(original);
  });

  it('preserves merged table cells and identifies selected cells', async () => {
    const user = userEvent.setup();
    const source = citation('', {
      element: { element_id: 'table:retained', kind: 'table', text: '', table: { cells: [
        { cell_id: 'header', row: 0, col: 0, col_span: 2, row_span: 1, text: '합계', is_header: true },
        { cell_id: 'left', row: 1, col: 0, row_span: 2, col_span: 1, text: '분류' },
        { cell_id: 'selected', row: 1, col: 1, row_span: 1, col_span: 1, text: '  12  ' },
        { cell_id: 'last', row: 2, col: 1, row_span: 1, col_span: 1, text: '9' },
      ] } },
      selection: { start: 0, end: null, cell_ids: ['selected'] },
    });
    render(<Harness source={source} />);
    await user.click(screen.getByRole('button', { name: '이 답변의 근거' }));
    const panel = screen.getByRole('dialog');
    const table = within(panel).getByRole('table', { name: '보관된 표와 선택 셀' });
    expect(within(panel).queryByRole('button', { name: '발췌 복사' })).not.toBeInTheDocument();
    expect(within(table).getByRole('columnheader', { name: '합계' })).toHaveAttribute('colspan', '2');
    expect(within(table).getByRole('cell', { name: '분류' })).toHaveAttribute('rowspan', '2');
    expect(table.querySelector('[data-selected="true"]')?.textContent).toBe('  12  ');
    expect(within(panel).getByText('선택한 표 셀: selected')).toBeVisible();
    expect(within(panel).getByText('강조한 셀을 답변 근거로 사용했습니다. 나머지 셀은 표의 문맥을 보여 줍니다.')).toBeVisible();
  });

  it('keeps the focused search visible below the sticky header after a height resize', async () => {
    const user = userEvent.setup();
    render(<Harness source={citation('Long source paragraph.\n\n'.repeat(40))} />);
    await user.click(screen.getByRole('button', { name: '이 답변의 근거' }));
    const panel = screen.getByRole('dialog');
    const header = panel.querySelector('header')!;
    const search = within(panel).getByRole('searchbox', { name: '발췌에서 찾기' });
    let panelHeight = 500;
    vi.spyOn(panel, 'getBoundingClientRect').mockImplementation(() => new DOMRect(0, 0, 390, panelHeight));
    vi.spyOn(header, 'getBoundingClientRect').mockImplementation(() => new DOMRect(0, 0, 390, 70));
    vi.spyOn(search, 'getBoundingClientRect').mockImplementation(() => new DOMRect(20, 300 - panel.scrollTop, 350, 40));
    search.focus();

    panelHeight = 180;
    fireEvent(window, new Event('resize'));

    expect(search).toHaveFocus();
    expect(search.getBoundingClientRect().top).toBeGreaterThanOrEqual(header.getBoundingClientRect().bottom);
    expect(search.getBoundingClientRect().bottom).toBeLessThanOrEqual(panelHeight - 12);

    panel.scrollTop = 290;
    fireEvent(window, new Event('resize'));
    expect(search.getBoundingClientRect().top).toBeGreaterThanOrEqual(header.getBoundingClientRect().bottom);
    expect(search).toHaveFocus();
  });

  it('keeps keyboard focus inside and restores the actual opener with Escape or close', async () => {
    const user = userEvent.setup();
    render(<Harness source={citation('Original excerpt.')} />);
    const opener = screen.getByRole('button', { name: '이 답변의 근거' });
    await user.click(opener);
    const panel = screen.getByRole('dialog');
    expect(panel).toContainElement(document.activeElement as HTMLElement);
    const close = within(panel).getByRole('button', { name: '근거 닫기' });
    close.focus();
    await user.keyboard('{Shift>}{Tab}{/Shift}');
    expect(panel).toContainElement(document.activeElement as HTMLElement);
    fireEvent.keyDown(panel, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
    await user.click(opener);
    await user.click(screen.getByRole('button', { name: '근거 닫기' }));
    expect(opener).toHaveFocus();
  });
});
