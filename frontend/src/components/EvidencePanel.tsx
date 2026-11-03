import { Fragment, useEffect, useId, useRef, useState } from 'react';
import type { KeyboardEvent, ReactNode } from 'react';
import type { components } from '../api/generated';
import { MarkdownView } from './MarkdownView';
import { formatAnchor, selectedSourceAnchors, sourceExcerpt, sourceLineRecords, sourceLocationLabel } from './sourceLocations';
import './evidence.css';

type Citation = components['schemas']['Citation'];
type TableData = components['schemas']['TableData'];
export type AnswerContext = { text: string; isCode?: boolean; inline?: boolean };

const BROAD_EXCERPT_CHARACTERS = 700;
const BROAD_EXCERPT_LINES = 12;
const CODE_CONTEXT_LINES = 2;
const RELATED_PREVIEW_LEADING_CHARACTERS = 120;
const RELATED_PREVIEW_CHARACTERS = 600;
const RELATED_PREVIEW_PARAGRAPHS = 3;
const FALLBACK_PREVIEW_LINES = 6;
const FALLBACK_PREVIEW_CHARACTERS = 420;

export function EvidencePanel({ citation, context = [], opener, onClose }: {
  citation: Citation;
  context?: readonly AnswerContext[];
  opener: HTMLElement;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const title = useRef<HTMLHeadingElement>(null);
  const titleId = useId();
  const searchId = useId();
  const [query, setQuery] = useState('');
  const [copyStatus, setCopyStatus] = useState({ excerpt: '', source: '' });
  const { evidence } = citation;
  const { element, snapshot } = evidence;
  const excerpt = sourceExcerpt(evidence);
  const sourceLines = sourceLineRecords(evidence);
  const sourceContexts = context.filter((item) => item.text.trim());
  const broad = Array.from(excerpt).length > BROAD_EXCERPT_CHARACTERS
    || excerpt.split(/\r\n|\r|\n/).length > BROAD_EXCERPT_LINES;
  const exactPhrases = [...new Set(sourceContexts.map((item) => item.text).filter((text) => excerpt.includes(text)))];
  const search = query.trim();
  const paragraphs = excerpt.split(/\r?\n\s*\r?\n/);
  const matches = search ? paragraphs.filter((paragraph) => paragraph.toLocaleLowerCase().includes(search.toLocaleLowerCase())) : [];
  const ranges = matchingCodeRanges(excerpt, evidence.selection?.start ?? 0, sourceContexts);
  const matchedLines = sourceLines.filter((line) => ranges.some(([start, end]) => start < line.end && line.start < end));
  const visibleLines = matchedLines.length
    ? sourceLines.filter((line) => matchedLines.some((match) => Math.abs(match.line_number - line.line_number) <= CODE_CONTEXT_LINES))
    : sourceLines;
  const status = [snapshot.capture_scope === 'provider_excerpt' ? '일부 수집' : '',
    snapshot.quality_issues?.includes('freshness_unverified') ? '최신성 미확인' : ''].filter(Boolean).join(' · ');

  useEffect(() => {
    const panel = dialog.current;
    if (panel === null) return;
    panel.showModal();
    title.current?.focus({ preventScroll: true });
    const revealFocusedControl = () => {
      if (document.activeElement instanceof HTMLElement) keepPanelControlVisible(panel, document.activeElement);
    };
    window.addEventListener('resize', revealFocusedControl);
    return () => {
      window.removeEventListener('resize', revealFocusedControl);
      panel.close();
      if (!opener.isConnected) return;
      opener.focus({ preventScroll: true });
      const bounds = opener.getBoundingClientRect();
      const header = document.querySelector<HTMLElement>('.app-header');
      const composer = document.querySelector<HTMLElement>('.composer-area');
      const top = header !== null && ['fixed', 'sticky'].includes(getComputedStyle(header).position)
        ? Math.max(0, header.getBoundingClientRect().bottom) : 0;
      const bottom = composer !== null && ['fixed', 'sticky'].includes(getComputedStyle(composer).position)
        ? Math.min(window.innerHeight, composer.getBoundingClientRect().top) : window.innerHeight;
      if (bounds.top < top || bounds.bottom > bottom || bounds.left < 0 || bounds.right > window.innerWidth) {
        opener.scrollIntoView?.({ block: 'center', inline: 'nearest' });
        const moved = opener.getBoundingClientRect();
        if (moved.top < top || moved.bottom > bottom) window.scrollBy?.(0, moved.top - top - Math.max(0, (bottom - top - moved.height) / 2));
      }
    };
  }, [opener]);

  function handleKeyDown(event: KeyboardEvent<HTMLDialogElement>) {
    if (event.key === 'Escape') {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key !== 'Tab') return;
    const panel = event.currentTarget;
    const controls = [...panel.querySelectorAll<HTMLElement>('a[href], button:not(:disabled), input:not(:disabled), summary, [tabindex="0"]')]
      .filter((element) => isVisibleControl(element, panel));
    const first = controls[0];
    const last = controls.at(-1);
    if (event.shiftKey && (document.activeElement === first || document.activeElement === title.current)) {
      event.preventDefault();
      last?.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first?.focus();
    }
  }

  async function copySource(target: 'excerpt' | 'source') {
    try {
      await navigator.clipboard.writeText(target === 'excerpt' ? excerpt : element.text ?? '');
      setCopyStatus((current) => ({ ...current, [target]: target === 'excerpt' ? '발췌를 복사했습니다.' : '보관 원문을 복사했습니다.' }));
    } catch {
      setCopyStatus((current) => ({ ...current, [target]: target === 'excerpt'
        ? '복사하지 못했습니다. 발췌를 선택해 복사해 주세요.'
        : '복사하지 못했습니다. 아래 원문을 선택해 복사해 주세요.' }));
    }
  }

  return <dialog ref={dialog} className="evidence-panel" aria-labelledby={titleId} onKeyDown={handleKeyDown}
    onCancel={(event) => { event.preventDefault(); onClose(); }}>
    <header className="evidence-header">
      <h2 id={titleId} ref={title} tabIndex={-1}>{snapshot.title || snapshot.source_uri}</h2>
      <button type="button" className="secondary-button" aria-label="근거 닫기" onClick={onClose}>닫기</button>
    </header>
    <div className="evidence-content" onFocusCapture={(event) => {
      if (dialog.current) keepPanelControlVisible(dialog.current, event.target);
    }}>
      <div className="evidence-source-summary" role="group" aria-label="출처 요약">
        <p className="evidence-source-identity"><span>근거 {citation.number}</span><span aria-hidden="true"> · </span>
          <span>{snapshot.source_type === 'official' ? '공식 문서' : '업로드 자료'}</span><span aria-hidden="true"> · </span>
          <span>{sourceLocationLabel(evidence)}</span>
        </p>
        {status && <p className="evidence-caption">{status}</p>}
      </div>
      {/^https?:\/\//i.test(snapshot.source_uri) && <a href={snapshot.source_uri} target="_blank" rel="noreferrer">현재 출처 페이지 열기</a>}
      <div className="evidence-excerpt-heading">
        <h3>답변에 사용한 부분</h3>
        {!element.table && <button type="button" className="secondary-button" onClick={() => void copySource('excerpt')}>발췌 복사</button>}
      </div>
      {!element.table && <p className="evidence-copy-status" role="status">{copyStatus.excerpt}</p>}
      {element.table ? <>
        <EvidenceTable table={element.table} selected={evidence.selection?.cell_ids ?? []} />
        {!!evidence.selection?.cell_ids?.length && <p className="evidence-caption">강조한 셀을 답변 근거로 사용했습니다. 나머지 셀은 표의 문맥을 보여 줍니다.</p>}
      </>
        : element.kind === 'code' ? <>
          {visibleLines.length ? <div className="evidence-code-lines">
            {visibleLines.map((line, index) => <Fragment key={line.line_number}>
              {index > 0 && line.line_number > visibleLines[index - 1].line_number + 1 && <div aria-label="생략한 원문">…</div>}
              <div role="group" aria-label={`원문 ${line.line_number}행`} className="evidence-code-line">
                <span aria-hidden="true" className="evidence-line-number">{line.line_number}</span>
                <code>{highlightRanges(line.text.replace(/[\r\n]+$/, ''), ranges.map(([start, end]) => [start - line.start, end - line.start]))}</code>
              </div>
            </Fragment>)}
          </div> : <pre><code>{excerpt}</code></pre>}
        </> : <>
          {broad && <>
            {exactPhrases.length > 0
              ? <><p className="evidence-caption">답변 문구와 정확히 일치하는 원문을 먼저 표시합니다. 연결된 전체 범위는 아래에서 확인하세요.</p>
                {exactPhrases.map((phrase) => <p className="evidence-literal" key={phrase}><mark>{phrase}</mark></p>)}</>
              : <p className="evidence-caption">넓은 원문 범위 · 문장별 일치 위치 미확인</p>}
            <label htmlFor={searchId}>발췌에서 찾기</label>
            <input type="search" id={searchId} value={query} onChange={(event) => setQuery(event.currentTarget.value)} placeholder="원문에서 찾을 단어 또는 문구" />
            {search ? matches.length > 0
              ? <><p className="evidence-caption">검색어와 일치하는 문단 {matches.length}개 · 내용의 근거성 판정은 아닙니다.</p>
                {matches.map((paragraph, index) => <p className="evidence-literal" key={index}>{highlightLiteral(paragraph, search)}</p>)}</>
              : <p>이 발췌에는 일치하는 문구가 없습니다. 다른 단어를 입력하거나 전체 발췌를 확인하세요.</p>
              : exactPhrases.length === 0 && <RelatedPreview excerpt={excerpt} context={sourceContexts} />}
            <details><summary>답변에 사용한 전체 발췌</summary><MarkdownView text={excerpt} sourceUri={snapshot.source_uri} providerExcerpt={snapshot.capture_scope === 'provider_excerpt'} /></details>
          </>}
          {!broad && (snapshot.source_type === 'official'
            ? <MarkdownView text={excerpt} sourceUri={snapshot.source_uri} providerExcerpt={snapshot.capture_scope === 'provider_excerpt'} />
            : <p className="evidence-literal">{excerpt}</p>)}
        </>}
      {sourceContexts.length > 0 && <details><summary>연결된 답변 보기</summary>
        <div className="evidence-answer-context"><p>이 근거가 연결된 답변</p>
          {sourceContexts.map((item, index) => item.isCode
            ? item.inline ? <p key={index}><code>{item.text}</code></p> : <pre key={index}><code>{item.text}</code></pre>
            : <MarkdownView key={index} text={item.text} />)}
        </div>
      </details>}
      <details><summary>출처와 수집 정보</summary>
        {element.heading_path?.length ? <p className="evidence-caption">문서 위치: {element.heading_path.join(' › ')}</p> : null}
        <p>원본 자료: {snapshot.source_uri}</p>
        {selectedSourceAnchors(evidence).map((anchor, index) => <p key={index}>
          {anchor.line_start !== null && anchor.line_start !== undefined ? '원문 요소 위치: ' : ''}{formatAnchor(anchor)}
        </p>)}
        {element.kind === 'code' && matchedLines.length > 0 && <p className="evidence-caption">답변의 코드와 일치하는 원문: {matchedLines.map((line) => `${line.line_number}행`).join(', ')}</p>}
        {element.kind === 'code' && ranges.length > 1 && <p className="evidence-caption">일치하는 위치를 모두 표시했습니다.</p>}
        {snapshot.capture_scope === 'provider_excerpt' && <p>원문의 일부만 수집한 자료입니다. 답변에는 당시 수집한 내용을 사용했습니다.</p>}
        {snapshot.quality_issues?.map((issue) => <p className="evidence-warning" key={issue}>{qualityMessage(issue)}</p>)}
      </details>
      <details><summary>{snapshot.source_type === 'official' ? '수집한 원문 그대로 보기' : '전체 원문 보기'}</summary>
        {element.table ? <EvidenceTable table={element.table} selected={evidence.selection?.cell_ids ?? []} label="전체 표 원문" />
          : <><button type="button" className="secondary-button" onClick={() => void copySource('source')}>보관 원문 복사</button>
            <p role="status">{copyStatus.source}</p><pre className="evidence-raw-source"><code>{element.text}</code></pre></>}
        <p className="evidence-caption">이 내용은 답변을 만들 때 보관한 원문입니다. 현재 파일의 변경이나 삭제와 무관합니다.</p>
        <p className="evidence-caption">자료 내용 해시: {snapshot.content_hash}</p>
      </details>
    </div>
  </dialog>;
}

function keepPanelControlVisible(panel: HTMLDialogElement, control: HTMLElement): void {
  if (!panel.querySelector('.evidence-content')?.contains(control)) return;
  const bounds = control.getBoundingClientRect();
  const top = (panel.querySelector('header')?.getBoundingClientRect().bottom ?? panel.getBoundingClientRect().top) + 12;
  const bottom = panel.getBoundingClientRect().bottom - 12;
  if (bounds.top < top) panel.scrollTop -= top - bounds.top;
  else if (bounds.bottom > bottom) panel.scrollTop += bounds.bottom - bottom;
}

function isVisibleControl(element: HTMLElement, panel: HTMLElement): boolean {
  for (let ancestor: HTMLElement | null = element; ancestor !== null && ancestor !== panel; ancestor = ancestor.parentElement) {
    if (ancestor.hidden || getComputedStyle(ancestor).display === 'none' || getComputedStyle(ancestor).visibility === 'hidden') return false;
    if (ancestor instanceof HTMLDetailsElement && !ancestor.open && !ancestor.querySelector(':scope > summary')?.contains(element)) return false;
  }
  return true;
}

function qualityMessage(issue: string): string {
  return ({ html_structure_flattened: '원문 웹페이지에서 본문을 추출했습니다. 표와 배치 등 일부 서식은 원본 페이지와 다를 수 있습니다.',
    freshness_unverified: '현재 공식 페이지를 확인하지 못해 검색에 보관된 내용을 사용했습니다. 최신 문서와 다를 수 있습니다.' })[issue] ?? issue;
}

function EvidenceTable({ table, selected, label = '보관된 표와 선택 셀' }: { table: TableData; selected: string[]; label?: string }) {
  const cells = table.cells;
  const positions = new Map(cells.map((cell) => [`${cell.row}:${cell.col}`, cell]));
  const rowCount = Math.max(0, ...cells.map((cell) => cell.row + cell.row_span));
  const columnCount = Math.max(0, ...cells.map((cell) => cell.col + cell.col_span));
  const covered = new Set<string>();
  return <div className="evidence-table"><table aria-label={label}><tbody>
    {Array.from({ length: rowCount }, (_, row) => <tr key={row}>
      {Array.from({ length: columnCount }, (_, column) => {
        const position = `${row}:${column}`;
        if (covered.has(position)) return null;
        const cell = positions.get(position);
        if (cell === undefined) return <td key={column} />;
        for (let r = cell.row; r < cell.row + cell.row_span; r += 1) {
          for (let c = cell.col; c < cell.col + cell.col_span; c += 1) covered.add(`${r}:${c}`);
        }
        const Cell = cell.is_header ? 'th' : 'td';
        return <Cell key={column} scope={cell.is_header ? 'col' : undefined} rowSpan={cell.row_span} colSpan={cell.col_span}
          data-selected={selected.includes(cell.cell_id) ? 'true' : undefined}>{cell.text}</Cell>;
      })}
    </tr>)}
  </tbody></table></div>;
}

function matchingCodeRanges(excerpt: string, offset: number, context: readonly AnswerContext[]): [number, number][] {
  const snippets = new Set(context.flatMap((item) => [item.text, ...[...item.text.matchAll(/`+([^`\n]+)`+/g)].map((match) => match[1])]).filter(Boolean));
  const ranges: [number, number][] = [];
  for (const snippet of snippets) {
    let cursor = excerpt.indexOf(snippet);
    while (cursor >= 0) {
      const start = offset + Array.from(excerpt.slice(0, cursor)).length;
      ranges.push([start, start + Array.from(snippet).length]);
      cursor = excerpt.indexOf(snippet, cursor + 1);
    }
  }
  const merged: [number, number][] = [];
  for (const range of ranges.sort((a, b) => a[0] - b[0] || a[1] - b[1])) {
    const previous = merged.at(-1);
    if (previous && range[0] <= previous[1]) previous[1] = Math.max(previous[1], range[1]);
    else merged.push([...range]);
  }
  return merged;
}

function highlightRanges(text: string, ranges: readonly (readonly number[])[]): ReactNode[] {
  const characters = Array.from(text);
  const result: ReactNode[] = [];
  let cursor = 0;
  for (const [start, end] of ranges) {
    const first = Math.max(cursor, start);
    const last = Math.min(characters.length, end);
    if (first >= last) continue;
    result.push(characters.slice(cursor, first).join(''), <mark key={first}>{characters.slice(first, last).join('')}</mark>);
    cursor = last;
  }
  result.push(characters.slice(cursor).join(''));
  return result;
}

function highlightLiteral(text: string, query: string): ReactNode[] {
  const ranges: [number, number][] = [];
  const lower = text.toLocaleLowerCase();
  const needle = query.toLocaleLowerCase();
  let cursor = lower.indexOf(needle);
  while (cursor >= 0) {
    ranges.push([Array.from(text.slice(0, cursor)).length, Array.from(text.slice(0, cursor + query.length)).length]);
    cursor = lower.indexOf(needle, cursor + query.length);
  }
  return highlightRanges(text, ranges);
}

function RelatedPreview({ excerpt, context }: { excerpt: string; context: readonly AnswerContext[] }) {
  const common = new Set(['the', 'and', 'for', 'with', 'from', 'this', 'that', 'these', 'those', 'are', 'was', 'were', 'will', 'can', 'may', 'has', 'have', 'had', 'not', 'but', 'into', 'when', 'then', 'than', 'its', 'each', 'both', 'only', 'also', 'use', 'using', 'used', 'returns', 'return']);
  const terms = [...new Set(context.flatMap((item) => item.text.match(/[A-Za-z_][A-Za-z0-9_.]*/g) ?? []).map((term) => term.replace(/\.+$/, '').toLowerCase()))]
    .filter((term) => term.length >= 3 && !common.has(term));
  const candidates = excerpt.split(/\n\s*\n/).map((paragraph, index) => {
    const words = new Set((paragraph.match(/[A-Za-z_][A-Za-z0-9_.]*/g) ?? []).map((term) => term.replace(/\.+$/, '').toLowerCase()));
    const matches = terms.filter((term) => words.has(term));
    const position = matches.length ? Math.min(...matches.map((term) => paragraph.toLowerCase().indexOf(term))) : 0;
    const start = Math.max(0, Math.min(position - RELATED_PREVIEW_LEADING_CHARACTERS, paragraph.length - RELATED_PREVIEW_CHARACTERS));
    return { paragraph, index, matches, start, description: (paragraph.match(/\p{L}+/gu)?.length ?? 0) > matches.length + 2 };
  }).filter((item) => item.matches.length).sort((a, b) => b.matches.length - a.matches.length || Number(b.description) - Number(a.description) || a.index - b.index).slice(0, RELATED_PREVIEW_PARAGRAPHS);
  if (candidates.length === 0) return <><p className="evidence-literal">{Array.from(excerpt.split(/\r\n|\r|\n/).slice(0, FALLBACK_PREVIEW_LINES).join('\n')).slice(0, FALLBACK_PREVIEW_CHARACTERS).join('')}…</p>
    <p className="evidence-caption">답변과 겹치는 용어를 찾지 못해 원문 앞부분을 표시합니다. 전체 발췌에서 확인하세요.</p></>;
  return <><p className="evidence-caption">답변에 나온 용어가 포함된 원문을 먼저 표시합니다. 용어 일치는 내용의 근거성 판정은 아닙니다.</p>
    {candidates.map(({ paragraph, index, start }) => <p className="evidence-literal" key={index}>{start > 0 ? '…' : ''}{paragraph.slice(start, start + RELATED_PREVIEW_CHARACTERS)}{start + RELATED_PREVIEW_CHARACTERS < paragraph.length ? '…' : ''}</p>)}</>;
}
