import { describe, expect, it } from 'vitest';
import type { components } from '../api/generated';
import { formatAnchor, selectedSourceAnchors, sourceExcerpt, sourceLineRecords, sourceLocationLabel } from './sourceLocations';

type SourceEvidence = Pick<components['schemas']['EvidenceRef'], 'element' | 'selection'>;
type Anchor = components['schemas']['SourceAnchor'];

function evidence(text: string, anchors: Anchor[], start = 0, end = Array.from(text).length): SourceEvidence {
  return {
    element: { element_id: 'code', kind: 'code', text, anchors },
    selection: { start, end, cell_ids: [] },
  };
}

describe('captured source locations', () => {
  it('preserves Unicode codepoint offsets and CRLF in a cropped source mapping', () => {
    const source = evidence('😀\r\n    value = 3\r\nnext\r\n', [
      { kind: 'code', start: 600, end: 624, line_start: 17, line_end: 19, precision: 'exact' },
    ], 7, 16);

    expect(sourceExcerpt(source)).toBe('value = 3');
    expect(sourceLineRecords(source)).toEqual([
      { line_number: 18, start: 7, end: 16, text: 'value = 3', partial_start: true, partial_end: true, cell_index: null },
    ]);
    expect(sourceLocationLabel(source)).toBe('사용 범위: 원문 18행');
    expect(selectedSourceAnchors(source)).toEqual([
      { kind: 'code', start: 607, end: 616, line_start: 18, line_end: 18, precision: 'exact' },
    ]);
    expect(source.element.text).toBe('😀\r\n    value = 3\r\nnext\r\n');
  });

  it.each([
    [{ kind: 'code', start: 0, end: 14, line_start: 40, line_end: 41, precision: 'element' }],
    [{ kind: 'code', line_start: 40, line_end: 41, precision: 'exact' }],
    [{ kind: 'code', start: 0, end: 7, line_start: 40, line_end: 40, precision: 'exact' }],
    [{ kind: 'code', start: 0, end: 14, line_start: 40, line_end: 44, precision: 'exact' }],
    [
      { kind: 'code', start: 0, end: 14, line_start: 40, line_end: 41, precision: 'exact' },
      { kind: 'code', start: 0, end: 14, line_start: 50, line_end: 51, precision: 'exact' },
    ],
  ] satisfies Anchor[][])('does not invent physical lines from incomplete or conflicting mapping %j', (...anchors) => {
    const source = evidence('header\ntarget\n', anchors);
    expect(sourceLineRecords(source)).toEqual([]);
    expect(sourceLocationLabel(source)).toBe('사용 범위: 수집한 요소의 1–2행 (원문 줄 번호는 확인되지 않았습니다)');
  });

  it('takes a Notebook cell only from the exact mapping used for the selected lines', () => {
    const source = evidence('first\nsecond\ntarget\nlast\n', [
      { kind: 'notebook', line_start: 19, line_end: 19, cell_index: 9, precision: 'exact' },
      { kind: 'notebook', start: 0, end: 25, line_start: 17, line_end: 20, cell_index: 2, precision: 'exact' },
    ], 13, 19);

    expect(sourceLineRecords(source)[0]?.cell_index).toBe(2);
    expect(sourceLocationLabel(source)).toBe('사용 범위: 셀 3의 19행');
    expect(selectedSourceAnchors(source).map(formatAnchor)).toContain('셀 10 · 19행');
  });

  it('counts only CR, LF, and CRLF as physical lines and preserves selected endings', () => {
    const source = evidence('a\fb\u2028c\rd\r\ne\n', [
      { kind: 'code', start: 0, end: 11, line_start: 1, line_end: 3, precision: 'exact' },
    ]);
    expect(sourceLineRecords(source).map(({ line_number, text }) => ({ line_number, text }))).toEqual([
      { line_number: 1, text: 'a\fb\u2028c\r' },
      { line_number: 2, text: 'd\r\n' },
      { line_number: 3, text: 'e\n' },
    ]);
  });

  it('keeps empty selections unnumbered', () => {
    const source = evidence('header\nvalue\n', [
      { kind: 'code', start: 0, end: 13, line_start: 17, line_end: 18, precision: 'exact' },
    ], 3, 3);
    expect(sourceLineRecords(source)).toEqual([]);
    expect(sourceExcerpt(source)).toBe('');
    expect(sourceLocationLabel(source)).toBe('사용 범위: 수집한 요소의 1행 (원문 줄 번호는 확인되지 않았습니다)');
  });

  it('renders selected table cells in original row order and retains coarse page fallback', () => {
    const page: Anchor = { kind: 'page', page_no: 3, precision: 'page', bbox: [0.1, 0.2, 0.8, 0.9] };
    const source: SourceEvidence = {
      element: {
        element_id: 'table', kind: 'table', text: '', anchors: [page],
        table: { cells: [
          { cell_id: 'b', row: 1, col: 0, text: '둘째', anchors: [] },
          { cell_id: 'a', row: 0, col: 0, text: '첫째', col_span: 2, anchors: [{ kind: 'page', page_no: 2, precision: 'page' }] },
          { cell_id: 'c', row: 1, col: 1, text: '선택 제외', anchors: [] },
        ] },
      },
      selection: { start: 0, end: null, cell_ids: ['b', 'a'] },
    };
    expect(sourceExcerpt(source)).toBe('첫째\n둘째');
    expect(sourceLineRecords(source)).toEqual([]);
    expect(sourceLocationLabel(source)).toBe('선택한 표 셀: b, a');
    expect(selectedSourceAnchors(source)).toEqual([
      { kind: 'page', page_no: 2, precision: 'page' }, page,
    ]);
    expect(formatAnchor(page)).toBe('3페이지 · 영역 [0.1, 0.2, 0.8, 0.9] (normalized, top_left)');
    expect(formatAnchor({ kind: 'page', page_no: 2 })).toBe('2페이지 · 페이지 안의 정확한 영역은 제공되지 않았습니다');
  });
});
