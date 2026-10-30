import type { components } from '../api/generated';

export type SourceEvidence = Pick<components['schemas']['EvidenceRef'], 'element' | 'selection'>;
export type SourceAnchor = components['schemas']['SourceAnchor'];
export type SourceLine = {
  line_number: number;
  start: number;
  end: number;
  text: string;
  partial_start: boolean;
  partial_end: boolean;
  cell_index: number | null;
};

export function sourceExcerpt(evidence: SourceEvidence): string {
  const selected = new Set(evidence.selection?.cell_ids ?? []);
  if (selected.size) {
    const rows = new Map<number, string[]>();
    const cells = [...(evidence.element.table?.cells ?? [])].sort((left, right) => left.row - right.row || left.col - right.col);
    for (const cell of cells) {
      if (!selected.has(cell.cell_id)) continue;
      const row = rows.get(cell.row) ?? [];
      row.push(cell.text);
      rows.set(cell.row, row);
    }
    return [...rows.values()].map((row) => row.join(' | ')).join('\n');
  }
  // API offsets count Unicode characters, whereas JS string offsets count UTF-16 units.
  return Array.from(evidence.element.text ?? '').slice(evidence.selection?.start ?? 0, evidence.selection?.end ?? undefined).join('');
}

export function selectedSourceAnchors(evidence: SourceEvidence): SourceAnchor[] {
  const source = sourceMap(evidence.element.text ?? '');
  const selected = new Set(evidence.selection?.cell_ids ?? []);
  const start = evidence.selection?.start ?? 0;
  const end = evidence.selection?.end ?? source.characters.length;
  let anchors: SourceAnchor[];
  if (selected.size) {
    const cells = (evidence.element.table?.cells ?? []).filter((cell) => selected.has(cell.cell_id));
    anchors = cells.flatMap((cell) => cell.anchors ?? []);
    if (cells.some((cell) => !cell.anchors?.length)) anchors.push(...(evidence.element.anchors ?? []));
  } else {
    anchors = (evidence.element.anchors ?? []).filter((anchor) => {
      if (anchor.start == null || anchor.end == null) return true;
      const bounds = textAnchorBounds(source, anchor);
      return (bounds?.[0] === 0 && bounds[1] === source.characters.length)
        || (anchor.start < end && start < anchor.end);
    });
  }
  const unique = new Map<string, SourceAnchor>();
  for (const original of anchors) {
    let anchor = { ...original };
    const bounds = selected.size ? null : textAnchorBounds(source, anchor);
    if (bounds && anchor.start != null && anchor.line_start != null) {
      const first = Math.max(start, bounds[0]);
      const last = Math.min(end, bounds[1]);
      if (first < last) {
        const lineOffset = anchor.line_start - lineAt(source, bounds[0]);
        anchor = {
          ...anchor,
          start: anchor.start + first - bounds[0],
          end: anchor.start + last - bounds[0],
          line_start: lineOffset + lineAt(source, first),
          line_end: lineOffset + lineAt(source, last - 1),
        };
      }
    }
    unique.set(anchorKey(anchor), anchor);
  }
  return [...unique.values()];
}

export function sourceLineRecords(evidence: SourceEvidence): SourceLine[] {
  if (evidence.selection?.cell_ids?.length) return [];
  const source = sourceMap(evidence.element.text ?? '');
  const start = evidence.selection?.start ?? 0;
  const end = evidence.selection?.end ?? source.characters.length;
  if (start === end) return [];
  const mappings = new Map<string, { anchor: SourceAnchor; bounds: [number, number] }>();
  for (const anchor of evidence.element.anchors ?? []) {
    const bounds = textAnchorBounds(source, anchor);
    if (bounds && bounds[0] < end && start < bounds[1]) mappings.set(anchorKey(anchor), { anchor, bounds });
  }
  if (mappings.size !== 1) return [];
  const { anchor, bounds } = [...mappings.values()][0];
  if (start < bounds[0] || end > bounds[1] || anchor.line_start == null) return [];
  const offset = anchor.line_start - lineAt(source, bounds[0]);
  const lines: SourceLine[] = [];
  const last = lineAt(source, end - 1);
  for (let line = lineAt(source, start); line <= last; line += 1) {
    const first = Math.max(start, source.offsets[line - 1]);
    const finish = Math.min(end, source.offsets[line]);
    lines.push({
      line_number: offset + line,
      start: first,
      end: finish,
      text: source.characters.slice(first, finish).join(''),
      partial_start: first > source.offsets[line - 1],
      partial_end: finish < source.offsets[line],
      cell_index: anchor.cell_index ?? null,
    });
  }
  return lines;
}

export function sourceLocationLabel(evidence: SourceEvidence): string {
  if (evidence.selection?.cell_ids?.length) return `선택한 표 셀: ${evidence.selection.cell_ids.join(', ')}`;
  const lines = sourceLineRecords(evidence);
  if (lines.length) {
    const scope = lines[0].cell_index == null ? '원문 ' : `셀 ${lines[0].cell_index + 1}의 `;
    return `사용 범위: ${scope}${lineRange(lines[0].line_number, lines[lines.length - 1].line_number)}`;
  }
  const source = sourceMap(evidence.element.text ?? '');
  const start = evidence.selection?.start ?? 0;
  const end = evidence.selection?.end ?? source.characters.length;
  return `사용 범위: 수집한 요소의 ${lineRange(lineAt(source, start), lineAt(source, Math.max(start, end - 1)))} (원문 줄 번호는 확인되지 않았습니다)`;
}

export function formatAnchor(anchor: SourceAnchor): string {
  const parts: string[] = [];
  if (anchor.cell_index != null) parts.push(`셀 ${anchor.cell_index + 1}`);
  if (anchor.cell_id) parts.push(`셀 ID ${anchor.cell_id}`);
  if (anchor.line_start != null) parts.push(lineRange(anchor.line_start, anchor.line_end ?? anchor.line_start));
  if (anchor.page_no != null) {
    parts.push(`${anchor.page_no}페이지`);
    parts.push(anchor.bbox
      ? `영역 [${anchor.bbox.join(', ')}] (${anchor.coordinate_space ?? 'normalized'}, ${anchor.coordinate_origin ?? 'top_left'})`
      : '페이지 안의 정확한 영역은 제공되지 않았습니다');
  }
  return parts.length ? parts.join(' · ') : '문서 요소 위치';
}

function sourceMap(text: string): { characters: string[]; offsets: number[] } {
  const characters = Array.from(text);
  const offsets = [0];
  for (let index = 0; index < characters.length; index += 1) {
    if (characters[index] === '\r' && characters[index + 1] === '\n') {
      index += 1;
      offsets.push(index + 1);
    } else if (characters[index] === '\r' || characters[index] === '\n') {
      offsets.push(index + 1);
    }
  }
  if (offsets[offsets.length - 1] !== characters.length) offsets.push(characters.length);
  return { characters, offsets };
}

function lineAt(source: ReturnType<typeof sourceMap>, offset: number): number {
  const next = source.offsets.findIndex((boundary) => boundary > offset);
  let line = next < 0 ? source.offsets.length : next;
  if (offset === source.characters.length && source.characters.length && !['\r', '\n'].includes(source.characters[source.characters.length - 1])) line -= 1;
  return Math.max(1, line);
}

function textAnchorBounds(source: ReturnType<typeof sourceMap>, anchor: SourceAnchor): [number, number] | null {
  if (!['code', 'notebook', 'text'].includes(anchor.kind) || anchor.precision !== 'exact'
    || anchor.start == null || anchor.end == null || anchor.line_start == null) return null;
  let bounds: [number, number];
  if (anchor.end - anchor.start === source.characters.length && source.characters.length) bounds = [0, source.characters.length];
  else if (0 <= anchor.start && anchor.start < anchor.end && anchor.end <= source.characters.length) bounds = [anchor.start, anchor.end];
  else return null;
  const expectedEnd = anchor.line_start + lineAt(source, bounds[1] - 1) - lineAt(source, bounds[0]);
  return anchor.line_end == null || anchor.line_end === expectedEnd ? bounds : null;
}

function anchorKey(anchor: SourceAnchor): string {
  return JSON.stringify(Object.fromEntries(Object.entries(anchor).sort(([left], [right]) => left.localeCompare(right))));
}

function lineRange(start: number, end: number): string {
  return start === end ? `${start}행` : `${start}–${end}행`;
}
