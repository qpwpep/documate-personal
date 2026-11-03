import { cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AgentResponse, AnswerResponse } from '../api/types';
import { AnswerView } from './AnswerView';
import sourceFixtures from './AnswerView.fixtures.json';

const unit = (text: string) => ({ text, basis: 'interaction' as const, refs: [], source_location: null, sentence_scope: null });

const originalShowModal = Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, 'showModal');
const originalClose = Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, 'close');
beforeEach(() => {
  // jsdom lacks this browser boundary; actual focus and layout also have E2E coverage.
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

// The content hash and unit paths were checked with Python's AnswerResponse.
const response: AnswerResponse = {
  content: {
    blocks: [
      { type: 'heading', level: 2, content: unit('원문 보존') },
      { type: 'paragraph', content: [unit('첫 문장.'), unit('둘째 문장.')] },
      { type: 'list', ordered: true, items: [unit('항목 하나'), unit('항목 둘')] },
      { type: 'code', language: 'python', inline: false, content: unit('def run():\n    return "<tag>"\n') },
      { type: 'code', language: '', inline: true, content: unit('__version__') },
      { type: 'table', columns: [unit('이름'), unit('값')], rows: [[unit('공백'), unit('  원문  ')]] },
      { type: 'paragraph', content: [unit('<script>alert("실행 안 됨")</script>')] },
    ],
  },
  citations: [],
  checks: [
    'b0.content', 'b1.content.0', 'b1.content.1', 'b2.items.0', 'b2.items.1', 'b3.content',
    'b4.content', 'b5.columns.0', 'b5.columns.1', 'b5.rows.0.0', 'b5.rows.0.1', 'b6.content.0',
  ].map((unit_id) => ({ unit_id, reference_status: 'not_required', support_status: 'not_evaluated', issues: [] })),
  issues: [{ code: 'source_limit', message: '일부 원문을 확인하지 못했습니다.', unit_id: null }],
  actions: [
    { kind: 'save_text', status: 'error', message: '파일을 저장하지 못했습니다.', invocation_id: null,
      file_path: null, slack: null, previous_slack: null, error: null, operation: null, artifact: null, verification: null, error_code: null },
    { kind: 'slack_notify', status: 'skipped', message: '수신자를 확인해 주세요.', invocation_id: null,
      file_path: null, slack: null, previous_slack: null, error: null, operation: null, artifact: null, verification: null, error_code: null },
  ],
  content_hash: '6a089f7a08d3dcaf3b9a3248d0bf5901908a81d5e74eada2b9fe09bb1f04a28f',
  retrieval_required: false,
  semantic_validation: null,
};

describe('AnswerView', () => {
  it('copies only the selected code block with its exact Unicode, spacing, and line endings', async () => {
    const user = userEvent.setup();
    const text = 'def 인사():\r\n\treturn "안녕 👋"  \r\n';
    const withCode: AnswerResponse = { ...response, content: { blocks: [
      { type: 'code', language: 'python', inline: false, content: unit('print("first")\n') },
      { type: 'code', language: 'python', inline: false, content: unit(text) },
    ] } };
    const original = JSON.stringify(withCode);
    const { container } = render(<AnswerView response={withCode} />);

    await user.click(screen.getAllByRole('button', { name: '코드 복사' })[1]);

    expect(await navigator.clipboard.readText()).toBe(text);
    expect(screen.getAllByRole('status', { name: '코드 복사 상태' })[1]).toHaveTextContent('코드를 복사했습니다.');
    expect(container.querySelectorAll('pre code')[1].textContent).toBe(text);
    expect(JSON.stringify(withCode)).toBe(original);
  });

  it('reports a denied code copy and lets the user try again without changing the code', async () => {
    const user = userEvent.setup();
    const write = vi.spyOn(navigator.clipboard, 'writeText').mockRejectedValueOnce(new DOMException('Denied', 'NotAllowedError'));
    const { container } = render(<AnswerView response={response} />);

    await user.click(screen.getByRole('button', { name: '코드 복사' }));

    expect(screen.getByRole('status')).toHaveTextContent('복사하지 못했습니다. 코드를 선택해 복사하거나 다시 시도해 주세요.');
    expect(container.querySelector('pre code')?.textContent).toBe('def run():\n    return "<tag>"\n');
    write.mockRestore();
    await user.click(screen.getByRole('button', { name: '코드 복사' }));
    expect(await navigator.clipboard.readText()).toBe('def run():\n    return "<tag>"\n');
    expect(screen.getByRole('status')).toHaveTextContent('코드를 복사했습니다.');
  });

  it('shows every body layout without changing code, spaces, or literal HTML', () => {
    const { container } = render(<AnswerView response={response} />);

    expect(screen.getByRole('heading', { name: '원문 보존', level: 2 })).toBeVisible();
    expect(screen.getByText('첫 문장.').closest('p')?.textContent).toBe('첫 문장. 둘째 문장.');
    expect(screen.getByText('항목 하나').closest('ol')).not.toBeNull();
    expect(screen.getByText('항목 둘')).toBeVisible();
    expect(container.querySelector('pre code')?.textContent).toBe('def run():\n    return "<tag>"\n');
    expect(screen.getByText('__version__').tagName).toBe('CODE');
    const table = screen.getByRole('table');
    expect(within(table).getByRole('columnheader', { name: '이름' })).toBeVisible();
    expect(within(table).getByRole('cell', { name: '원문' }).textContent).toBe('  원문  ');
    expect(screen.getByText('<script>alert("실행 안 됨")</script>')).toBeVisible();
    expect(container.querySelector('script')).toBeNull();
  });

  it('preserves block Markdown returned as a single paragraph by an exact input copy', () => {
    const text = '# Copied title\n\n3. first step\n4. second step\n\n| name | value |\n| --- | --- |\n| x | 2 |\n\n```python\ndef run():\n    return "<tag>"\n```';
    const copied: AnswerResponse = { ...response,
      content: { blocks: [{ type: 'paragraph', content: [unit(text)] }] },
      checks: [], issues: [], actions: [],
    };
    const original = JSON.stringify(copied);
    const { container, rerender } = render(<AnswerView response={copied} />);
    const body = container.querySelector('.answer-body') as HTMLElement;

    expect(within(body).getByRole('heading', { name: 'Copied title', level: 1 })).toBeVisible();
    const list = within(body).getByRole('list');
    expect(list.tagName).toBe('OL');
    expect(list).toHaveAttribute('start', '3');
    expect(within(list).getAllByRole('listitem').map((item) => item.textContent)).toEqual(['first step', 'second step']);
    const table = within(body).getByRole('table');
    expect(within(table).getAllByRole('columnheader').map((cell) => cell.textContent)).toEqual(['name', 'value']);
    expect(within(table).getAllByRole('cell').map((cell) => cell.textContent)).toEqual(['x', '2']);
    expect(body.querySelector('pre code')?.textContent).toBe('def run():\n    return "<tag>"\n');
    expect(body.querySelector('p :is(h1, ol, table, pre)')).toBeNull();
    expect(JSON.stringify(copied)).toBe(original);

    rerender(<AnswerView response={{ ...copied,
      content: { blocks: [{ type: 'paragraph', content: [unit('    def run():\n        return 1\n')] }] },
    }} />);
    expect(body.querySelector('pre code')?.textContent).toBe('def run():\n    return 1\n');
  });

  it('keeps answer limits and each delivery result separate from the body', () => {
    render(<AnswerView response={response} />);

    expect(screen.getByText('확인할 한계: 일부 원문을 확인하지 못했습니다.')).toBeVisible();
    const actions = screen.getByRole('region', { name: '전달 결과' });
    expect(within(actions).getByText('파일 저장 · 실패')).toBeVisible();
    expect(within(actions).getByText('Slack 전송 · 실행하지 않음')).toBeVisible();
    expect(within(actions).getByText('파일을 저장하지 못했습니다.')).toBeVisible();
    expect(within(actions).getByText('수신자를 확인해 주세요.')).toBeVisible();
    expect(within(actions).queryByText('성공')).not.toBeInTheDocument();
  });

  it('lists each used source once after the body while preserving every inline citation', () => {
    const sectioned = sourceFixtures.sectioned as AnswerResponse;
    const original = JSON.stringify(sectioned);
    const { container } = render(<><AnswerView response={sectioned} /><AnswerView response={sectioned} /></>);

    expect(screen.getAllByRole('button', { name: '근거 [1] 열기' })).toHaveLength(6);
    const sources = screen.getAllByRole('button', { name: '근거 [1] 같은 출처' });
    expect(sources).toHaveLength(2);
    for (const source of sources) {
      expect(source.closest('.answer-body')).toBeNull();
      expect(source.closest('.answer-toolbar')).not.toBeNull();
    }
    const answerIds = Array.from(container.querySelectorAll('.answer-view')).map((answer) => answer.id);
    expect(new Set(answerIds).size).toBe(2);
    expect(JSON.stringify(sectioned)).toBe(original);
  });

  it('opens all connected sections from the answer source button and omits unused sources', async () => {
    const user = userEvent.setup();
    const sectioned = structuredClone(sourceFixtures.sectioned) as AnswerResponse;
    sectioned.citations.push({ ...sectioned.citations[0], number: 2, evidence: {
      ...sectioned.citations[0].evidence, id: 'unused-source', snapshot: {
        ...sectioned.citations[0].evidence.snapshot, title: '사용하지 않은 출처',
      },
    } });
    render(<AnswerView response={sectioned} />);
    expect(screen.queryByRole('button', { name: '근거 [2] 사용하지 않은 출처' })).not.toBeInTheDocument();
    const opener = screen.getByRole('button', { name: '근거 [1] 같은 출처' });
    await user.click(opener);
    const panel = screen.getByRole('dialog', { name: '같은 출처' });
    await user.click(within(panel).getByText('연결된 답변 보기'));
    expect(within(panel).getByText('첫 설명')).toBeVisible();
    expect(within(panel).getByText('둘째 설명')).toBeVisible();
    expect(within(panel).getByText('마지막 설명')).toBeVisible();
    await user.keyboard('{Escape}');
    expect(opener).toHaveFocus();
  });

  it('keeps sentence scopes as metadata without changing prose or citation context', async () => {
    const user = userEvent.setup();
    const scoped = structuredClone(sourceFixtures.sectioned) as AnswerResponse;
    const paragraph = scoped.content.blocks[1];
    if (paragraph.type !== 'paragraph') throw new Error('Fixture must contain a paragraph.');
    paragraph.content[0].sentence_scope = 1;
    paragraph.content[1].sentence_scope = 1;
    const original = JSON.stringify(scoped);
    const { container } = render(<AnswerView response={scoped} />);

    expect(container.querySelector('.answer-body p')?.textContent).toBe('첫 설명[1] 둘째 설명[1]');
    await user.click(screen.getByRole('button', { name: '근거 [1] 같은 출처' }));
    const panel = screen.getByRole('dialog', { name: '같은 출처' });
    await user.click(within(panel).getByText('연결된 답변 보기'));
    expect(within(panel).getByText('첫 설명')).toBeVisible();
    expect(within(panel).getByText('둘째 설명')).toBeVisible();
    expect(JSON.stringify(scoped)).toBe(original);
    expect(container).not.toHaveTextContent('sentence_scope');
  });

  it('distinguishes source connections, exact excerpts, and semantic evaluation', async () => {
    const user = userEvent.setup();
    render(<AnswerView response={sourceFixtures.quote as AnswerResponse}
      actions={<button type="button">TXT 파일 만들기</button>} requestId="request-info" />);

    expect(screen.getByRole('button', { name: 'TXT 파일 만들기' }).closest('.answer-toolbar')).not.toBeNull();
    expect(screen.getByText('출처가 연결된 내용 1개 · 원문 발췌 1개 중 원문 일치 1개')).not.toBeVisible();
    expect(screen.getByText('요청 ID: request-info')).not.toBeVisible();
    await user.click(screen.getByText('답변 정보'));
    expect(screen.getByText('출처가 연결된 내용 1개 · 원문 발췌 1개 중 원문 일치 1개')).toBeVisible();
    expect(screen.getByText('요청 ID: request-info')).toBeVisible();
    expect(screen.getByText('자동 확인은 답변 전체의 정확도나 실행 성공을 보증하지 않습니다.')).toBeVisible();
    await user.click(screen.getByText('기술 상세'));
    expect(screen.getByText('답변 본문 조건 충족: 검증 안 됨')).toBeVisible();
  });

  it('opens only the selected answer section and returns focus to its actual citation', async () => {
    const user = userEvent.setup();
    const second = structuredClone(sourceFixtures.sectioned) as AnswerResponse;
    const last = second.content.blocks.at(-1);
    if (last?.type !== 'paragraph') throw new Error('Fixture must end in a paragraph.');
    last.content[0].text = '두 번째 답변 전용 문장';
    render(<><AnswerView response={sourceFixtures.sectioned as AnswerResponse} /><AnswerView response={second} /></>);

    const opener = screen.getAllByRole('button', { name: '근거 [1] 열기' }).at(-1)!;
    await user.click(opener);
    const panel = screen.getByRole('dialog', { name: '같은 출처' });
    await user.click(within(panel).getByText('연결된 답변 보기'));
    expect(within(panel).getByText('두 번째 답변 전용 문장')).toBeVisible();
    expect(within(panel).queryByText('첫 설명')).not.toBeInTheDocument();
    expect(within(panel).queryByText('마지막 설명')).not.toBeInTheDocument();
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
  });

  it('keeps unknown Slack delivery and unverified saves distinct from success', () => {
    const deliveryResponse: AnswerResponse = { ...response, actions: [
      { ...response.actions[0], status: 'unknown', message: '저장 여부를 확인하지 못했습니다.', verification: 'unverifiable' },
      { ...response.actions[1], status: 'unknown', message: '응답을 받지 못했습니다.', slack: {
        intent: { state: 'omitted' }, status: 'unknown', invocation_id: 'delivery-1',
        selection: { source: 'configured_default', request_id: 'request-1', selector: { kind: 'channel', value: 'C123' } },
        operation: { operation_id: 'delivery-operation', request_id: 'request-1', payload_sha256: 'b'.repeat(64), byte_count: 18 },
        target: { channel_id: 'C123', user_id: null }, message_ts: null, resolved_user_id: null,
        failure: { code: 'delivery_unknown', stage: 'send', message: '전달 여부를 확인할 수 없습니다.', next_action: 'verify_delivery',
          retry_after_seconds: null, slack_error: null },
      } },
    ] };
    render(<AnswerView response={deliveryResponse} />);

    const actions = screen.getByRole('region', { name: '전달 결과' });
    expect(within(actions).getByText('파일 저장 · 결과 확인 필요')).toBeVisible();
    expect(within(actions).getByText('Slack 전송 · 전달 여부 불명')).toBeVisible();
    expect(within(actions).getByText('기본 수신자: C123')).toBeVisible();
    expect(within(actions).getByText('Slack에서 전달 여부를 확인해 주세요. 자동으로 다시 전송하지 않습니다.')).toBeVisible();
    expect(within(actions).queryByText(/메시지 식별자/)).not.toBeInTheDocument();
    expect(within(actions).queryByRole('button')).not.toBeInTheDocument();
  });

  it('separates an unsent corrected body from the previous body whose Slack delivery is unknown', () => {
    const corrected: AnswerResponse = { ...response,
      content: { blocks: [{ type: 'paragraph', content: [unit('수정한 본문입니다.')] }] },
      actions: [{ ...response.actions[1], message: '수정한 본문은 전송하지 않았습니다.',
        previous_slack: {
          invocation_id: 'previous-send', intent: { state: 'omitted' },
          selection: { source: 'configured_default', request_id: 'request-1', selector: { kind: 'channel', value: 'C123' } },
          operation: { operation_id: 'previous-operation', request_id: 'request-1', payload_sha256: 'a'.repeat(64), byte_count: 18 },
          status: 'unknown', target: { channel_id: 'C123', user_id: null },
          message_ts: null, resolved_user_id: null,
          failure: { code: 'delivery_unknown', stage: 'send', message: '이전 전송의 응답을 받지 못했습니다.',
            next_action: 'verify_delivery', retry_after_seconds: null, slack_error: null },
        },
      }],
    };
    const original = JSON.stringify(corrected);
    render(<AnswerView response={corrected} />);

    expect(screen.getByText('수정한 본문입니다.')).toBeVisible();
    const actions = screen.getByRole('region', { name: '전달 결과' });
    expect(within(actions).getByText('Slack 전송 · 현재 본문 미전송')).toBeVisible();
    expect(within(actions).getByText('수정한 본문은 전송하지 않았습니다.')).toBeVisible();
    const previous = within(actions).getByRole('region', { name: '이전 본문의 Slack 전달 결과' });
    expect(within(previous).getByText('이전 본문 · 전달 여부 불명')).toBeVisible();
    expect(within(previous).getByText('이전 수신자: C123')).toBeVisible();
    expect(within(previous).getByText('이전 전송의 응답을 받지 못했습니다.')).toBeVisible();
    expect(within(previous).getByText('Slack에서 전달 여부를 확인해 주세요. 자동으로 다시 전송하지 않습니다.')).toBeVisible();
    expect(within(actions).queryByText('Slack 전송 · 전달 여부 불명')).not.toBeInTheDocument();
    expect(within(actions).queryByText(/메시지 식별자|완료/)).not.toBeInTheDocument();
    expect(within(actions).queryByRole('button')).not.toBeInTheDocument();
    expect(actions).not.toHaveTextContent('a'.repeat(64));
    expect(JSON.stringify(corrected)).toBe(original);
  });

  it('keeps source mismatch warnings beside the affected content', () => {
    render(<AnswerView response={sourceFixtures.mismatch as AnswerResponse} />);
    const warning = screen.getByText('이 인용문은 보관한 원문과 일치하지 않습니다.');
    expect(warning).toBeVisible();
    expect(warning.closest('.answer-block')).toHaveTextContent('원문과 다른 인용');
  });

  it('identifies unassessed content and offers a concrete next step for unknown locations', async () => {
    const user = userEvent.setup();
    render(<AnswerView response={{ ...sourceFixtures.quote as AnswerResponse, issues: [
      { code: 'source_location_unverified', unit_id: 'b0.content.0', message: '정확한 원문 위치를 확인하지 못했습니다.' },
    ] }} />);

    expect(screen.getByText('출처에서 원문 위치를 확인하거나, 위치를 확인할 수 있는 원본 파일을 첨부해 주세요.')).toBeVisible();
    expect(screen.getByText('확인할 한계: 내용별 한계 1개는 해당 내용 아래에서 확인할 수 있습니다.')).toBeVisible();
    await user.click(screen.getByText('답변 정보'));
    await user.click(screen.getByText('기술 상세'));
    const contentCheck = screen.getByRole('region', { name: '내용 1 검증' });
    expect(within(contentCheck).getByText(/보관된 원문 문장입니다/)).toBeVisible();
    expect(within(contentCheck).getByText('근거 지지: 검증 안 됨')).toBeVisible();
  });

  it('summarizes partial scope and preserves every limit in collapsed details', async () => {
    const user = userEvent.setup();
    const partialResponse: AnswerResponse = {
      ...response,
      issues: [
        { code: 'answer_incomplete', message: '요청 전체를 완성하지 못했습니다.', unit_id: null },
        { code: 'evidence_packet_incomplete', message: '검색한 원문을 답변 준비 범위에 모두 담지 못했습니다.', unit_id: null },
        { code: 'source_excerpt_fallback', message: '요청한 설명 대신 확인 가능한 원문 발췌만 제공합니다.', unit_id: null },
        { code: 'scope_limited', unit_id: 'b1.content.0', message: '이 문장의 추가 조건은 확인하지 못했습니다.' },
      ],
    };
    const turn: AgentResponse = {
      status: 'partial', response: partialResponse, answer_id: 'answer-partial', message: '',
      problem: { code: 'evidence_insufficient', stage: 'validation', message: '근거를 충분히 확인하지 못했습니다.', next_action: 'supply_information', retry_after_seconds: null },
      request_id: 'request-partial', missing_slots: [], trace: '', debug: null, upload_manifest: { epoch: 'session', revision: 0, files: [] },
    };
    render(<AnswerView response={partialResponse} turn={turn} />);

    const summary = screen.getByRole('region', { name: '부분 답변' });
    expect(within(summary).getByText('근거를 충분히 확인하지 못했습니다.')).toBeVisible();
    expect(within(summary).getByText('요청한 설명 대신 확인 가능한 원문 발췌만 제공합니다.')).toBeVisible();
    expect(within(summary).getByText(/질문을 항목별로 나누어/)).toBeVisible();
    expect(within(summary).queryByText(/자료를 추가/)).not.toBeInTheDocument();
    await user.click(screen.getByText('답변의 한계와 처리 정보'));
    expect(screen.getByText('근거를 충분히 확인하지 못했습니다.')).toBeVisible();
    expect(screen.getByText('오류 식별자: request-partial')).toBeVisible();
    await user.click(screen.getByText('이 내용의 한계 (1개)'));
    expect(screen.getByText('이 문장의 추가 조건은 확인하지 못했습니다.')).toBeVisible();
  });

  it('does not call a checked answer incomplete when only saving failed', async () => {
    const user = userEvent.setup();
    const checked: AnswerResponse = {
      ...response, issues: [], semantic_validation: {
        content_hash: response.content_hash, evidence_fingerprint: '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
        model: 'gpt-6-luna', threshold: 0.8,
        checks: [{ name: 'request_fulfilled', probability: 0.95, unit_id: null, status: 'evaluated' }],
      },
    };
    const turn: AgentResponse = {
      status: 'partial', response: checked, answer_id: 'saved-answer', message: '',
      problem: { code: 'save_failed', stage: 'action_postprocess', message: '파일을 만들지 못했습니다.', next_action: 'retry_later', retry_after_seconds: null },
      request_id: 'save-request', missing_slots: [], trace: '', debug: null, upload_manifest: { epoch: 'session', revision: 0, files: [] },
    };
    render(<AnswerView response={checked} turn={turn} />);

    expect(screen.queryByRole('region', { name: '부분 답변' })).not.toBeInTheDocument();
    expect(screen.getByText('파일을 만들지 못했습니다.')).toBeVisible();
    await user.click(screen.getByText('답변 정보'));
    expect(screen.getByText(/이번 요청 처리는 완료되지 않았습니다/)).toBeVisible();
    await user.click(screen.getByText('기술 상세'));
    expect(screen.getByText(/답변 본문 조건 충족: 통과/)).toBeVisible();
  });
});
