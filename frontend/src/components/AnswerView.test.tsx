import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { AnswerResponse } from '../api/types';
import { AnswerView } from './AnswerView';

const unit = (text: string) => ({ text, basis: 'interaction' as const, refs: [], source_location: null });

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
      file_path: null, slack: null, error: null, operation: null, artifact: null, verification: null, error_code: null },
    { kind: 'slack_notify', status: 'skipped', message: '수신자를 확인해 주세요.', invocation_id: null,
      file_path: null, slack: null, error: null, operation: null, artifact: null, verification: null, error_code: null },
  ],
  content_hash: '6a089f7a08d3dcaf3b9a3248d0bf5901908a81d5e74eada2b9fe09bb1f04a28f',
  retrieval_required: false,
  semantic_validation: null,
};

describe('AnswerView', () => {
  it('shows every body layout without changing code, spaces, or literal HTML', () => {
    const { container } = render(<AnswerView response={response} />);

    expect(screen.getByRole('heading', { name: '원문 보존', level: 2 })).toBeVisible();
    expect(screen.getByText('첫 문장. 둘째 문장.')).toBeVisible();
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

  it('keeps answer limits and each delivery result separate from the body', () => {
    render(<AnswerView response={response} />);

    expect(screen.getByText('일부 원문을 확인하지 못했습니다.')).toBeVisible();
    const actions = screen.getByRole('region', { name: '전달 결과' });
    expect(within(actions).getByText('파일 저장 · 실패')).toBeVisible();
    expect(within(actions).getByText('Slack 전송 · 실행하지 않음')).toBeVisible();
    expect(within(actions).getByText('파일을 저장하지 못했습니다.')).toBeVisible();
    expect(within(actions).getByText('수신자를 확인해 주세요.')).toBeVisible();
    expect(within(actions).queryByText('성공')).not.toBeInTheDocument();
  });
});
