import { describe, expect, it, vi } from 'vitest';
import fixtures from '../../tests/fixtures/web_contracts/agent_responses.json';
import { parseAgentResponse } from './api/validation';
import { parseAnswerExport } from './api/delivery-validation';
import sourceFixtures from './components/AnswerView.fixtures.json';
import {
  createConversation,
  createConversationState,
  restoreConversations,
  serializeConversations,
  type ChatTurn,
} from './conversations';

function completedTurn(): ChatTurn {
  const result = parseAgentResponse(fixtures.responses.completed);
  return {
    id: 'turn-completed', query: '  질문 원문\r\n둘째 줄  ', result,
    notices: ['앞선 수신 안내'], error: null, executionId: result.execution_id,
    interrupted: false, receiving: false, exportResult: null, exportError: null,
  };
}

describe('브라우저 대화 표시 상태', () => {
  it('같은 원문을 사용한 여러 답변은 원문을 한 번 저장하고 할당량 안에서 복원한다', () => {
    const state = createConversationState();
    const response = structuredClone(sourceFixtures.quote);
    const sourceText = '# source-storage-regression\n'.repeat(2800);
    response.sources[response.citations[0].source_id as keyof typeof response.sources].element.text = sourceText;
    const result = parseAgentResponse({ ...fixtures.responses.completed, response });
    state.conversations[0].turns = Array.from({ length: 80 }, (_, index) => ({
      ...completedTurn(), id: `turn-${index}`, result,
    }));

    const serialized = serializeConversations(state);

    expect(serialized.split('source-storage-regression').length - 1).toBe(2800);
    expect(() => sessionStorage.setItem('large-conversation-regression', serialized)).not.toThrow();
    const restored = restoreConversations(serialized);
    expect(restored.conversations[0].turns).toEqual(state.conversations[0].turns);
    sessionStorage.removeItem('large-conversation-regression');
  });

  it('저장한 답변이 참조하는 원문이 누락되면 복원하지 않는다', () => {
    const state = createConversationState();
    state.conversations[0].turns = [{ ...completedTurn(),
      result: parseAgentResponse({ ...fixtures.responses.completed, response: sourceFixtures.quote }),
    }];
    const snapshot = JSON.parse(serializeConversations(state));
    snapshot.sources = {};
    expect(() => restoreConversations(JSON.stringify(snapshot))).toThrow('대화 기록');
  });

  it('randomUUID가 없는 HTTP 환경에서도 서로 다른 대화를 만들고 복원한다', () => {
    vi.stubGlobal('crypto', { getRandomValues: crypto.getRandomValues.bind(crypto) });
    try {
      const state = createConversationState();
      const next = createConversation();
      state.conversations.push(next);
      state.activeId = next.id;

      expect(next.id).not.toBe(state.conversations[0].id);
      for (const conversation of state.conversations) {
        expect(conversation.id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
      }
      expect(restoreConversations(serializeConversations(state))).toEqual(state);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('서로 다른 대화의 정확한 초안·답변·첨부 표시와 Slack 입력을 복원한다', () => {
    const state = createConversationState();
    const first = state.conversations[0];
    first.draft = '  첫 초안\n  들여쓰기\n';
    first.turns = [completedTurn()];
    first.confirmedEpoch = 'contract-epoch';
    first.manifest = fixtures.responses.completed.upload_manifest;
    first.connection = 'ready';
    first.slack.kind = 'email';
    first.slack.drafts.email = '  user@example.com  ';
    const second = createConversation();
    second.draft = '\t별도 초안\n';
    state.conversations.push(second);
    state.activeId = second.id;
    state.theme = 'dark';

    const restored = restoreConversations(serializeConversations(state));

    expect(restored.activeId).toBe(second.id);
    expect(restored.theme).toBe('dark');
    expect(restored.conversations[0].draft).toBe(first.draft);
    expect(restored.conversations[1].draft).toBe(second.draft);
    expect(restored.conversations[0].turns).toEqual(first.turns);
    expect(restored.conversations[0].slack).toEqual(first.slack);
    expect(restored.conversations[0].manifest).toEqual(first.manifest);
    expect(restored.conversations[0].connection).toBe('unknown');
    expect(restored.conversations[0].busy).toBeNull();
  });

  it('실행 중 질문은 한 번의 중단 기록으로 복원하고 실행 상태는 복원하지 않는다', () => {
    const state = createConversationState();
    const conversation = state.conversations[0];
    conversation.turns = [{
      ...completedTurn(), result: null, query: '결과를 저장하고 보내 줘', receiving: true,
    }];
    conversation.busy = 'chat';
    conversation.draft = '다음 질문의 미전송 초안';

    const restored = restoreConversations(serializeConversations(state)).conversations[0];

    expect(restored.turns).toHaveLength(1);
    expect(restored.turns[0].query).toBe('결과를 저장하고 보내 줘');
    expect(restored.turns[0].interrupted).toBe(true);
    expect(restored.turns[0].receiving).toBe(false);
    expect(restored.turns[0].error).toContain('서버 작업은 계속될 수 있습니다');
    expect(restored.draft).toBe('다음 질문의 미전송 초안');
    expect(restored.busy).toBeNull();
  });

  it('선택한 파일 바이트와 실행 계획을 저장하지 않고 재선택 안내와 보류 질문을 남긴다', () => {
    const state = createConversationState();
    state.conversations[0].attachments.files = [new File(['private-file-bytes'], '자료.py')];
    state.conversations[0].attachments.pendingPrompt = '  보류 질문\n';
    const serialized = serializeConversations(state);
    const restored = restoreConversations(serialized).conversations[0];

    expect(serialized).not.toContain('private-file-bytes');
    expect(serialized).not.toContain('자료.py');
    expect(restored.attachments.files).toEqual([]);
    expect(restored.attachments.pendingPrompt).toBe('  보류 질문\n');
    expect(restored.attachments.restoreFilesRequired).toBe(true);
    expect(restored.busy).toBeNull();
  });

  it('만료된 대화와 완료된 답변을 유지하고 완료 직후 복원도 중단으로 바꾸지 않는다', () => {
    const state = createConversationState();
    const conversation = state.conversations[0];
    conversation.turns = [{ ...completedTurn(), receiving: true }];
    conversation.connection = 'expired';

    const restored = restoreConversations(serializeConversations(state)).conversations[0];

    expect(restored.connection).toBe('expired');
    expect(restored.turns[0].result).toEqual(conversation.turns[0].result);
    expect(restored.turns[0].interrupted).toBe(false);
    expect(restored.turns[0].receiving).toBe(false);
  });

  it('당시 원문 근거와 답변별 저장 결과를 그대로 복원한다', () => {
    const state = createConversationState();
    const turn = completedTurn();
    turn.result = parseAgentResponse({ ...fixtures.responses.completed, response: sourceFixtures.quote });
    turn.exportResult = parseAnswerExport({
      status: 'failed', answer_id: turn.result.answer_id, execution_id: 'save-attempt',
      message: '파일을 저장하지 못했습니다.',
      receipt: { kind: 'save_text', status: 'error', message: '저장 공간을 확인해 주세요.', invocation_id: null,
        file_path: null, slack: null, previous_slack: null, error: null, operation: null, artifact: null, verification: 'failed', error_code: 'write_failed' },
    }, turn.result.answer_id!);
    turn.exportError = '저장 공간을 확인해 주세요.';
    state.conversations[0].turns = [turn];

    const restored = restoreConversations(serializeConversations(state)).conversations[0].turns[0];

    expect(restored.result?.response?.citations).toEqual(turn.result.response?.citations);
    expect(restored.result?.response?.citations.length).toBeGreaterThan(0);
    expect(restored.exportResult).toEqual(turn.exportResult);
    expect(restored.exportError).toBe(turn.exportError);
  });

  it.each([
    'not-json',
    JSON.stringify({ version: 999 }),
    JSON.stringify({ version: 1, conversations: [], activeId: 'missing', theme: 'system' }),
  ])('잘못된 snapshot을 조용히 보정하지 않는다: %s', (snapshot) => {
    expect(() => restoreConversations(snapshot)).toThrow('대화 기록');
  });

  it('선택 대화가 없거나 ID가 중복된 snapshot을 거절한다', () => {
    const state = createConversationState();
    state.activeId = 'missing';
    expect(() => restoreConversations(serializeConversations(state))).toThrow('대화 기록');
    state.activeId = state.conversations[0].id;
    state.conversations.push(state.conversations[0]);
    expect(() => restoreConversations(serializeConversations(state))).toThrow('대화 기록');
  });

  it('복원한 답변의 상태 계약도 API와 같은 경계에서 검증한다', () => {
    const state = createConversationState();
    state.conversations[0].turns = [completedTurn()];
    const serialized = serializeConversations(state).replace('"status":"completed"', '"status":"failed"');
    expect(() => restoreConversations(serialized)).toThrow('대화 기록');
  });

  it('과거의 검증되지 않은 저장 성공 기록을 새 성공 계약으로 복원하지 않는다', () => {
    const state = createConversationState();
    state.conversations[0].turns = [completedTurn()];
    const snapshot = JSON.parse(serializeConversations(state));
    snapshot.conversations[0].turns[0].result.response.actions = [{
      kind: 'save_text', status: 'success', message: '이전 저장 기록', invocation_id: null,
      file_path: 'old-answer.txt', slack: null, error: null, operation: null, artifact: null,
      verification: null, error_code: null,
    }];

    expect(() => restoreConversations(JSON.stringify(snapshot))).toThrow('대화 기록');
  });
});
