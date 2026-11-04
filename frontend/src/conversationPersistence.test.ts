import { afterEach, describe, expect, it, vi } from 'vitest';
import { ConversationPersistence, loadConversationState } from './conversationPersistence';
import { createConversationState, restoreConversations, STORAGE_KEY } from './conversations';

afterEach(() => {
  vi.useRealTimers();
  sessionStorage.clear();
});

describe('대화 저장 수명', () => {
  it('지원하지 않는 이전 형식은 보존하고 명시적 정리만 해당 기록을 삭제한다', () => {
    sessionStorage.setItem('documate.react.conversations.v1', '{"version":1}');
    const state = createConversationState();
    const persistence = new ConversationPersistence(() => sessionStorage, () => {});
    persistence.schedule(state);
    persistence.flush();

    const initial = loadConversationState(() => sessionStorage);

    expect(initial.state.conversations).toHaveLength(1);
    expect(initial.obsoleteRecord).toBe(true);
    expect(sessionStorage.getItem('documate.react.conversations.v1')).toBe('{"version":1}');

    expect(persistence.clearObsoleteRecord()).toBe(true);
    expect(sessionStorage.getItem('documate.react.conversations.v1')).toBeNull();
    expect(restoreConversations(sessionStorage.getItem(STORAGE_KEY)!)).toEqual(state);
  });

  it('연속 입력은 저장을 모으고 요청 전 flush는 최신 초안을 즉시 보존한다', () => {
    vi.useFakeTimers();
    const errors: (string | null)[] = [];
    const persistence = new ConversationPersistence(() => sessionStorage, (error) => errors.push(error));
    const state = createConversationState();
    persistence.schedule(state);
    persistence.schedule({ ...state, conversations: [{ ...state.conversations[0], draft: '첫 입력' }] });
    persistence.schedule({ ...state, conversations: [{ ...state.conversations[0], draft: '  최신 초안\n' }] });

    expect(sessionStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(persistence.flush()).toBe(true);
    expect(restoreConversations(sessionStorage.getItem(STORAGE_KEY)!).conversations[0].draft).toBe('  최신 초안\n');
    expect(errors.at(-1)).toBeNull();
    vi.runAllTimers();
    expect(restoreConversations(sessionStorage.getItem(STORAGE_KEY)!).conversations[0].draft).toBe('  최신 초안\n');
  });

  it('입력이 멈추면 최신 상태를 저장한다', () => {
    vi.useFakeTimers();
    const persistence = new ConversationPersistence(() => sessionStorage, () => {});
    const state = createConversationState();
    state.conversations[0].draft = '자동 보존할 초안';
    persistence.schedule(state);

    vi.runAllTimers();

    expect(restoreConversations(sessionStorage.getItem(STORAGE_KEY)!).conversations[0].draft).toBe('자동 보존할 초안');
  });

  it('할당량 초과를 안내하고 오래된 대화를 삭제한 다음 저장에 성공하면 경고를 해제한다', () => {
    const errors: (string | null)[] = [];
    const persistence = new ConversationPersistence(() => sessionStorage, (error) => errors.push(error));
    const state = createConversationState();
    persistence.schedule(state);
    expect(persistence.flush()).toBe(true);
    const largeConversation = { ...state.conversations[0], id: 'large', draft: '원'.repeat(5_100_000) };
    persistence.schedule({ ...state, conversations: [...state.conversations, largeConversation] });

    expect(persistence.flush()).toBe(false);
    expect(errors.at(-1)).toContain('삭제');
    expect(restoreConversations(sessionStorage.getItem(STORAGE_KEY)!).conversations).toHaveLength(1);

    persistence.schedule(state);
    expect(persistence.flush()).toBe(true);
    expect(errors.at(-1)).toBeNull();
  });
});
