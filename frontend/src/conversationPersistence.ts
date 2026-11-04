import { createConversationState, restoreConversations, serializeConversations, STORAGE_KEY, type ConversationState } from './conversations';

type ConversationStorage = Pick<Storage, 'setItem' | 'removeItem'>;
const obsoleteStorageKey = 'documate.react.conversations.v1';

export function loadConversationState(storage: () => Pick<Storage, 'getItem'>): {
  state: ConversationState; error: string | null; obsoleteRecord: boolean;
} {
  let obsoleteRecord = false;
  try {
    const saved = storage();
    const snapshot = saved.getItem(STORAGE_KEY);
    obsoleteRecord = saved.getItem(obsoleteStorageKey) !== null;
    return {
      state: snapshot ? restoreConversations(snapshot) : createConversationState(),
      error: null, obsoleteRecord,
    };
  } catch {
    return { state: createConversationState(), error: '저장된 대화를 복원하지 못했습니다. 새 대화를 시작할 수 있습니다.', obsoleteRecord };
  }
}

/** Browser persistence owns write timing; React only supplies immutable snapshots. */
export class ConversationPersistence {
  private pending: ConversationState | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private readonly storage: () => ConversationStorage,
    private readonly onResult: (error: string | null) => void,
  ) {}

  schedule(state: ConversationState): void {
    this.pending = state;
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = setTimeout(() => this.flush(), 250);
  }

  clearObsoleteRecord(): boolean {
    try {
      this.storage().removeItem(obsoleteStorageKey);
    } catch {
      this.onResult('이전 형식의 대화 기록을 정리하지 못했습니다. 브라우저 저장소 사용을 확인해 주세요.');
      return false;
    }
    this.onResult(null);
    this.flush();
    return true;
  }

  flush(): boolean {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    if (this.pending === null) return true;
    const snapshot = serializeConversations(this.pending);
    try {
      this.storage().setItem(STORAGE_KEY, snapshot);
    } catch {
      this.onResult('대화를 브라우저에 저장하지 못했습니다. 오래된 대화를 삭제하면 저장 공간을 확보할 수 있습니다. 저장이 복구되기 전 새로고침하면 최신 기록을 잃을 수 있습니다.');
      return false;
    }
    this.pending = null;
    this.onResult(null);
    return true;
  }
}
