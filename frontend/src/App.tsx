import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { AgentSessionClient, ApiError, fetchClientConfig } from './api/client';
import type { AgentResponse, ClientConfig, UploadManifest } from './api/types';
import { useAttachments } from './attachments/useAttachments';
import { AnswerView } from './components/AnswerView';
import { AnswerDelivery } from './components/AnswerDelivery';
import { SlackSettings } from './components/SlackSettings';
import { selectedSlackRecipient } from './components/SlackSettingsState';
import { ConversationSidebar } from './components/ConversationSidebar';
import {
  STORAGE_KEY, conversationTitle, createConversation, createConversationState,
  restoreConversations, serializeConversations, type ChatTurn, type Conversation, type ConversationState,
} from './conversations';

const resultLabels = {
  completed: '답변 받음', partial: '일부 처리', needs_input: '추가 정보 필요', failed: '요청 실패', refused: '응답 거절',
};
const stageLabels: Record<string, string> = {
  summarize: '이전 대화를 정리하고 있습니다.', planner: '질문을 확인하고 있습니다.',
  retrieval: '자료를 찾고 있습니다.', synthesis: '답변을 작성하고 있습니다.',
  pre_synthesis_validation: '검색한 근거를 확인하고 있습니다.',
  post_synthesis_validation: '답변과 근거를 확인하고 있습니다.', action_postprocess: '전달 결과를 확인하고 있습니다.',
};
const docExamples = [
  'pandas merge 사용법을 공식 문서 기준으로 설명해줘',
  'NumPy 배열 reshape 예제를 공식 문서 기준으로 알려줘',
  'FastAPI 라우터 구조를 간단히 설명해줘',
  'scikit-learn train_test_split 사용법을 정리해줘',
];
const uploadExamples = ['첨부한 자료의 핵심 내용을 요약하고 근거를 보여줘', '첨부한 자료에서 중요한 용어를 설명해줘'];

function initialState() {
  try {
    const snapshot = sessionStorage.getItem(STORAGE_KEY);
    return { state: snapshot ? restoreConversations(snapshot) : createConversationState(), error: null };
  } catch {
    return { state: createConversationState(), error: '저장된 대화를 복원하지 못했습니다. 새 대화를 시작할 수 있습니다.' };
  }
}

export function App({ apiBaseUrl = '' }: { apiBaseUrl?: string }) {
  const [initial] = useState(initialState);
  const [state, setState] = useState(initial.state);
  const currentState = useRef(state);
  const mounted = useRef(true);
  const [storageError, setStorageError] = useState<string | null>(initial.error);
  const [config, setConfig] = useState<ClientConfig | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [configAttempt, setConfigAttempt] = useState(0);
  const [progress, setProgress] = useState<Record<string, string>>({});
  const [sidebarModal, setSidebarModal] = useState(false);
  const clients = useRef(new Map<string, AgentSessionClient>());
  const requests = useRef(new Map<string, AbortController>());
  const validations = useRef(new Map<string, number>());
  const input = useRef<HTMLTextAreaElement>(null);

  const mutate = useCallback((update: (value: ConversationState) => ConversationState) => {
    if (!mounted.current) return;
    const next = update(currentState.current);
    currentState.current = next;
    setState(next);
    // Persist the utterance before HTTP begins. Restore records, never commands.
    try {
      sessionStorage.setItem(STORAGE_KEY, serializeConversations(next));
    } catch {
      setStorageError('대화를 브라우저에 저장하지 못했습니다. 이 탭에서는 계속 사용할 수 있지만 새로고침 후 기록이 복원되지 않을 수 있습니다.');
    }
  }, []);
  const updateConversation = useCallback((id: string, update: (value: Conversation) => Conversation) => {
    mutate((value) => ({ ...value, conversations: value.conversations.map((item) => item.id === id ? update(item) : item) }));
  }, [mutate]);
  const getClient = useCallback((id: string) => {
    const key = `${apiBaseUrl}|${id}`;
    let client = clients.current.get(key);
    if (!client) {
      client = new AgentSessionClient({ sessionId: id, baseUrl: apiBaseUrl });
      clients.current.set(key, client);
    }
    return client;
  }, [apiBaseUrl]);

  const ensureSession = useCallback(async (id: string, signal?: AbortSignal): Promise<UploadManifest | null> => {
    const selected = currentState.current.conversations.find((item) => item.id === id);
    if (!selected || selected.connection === 'expired') return null;
    const generation = (validations.current.get(id) ?? 0) + 1;
    validations.current.set(id, generation);
    updateConversation(id, (item) => ({ ...item, connection: 'checking', connectionError: null }));
    try {
      const manifest = await getClient(id).refreshUploads({ signal });
      if (!mounted.current || signal?.aborted || validations.current.get(id) !== generation) return null;
      const latest = currentState.current.conversations.find((item) => item.id === id);
      if (!latest) return null;
      if (latest.confirmedEpoch !== null && latest.confirmedEpoch !== manifest.epoch) {
        getClient(id).manifest = null;
        updateConversation(id, (item) => ({ ...item, connection: 'expired', connectionError: null }));
        return null;
      }
      updateConversation(id, (item) => ({ ...item, manifest, confirmedEpoch: manifest.epoch, connection: 'ready', connectionError: null }));
      return manifest;
    } catch (error: unknown) {
      if (!mounted.current || signal?.aborted || validations.current.get(id) !== generation) return null;
      updateConversation(id, (item) => ({ ...item, connection: 'unknown',
        connectionError: error instanceof ApiError ? error.message : '서버 연결을 확인하지 못했습니다. 기록은 보존되며 다시 확인할 수 있습니다.' }));
      return null;
    }
  }, [getClient, updateConversation]);

  useEffect(() => {
    mounted.current = true;
    const running = requests.current;
    return () => {
      mounted.current = false;
      for (const controller of running.values()) controller.abort();
    };
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    fetchClientConfig(apiBaseUrl, { signal: controller.signal }).then((settings) => {
      if (controller.signal.aborted) return;
      setConfig(settings); setConfigError(null);
    }).catch((error: unknown) => {
      if (controller.signal.aborted) return;
      setConfigError(error instanceof ApiError ? error.message : '서버 설정을 확인하지 못했습니다.');
    });
    return () => controller.abort();
  }, [apiBaseUrl, configAttempt]);
  useEffect(() => {
    const controller = new AbortController();
    const selected = currentState.current.conversations.find((item) => item.id === state.activeId);
    if (config && selected && selected.busy !== 'chat') void ensureSession(selected.id, controller.signal);
    return () => controller.abort();
  }, [state.activeId, config, ensureSession]);
  useEffect(() => {
    document.documentElement.dataset.theme = state.theme;
    return () => { delete document.documentElement.dataset.theme; };
  }, [state.theme]);

  const conversation = state.conversations.find((item) => item.id === state.activeId)!;
  const client = getClient(conversation.id);
  const navigationBlocked = state.conversations.some((item) => item.busy === 'upload' || item.busy === 'save');
  const readOnly = conversation.connection === 'expired';
  const ready = config !== null && conversation.connection === 'ready';
  const queryTooLong = config !== null && (Array.from(conversation.draft).length > config.query.max_chars
    || new TextEncoder().encode(conversation.draft).byteLength > config.query.max_utf8_bytes);

  function setBusy(id: string, operation: 'upload' | 'save', busy: boolean) {
    updateConversation(id, (item) => ({ ...item, busy: busy ? operation : item.busy === operation ? null : item.busy }));
  }
  function acceptManifest(id: string, manifest: UploadManifest | null) {
    updateConversation(id, (item) => {
      if (item.connection === 'expired') return item;
      if (!manifest) return { ...item, connection: 'unknown', connectionError: '첨부 상태를 다시 확인해 주세요.' };
      if (item.confirmedEpoch !== null && item.confirmedEpoch !== manifest.epoch) {
        getClient(id).manifest = null;
        return { ...item, connection: 'expired' };
      }
      return { ...item, manifest, confirmedEpoch: manifest.epoch, connection: 'ready', connectionError: null };
    });
  }

  async function sendQuestion(id: string, query: string, consumeDraft = false) {
    const selected = currentState.current.conversations.find((item) => item.id === id);
    // Attachment completion hands off the original prompt before checking HTTP.
    // Keep it recoverable until a submitted turn owns it, without replacing a newer draft.
    if (selected && !consumeDraft) updateConversation(id, (item) => ({ ...item,
      attachments: { ...item.attachments, pendingPrompt: query } }));
    if (!selected || selected.busy || requests.current.has(id) || selected.connection === 'expired' || !query.trim() || !config) return;
    if (Array.from(query).length > config.query.max_chars || new TextEncoder().encode(query).byteLength > config.query.max_utf8_bytes) return;
    const controller = new AbortController();
    requests.current.set(id, controller);
    updateConversation(id, (item) => ({ ...item, busy: 'chat' }));
    setProgress((value) => ({ ...value, [id]: '서버 연결을 확인하고 있습니다.' }));
    let turnId: string | null = null;
    try {
      if (!await ensureSession(id, controller.signal) || controller.signal.aborted) return;
      turnId = crypto.randomUUID();
      const turn: ChatTurn = { id: turnId, query, result: null, notices: [], error: null, requestId: null,
        interrupted: false, receiving: true, exportResult: null, exportError: null };
      updateConversation(id, (item) => ({ ...item, draft: consumeDraft && item.draft === query ? '' : item.draft,
        attachments: item.attachments.pendingPrompt === query ? { ...item.attachments, pendingPrompt: null, phase: 'idle' } : item.attachments,
        turns: [...item.turns, turn] }));
      const recipient = selectedSlackRecipient(currentState.current.conversations.find((item) => item.id === id)!.slack);
      for await (const incoming of getClient(id).stream(query, { signal: controller.signal, slackRecipient: recipient })) {
        if (!mounted.current || requests.current.get(id) !== controller || controller.signal.aborted) break;
        if (incoming.event === 'request_started') {
          updateConversation(id, (item) => ({ ...item, turns: item.turns.map((saved) => saved.id === turnId ? { ...saved, requestId: incoming.data.request_id } : saved) }));
        } else if (incoming.event === 'progress_snapshot') {
          setProgress((value) => ({ ...value, [id]: incoming.data.summary }));
        } else if (incoming.event === 'stage_started' || incoming.event === 'heartbeat') {
          setProgress((value) => ({ ...value, [id]: stageLabels[incoming.data.stage] ?? '답변을 준비하고 있습니다.' }));
        } else if (incoming.event === 'error') {
          updateConversation(id, (item) => ({ ...item, turns: item.turns.map((saved) => saved.id === turnId
            ? { ...saved, notices: [...saved.notices, incoming.data.message] } : saved) }));
        } else if (incoming.event === 'final_response') {
          acceptManifest(id, incoming.data.upload_manifest);
          updateConversation(id, (item) => ({ ...item, turns: item.turns.map((saved) => saved.id === turnId
            ? { ...saved, result: incoming.data, receiving: false, requestId: incoming.data.request_id || saved.requestId } : saved) }));
        }
      }
    } catch (error: unknown) {
      if (!mounted.current || requests.current.get(id) !== controller) return;
      const message = controller.signal.aborted
        ? '응답 수신을 중단했습니다. 서버 작업은 계속될 수 있습니다. 질문을 자동으로 다시 보내지 않습니다.'
        : error instanceof ApiError ? error.message : '응답을 받지 못했습니다. 처리 결과를 확인한 뒤 질문을 수정해 주세요.';
      updateConversation(id, (item) => ({ ...item, connection: 'unknown', connectionError: '다음 질문을 보내기 전에 연결을 다시 확인해 주세요.',
        turns: item.turns.map((saved) => saved.id === turnId ? { ...saved, receiving: false, interrupted: true, error: message,
          requestId: error instanceof ApiError ? error.requestId ?? saved.requestId : saved.requestId } : saved) }));
    } finally {
      if (requests.current.get(id) === controller) {
        requests.current.delete(id);
        updateConversation(id, (item) => ({ ...item, busy: item.busy === 'chat' ? null : item.busy }));
        if (mounted.current) setProgress((value) => ({ ...value, [id]: '' }));
      }
    }
  }

  const attachments = useAttachments({
    client, config, state: conversation.attachments, manifest: conversation.manifest, manifestConfirmed: ready,
    disabled: readOnly || conversation.busy === 'chat' || conversation.busy === 'save' || !config,
    onChange: (next) => updateConversation(conversation.id, (item) => ({ ...item, attachments: next })),
    ensureSession: () => ensureSession(conversation.id), onManifestChange: (manifest) => acceptManifest(conversation.id, manifest),
    onBusyChange: (busy) => setBusy(conversation.id, 'upload', busy),
    onSendPrompt: (query) => { void sendQuestion(conversation.id, query); },
    onRestorePrompt: (query) => { updateConversation(conversation.id, (item) => ({ ...item, draft: query })); input.current?.focus(); },
  });
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (conversation.busy || !ready || queryTooLong || attachments.hasPending) return;
    if (conversation.attachments.files.length) {
      const query = conversation.draft.trim() ? conversation.draft : null;
      if (query) updateConversation(conversation.id, (item) => ({ ...item, draft: '' }));
      void attachments.submit(query);
    } else if (conversation.draft.trim()) void sendQuestion(conversation.id, conversation.draft, true);
  }
  function newConversation() {
    if (navigationBlocked) return;
    const next = createConversation();
    mutate((value) => ({ ...value, activeId: next.id, conversations: [...value.conversations, next] }));
    input.current?.focus();
  }
  function reconnect() {
    if (!config) setConfigAttempt((attempt) => attempt + 1);
    void ensureSession(conversation.id);
  }
  function fillExample(prompt: string) {
    updateConversation(conversation.id, (item) => item.draft || item.busy || item.connection !== 'ready'
      || item.attachments.phase !== 'idle' || item.attachments.pendingPrompt ? item : { ...item, draft: prompt });
    input.current?.focus();
  }
  const lastTurn = conversation.turns.at(-1);
  const canContinueUploadQuestion = lastTurn?.result?.status === 'needs_input' && lastTurn.result.missing_slots.includes('upload')
    && (conversation.manifest?.files.length ?? 0) > 0 && !attachments.hasPending && conversation.attachments.files.length === 0;
  const examples = (conversation.manifest?.files.length ?? 0) > 0
    ? [docExamples[0], uploadExamples[0], docExamples[1], uploadExamples[1]] : docExamples;
  const connectionMessage = conversation.busy === 'chat' ? progress[conversation.id] || '응답을 준비하고 있습니다.'
    : readOnly ? '서버 세션이 만료되어 읽기 전용입니다. 새 채팅에서 계속할 수 있습니다.'
      : configError ?? (conversation.connection === 'checking' || !config ? '서버 연결을 확인하고 있습니다.'
        : conversation.connection === 'ready' ? '질문할 준비가 되었습니다.' : conversation.connectionError ?? '서버 연결을 다시 확인해 주세요.');

  return (
    <div className="app-layout">
      <ConversationSidebar items={state.conversations.map((item) => ({ id: item.id, title: conversationTitle(item), busy: item.busy === 'chat' }))}
        activeId={state.activeId} onSelect={(id) => { if (!navigationBlocked) mutate((value) => ({ ...value, activeId: id })); }}
        onNew={newConversation} disabled={navigationBlocked} theme={state.theme}
        onThemeChange={(theme) => mutate((value) => ({ ...value, theme }))} onModalChange={setSidebarModal}>
        <SlackSettings value={conversation.slack} baseUrl={apiBaseUrl} disabled={readOnly || conversation.busy !== null}
          onChange={(slack) => updateConversation(conversation.id, (item) => ({ ...item, slack }))} />
      </ConversationSidebar>
      <div className="app-shell" inert={sidebarModal}>
        <header className="app-header">
          <div className="brand"><span className="brand-mark" aria-hidden="true">DM</span><span>DocuMate<small>문서와 함께 생각하기</small></span></div>
          <div className="conversation-heading"><strong>{conversationTitle(conversation)}</strong>
            <small>{conversation.connection === 'ready' ? '검색 가능' : '이전 첨부 기록'} {conversation.manifest?.files.length ?? 0}개</small></div>
        </header>
        {storageError && <p className="notice" role="alert">{storageError}</p>}
        <main className="conversation" aria-label="질문과 답변">
          {conversation.turns.length === 0 && <section className="welcome">
            <p className="eyebrow">공식 문서와 나의 자료</p><h1>문서와 자료에<br />질문하세요.</h1>
            <p>사용법을 확인하거나 파일을 첨부하세요. 답변의 근거에서 사용한 원문을 확인할 수 있습니다.</p>
            {ready && !conversation.busy && !attachments.hasPending && conversation.attachments.files.length === 0 && <div className="examples" aria-label="질문 예시">
              <p>예시를 고른 뒤 확인하거나 수정하고 전송하세요.</p>
              {examples.map((prompt) => <button className="secondary-button" type="button" key={prompt}
                disabled={Boolean(conversation.draft)} onClick={() => fillExample(prompt)}>{prompt}</button>)}
            </div>}
          </section>}
          {conversation.turns.map((turn, index) => {
            const result = displayedResult(turn);
            const recovery = index === conversation.turns.length - 1 && !turn.receiving && canRecover(turn)
              && !attachments.hasPending && conversation.attachments.files.length === 0;
            return <section className="turn" key={turn.id} aria-label="대화 기록">
              <div className="user-message"><span className="message-label">내 질문</span><p>{turn.query}</p></div>
              <div className="assistant-message"><span className="message-label">DocuMate</span>
                {result && <>
                  <p className={`result-status result-${result.status}`}>{resultLabels[result.status]}</p>
                  {result.response ? <AnswerView response={result.response} turn={result} /> : <>
                    {result.message && <p className="result-message">{result.message}</p>}
                    {result.problem && result.problem.message !== result.message && <p className="notice">{result.problem.message}</p>}
                  </>}
                  {result.answer_id && result.response && <AnswerDelivery key={result.answer_id} client={client}
                    answerId={result.answer_id} response={result.response} currentExport={turn.exportResult}
                    disabled={!ready || conversation.busy !== null} onBeforeSave={async () => await ensureSession(conversation.id) !== null}
                    onBusyChange={(busy) => setBusy(conversation.id, 'save', busy)} error={turn.exportError}
                    onError={(error) => updateConversation(conversation.id, (item) => ({ ...item,
                      turns: item.turns.map((saved) => saved.id === turn.id ? { ...saved, exportError: error } : saved) }))}
                    onExportResult={(exportResult) => updateConversation(conversation.id, (item) => ({ ...item,
                      turns: item.turns.map((saved) => saved.id === turn.id ? { ...saved, exportResult, exportError: null } : saved) }))} />}
                </>}
                {turn.notices.length > 0 && <details className="transport-notices"><summary>수신 중 안내</summary>{turn.notices.map((notice, i) => <p key={i}>{notice}</p>)}</details>}
                {turn.error && <p role="alert" className="notice">{turn.error}</p>}
                {turn.receiving && <p className="pending-label">답변을 기다리고 있습니다.</p>}
                {turn.requestId && <p className="request-id">요청 ID: {turn.requestId}</p>}
                {recovery && <div className="recovery-actions">
                  {canRetry(turn) && <button type="button" className="secondary-button" disabled={!ready || conversation.busy !== null}
                    onClick={() => void sendQuestion(conversation.id, turn.query)}>같은 질문 다시 보내기</button>}
                  <button type="button" className="secondary-button" disabled={readOnly || conversation.busy !== null}
                    onClick={() => { updateConversation(conversation.id, (item) => ({ ...item, draft: turn.query })); input.current?.focus(); }}>질문 수정</button>
                </div>}
              </div>
            </section>;
          })}
        </main>
        <footer className="composer-area">
          <div className="connection-status" role="status" aria-live="polite">{connectionMessage}</div>
          {!readOnly && (conversation.connection === 'unknown' || configError) && <button type="button" className="secondary-button reconnect-button"
            disabled={conversation.busy !== null} onClick={reconnect}>연결 다시 확인</button>}
          {canContinueUploadQuestion && <button type="button" className="secondary-button" disabled={!ready || conversation.busy !== null}
            onClick={() => void sendQuestion(conversation.id, lastTurn.query)}>이전 질문 계속하기</button>}
          {attachments.panel}
          <form className="composer" onSubmit={submit}>
            <label className="visually-hidden" htmlFor="query">질문</label>
            <textarea ref={input} id="query" value={conversation.draft} readOnly={readOnly}
              onChange={(event) => updateConversation(conversation.id, (item) => ({ ...item, draft: event.target.value }))}
              placeholder="질문을 입력하세요" rows={3} aria-invalid={queryTooLong || undefined}
              aria-describedby={queryTooLong ? 'query-limit' : 'composer-hint'}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey && !event.ctrlKey && !event.altKey && !event.metaKey && !event.nativeEvent.isComposing) {
                  event.preventDefault(); event.currentTarget.form?.requestSubmit();
                }
              }} />
            <div className="composer-actions"><p id="composer-hint">Enter로 전송 · Shift+Enter로 줄바꿈</p>
              {conversation.busy === 'chat'
                ? <button type="button" className="secondary-button" onClick={() => requests.current.get(conversation.id)?.abort()}>응답 수신 중단</button>
                : <button type="submit" disabled={!ready || conversation.busy !== null || queryTooLong || attachments.hasPending
                  || (!conversation.draft.trim() && !conversation.attachments.files.length)}>{conversation.draft.trim() ? '질문 보내기' : conversation.attachments.files.length ? '첨부만 전송' : '질문 보내기'}</button>}
            </div>
          </form>
          {queryTooLong && <p id="query-limit" role="alert" className="notice">질문이 서버의 길이 제한을 넘었습니다. 내용을 나누어 보내 주세요.</p>}
        </footer>
      </div>
    </div>
  );
}

function displayedResult(turn: ChatTurn): AgentResponse | null {
  const result = turn.result;
  if (!result?.response || !turn.exportResult) return result;
  const saveRecovered = turn.exportResult.status === 'completed' && result.problem?.code === 'save_failed';
  return { ...result, ...(saveRecovered ? { status: 'completed' as const, problem: null, message: '' } : {}),
    response: { ...result.response, actions: [
      ...result.response.actions.filter((action) => action.kind !== 'save_text'),
      ...turn.exportResult.response.actions.filter((action) => action.kind === 'save_text'),
    ] } };
}
function canRecover(turn: ChatTurn): boolean {
  if (!turn.result) return turn.interrupted || turn.error !== null;
  return (turn.result.status === 'failed' || turn.result.status === 'partial')
    && turn.result.problem?.code !== 'save_failed' && !(turn.result.response?.actions.length);
}
function canRetry(turn: ChatTurn): boolean {
  const problem = turn.result?.problem;
  return Boolean(problem && ['retry_later', 'supply_information'].includes(problem.next_action)
    && !['action_postprocess', 'response_assembly'].includes(problem.stage));
}
