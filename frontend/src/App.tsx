import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { AgentSessionClient, ApiError, fetchClientConfig } from './api/client';
import type { AgentResponse, ClientConfig } from './api/types';
import { AnswerView } from './components/AnswerView';

interface Turn {
  id: string;
  query: string;
  result: AgentResponse | null;
  notices: string[];
  error: string | null;
  requestId: string | null;
}

const resultLabels = {
  completed: '답변 받음',
  partial: '일부 처리',
  needs_input: '추가 정보 필요',
  failed: '요청 실패',
  refused: '응답 거절',
};

const stageLabels: Record<string, string> = {
  planner: '질문을 확인하고 있습니다.',
  retrieve: '자료를 찾고 있습니다.',
  retrieval: '자료를 찾고 있습니다.',
  synthesize: '답변을 작성하고 있습니다.',
  synthesis: '답변을 작성하고 있습니다.',
  validation: '답변과 근거를 확인하고 있습니다.',
};

export function App({ apiBaseUrl = '' }: { apiBaseUrl?: string }) {
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());
  const client = useMemo(() => new AgentSessionClient({ sessionId, baseUrl: apiBaseUrl }), [sessionId, apiBaseUrl]);
  const [config, setConfig] = useState<ClientConfig | null>(null);
  const [connection, setConnection] = useState<'connecting' | 'ready' | 'failed'>('connecting');
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [connectionAttempt, setConnectionAttempt] = useState(0);
  const [draft, setDraft] = useState('');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState('');
  const activeRequest = useRef<AbortController | null>(null);
  const input = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      fetchClientConfig(apiBaseUrl, { signal: controller.signal }),
      client.refreshUploads({ signal: controller.signal }),
    ]).then(([settings]) => {
      if (controller.signal.aborted) return;
      setConfig(settings);
      setConnection('ready');
      setConnectionError(null);
    }).catch((error: unknown) => {
      if (controller.signal.aborted) return;
      setConnection('failed');
      setConnectionError(error instanceof ApiError ? error.message : '서버에 연결하지 못했습니다. 실행 상태를 확인해 주세요.');
    });
    return () => controller.abort();
  }, [apiBaseUrl, client, connectionAttempt]);

  useEffect(() => () => activeRequest.current?.abort(), []);

  const queryTooLong = config !== null && (
    Array.from(draft).length > config.query.max_chars
    || new TextEncoder().encode(draft).byteLength > config.query.max_utf8_bytes
  );

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (activeRequest.current || busy || connection !== 'ready' || !draft.trim() || queryTooLong) return;
    const turnId = crypto.randomUUID();
    const query = draft;
    const controller = new AbortController();
    activeRequest.current = controller;
    setTurns((current) => [...current, { id: turnId, query, result: null, notices: [], error: null, requestId: null }]);
    setDraft('');
    setBusy(true);
    setProgress('질문을 보내고 있습니다.');
    try {
      for await (const incoming of client.stream(query, { signal: controller.signal })) {
        if (incoming.event === 'request_started') {
          setTurns((current) => current.map((turn) => turn.id === turnId
            ? { ...turn, requestId: incoming.data.request_id } : turn));
        } else if (incoming.event === 'progress_snapshot') {
          setProgress(incoming.data.summary);
        } else if (incoming.event === 'stage_started' || incoming.event === 'heartbeat') {
          setProgress(stageLabels[incoming.data.stage] ?? '답변을 준비하고 있습니다.');
        } else if (incoming.event === 'error') {
          setTurns((current) => current.map((turn) => turn.id === turnId
            ? { ...turn, notices: [...turn.notices, incoming.data.message] } : turn));
        } else if (incoming.event === 'final_response') {
          setTurns((current) => current.map((turn) => turn.id === turnId
            ? { ...turn, result: incoming.data, requestId: incoming.data.request_id || turn.requestId } : turn));
        }
      }
    } catch (error: unknown) {
      const message = controller.signal.aborted
        ? '응답 수신을 중단했습니다. 서버 작업은 계속될 수 있습니다. 질문을 자동으로 다시 보내지 않습니다.'
        : error instanceof ApiError ? error.message : '응답을 받지 못했습니다. 연결을 확인한 뒤 다시 시도해 주세요.';
      setTurns((current) => current.map((turn) => turn.id === turnId
        ? { ...turn, error: message, requestId: error instanceof ApiError ? error.requestId ?? turn.requestId : turn.requestId } : turn));
      setConnection('failed');
      setConnectionError('다음 질문을 보내기 전에 서버 연결과 첨부 상태를 다시 확인해 주세요.');
    } finally {
      activeRequest.current = null;
      setBusy(false);
      setProgress('');
    }
  }

  function reconnect() {
    setConnection('connecting');
    setConnectionError(null);
    setConnectionAttempt((attempt) => attempt + 1);
  }

  function newConversation() {
    setTurns([]);
    setDraft('');
    setConnection('connecting');
    setConnectionError(null);
    setSessionId(crypto.randomUUID());
    input.current?.focus();
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <a className="brand" href="/app/" aria-label="DocuMate 처음으로">
          <span className="brand-mark" aria-hidden="true">DM</span>
          <span>DocuMate<small>문서와 함께 생각하기</small></span>
        </a>
        <button className="secondary-button" type="button" onClick={newConversation} disabled={busy}>새 대화</button>
      </header>

      <main className="conversation" aria-label="질문과 답변">
        {turns.length === 0 && (
          <section className="welcome">
            <p className="eyebrow">공식 문서를 바탕으로</p>
            <h1>궁금한 것을<br />함께 풀어보세요.</h1>
            <p>개념 설명부터 코드 사용법까지, 확인하고 싶은 내용을 질문해 주세요.</p>
            <p className="preview-note">새 화면을 준비하고 있습니다. 첨부와 자세한 근거 확인은 기존 화면에서 이용할 수 있습니다. 이 화면의 대화는 새로고침하면 초기화됩니다.</p>
          </section>
        )}

        {turns.map((turn) => (
          <section className="turn" key={turn.id} aria-label="대화 기록">
            <div className="user-message"><span className="message-label">내 질문</span><p>{turn.query}</p></div>
            <div className="assistant-message">
              <span className="message-label">DocuMate</span>
              {turn.result && (
                <>
                  <p className={`result-status result-${turn.result.status}`}>{resultLabels[turn.result.status]}</p>
                  {turn.result.message && <p className="result-message">{turn.result.message}</p>}
                  {turn.result.problem && turn.result.problem.message !== turn.result.message && <p className="notice">{turn.result.problem.message}</p>}
                  {turn.result.response && <AnswerView response={turn.result.response} />}
                </>
              )}
              {turn.notices.length > 0 && (
                <details className="transport-notices">
                  <summary>수신 중 안내</summary>
                  {turn.notices.map((notice, index) => <p key={index}>{notice}</p>)}
                </details>
              )}
              {turn.error && <p role="alert" className="notice">{turn.error}</p>}
              {!turn.result && !turn.error && <p className="pending-label">답변을 기다리고 있습니다.</p>}
              {turn.requestId && <p className="request-id">요청 ID: {turn.requestId}</p>}
            </div>
          </section>
        ))}
      </main>

      <footer className="composer-area">
        <div className="connection-status" role="status" aria-live="polite">
          {busy ? progress : connection === 'connecting' ? '서버 연결을 확인하고 있습니다.' : connection === 'ready' ? '질문할 준비가 되었습니다.' : connectionError}
        </div>
        {connection === 'failed' && <button type="button" className="secondary-button reconnect-button" onClick={reconnect}>연결 다시 확인</button>}
        <form className="composer" onSubmit={(event) => { void submit(event); }}>
          <label className="visually-hidden" htmlFor="query">질문</label>
          <textarea
            ref={input}
            id="query"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="질문을 입력하세요"
            rows={3}
            aria-invalid={queryTooLong || undefined}
            aria-describedby={queryTooLong ? 'query-limit' : 'composer-hint'}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey && !event.ctrlKey && !event.altKey && !event.metaKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
          />
          <div className="composer-actions">
            <p id="composer-hint">Enter로 전송 · Shift+Enter로 줄바꿈</p>
            {busy
              ? <button type="button" className="secondary-button" onClick={() => activeRequest.current?.abort()}>응답 수신 중단</button>
              : <button type="submit" disabled={connection !== 'ready' || !draft.trim() || queryTooLong}>질문 보내기</button>}
          </div>
        </form>
        {queryTooLong && <p id="query-limit" role="alert" className="notice">질문이 서버의 길이 제한을 넘었습니다. 내용을 나누어 보내 주세요.</p>}
      </footer>
    </div>
  );
}
