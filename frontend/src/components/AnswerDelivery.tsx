import { useEffect, useRef, useState } from 'react';
import { AgentSessionClient, ApiError } from '../api/client';
import type { AnswerExportResult, AnswerResponse } from '../api/types';

type Props = {
  client: AgentSessionClient;
  answerId: string;
  response: AnswerResponse;
  currentExport: AnswerExportResult | null;
  onExportResult: (result: AnswerExportResult) => void;
  onBusyChange?: (busy: boolean) => void;
  onBeforeSave?: () => Promise<boolean>;
  disabled?: boolean;
  error?: string | null;
  onError?: (message: string | null) => void;
};

export function AnswerDelivery({ client, answerId, response, currentExport, onExportResult,
  onBusyChange, onBeforeSave, disabled = false, error, onError }: Props) {
  const active = useRef<AbortController | null>(null);
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const receipt = (currentExport?.response ?? response).actions.find((action) => action.kind === 'save_text');
  const artifact = receipt?.artifact;
  const expiresAt = artifact ? artifact.expires_at * 1000 : null;
  const expired = expiresAt !== null && expiresAt <= now;
  const ready = receipt?.status === 'success' && receipt.verification === 'verified' && artifact && !expired;
  const message = error ?? localError;
  const retry = Boolean(message) || currentExport?.status === 'failed'
    || Boolean(receipt && (receipt.status !== 'success' || receipt.verification !== 'verified' || !artifact));

  useEffect(() => () => active.current?.abort(), []);
  useEffect(() => {
    if (expiresAt === null || expiresAt <= now) return;
    const timer = setTimeout(() => setNow(Date.now()), Math.min(expiresAt - Date.now(), 2_147_483_647));
    return () => clearTimeout(timer);
  }, [expiresAt, now]);

  async function save() {
    if (active.current || disabled || unavailable) return;
    const controller = new AbortController();
    active.current = controller;
    setBusy(true);
    onBusyChange?.(true);
    setLocalError(null);
    onError?.(null);
    try {
      if (onBeforeSave && !await onBeforeSave()) return;
      const result = await client.saveAnswer(answerId, { signal: controller.signal });
      if (!controller.signal.aborted) onExportResult(result);
    } catch (caught) {
      if (controller.signal.aborted) return;
      const description = caught instanceof ApiError ? caught.message
        : '저장 결과를 확인하지 못했습니다. 기존 답변은 유지됩니다. 같은 답변으로 다시 시도해 주세요.';
      if (caught instanceof ApiError && ['answer_unavailable', 'session_expired', 'session_unavailable'].includes(caught.code)) {
        setUnavailable(true);
      }
      if (onError) onError(description);
      else setLocalError(description);
    } finally {
      active.current = null;
      setBusy(false);
      onBusyChange?.(false);
    }
  }

  return (
    <section className="answer-delivery" aria-label="이 답변 TXT 저장">
      {message && <p role="alert">{message}</p>}
      {unavailable && <p>서버 기록을 사용할 수 없어 이 답변을 새로 저장할 수 없습니다. 화면의 본문과 근거는 계속 확인할 수 있습니다.</p>}
      {expired && <p>저장 파일의 보관 기한이 지났습니다. 답변은 유지됩니다. 내려받으려면 TXT를 다시 만들어 주세요.</p>}
      {ready ? <>
        <p>다운로드 가능 기한: <time dateTime={new Date(artifact.expires_at * 1000).toISOString()}>
          {new Date(artifact.expires_at * 1000).toLocaleString('ko-KR')}</time></p>
        <a href={client.downloadUrl(artifact.filename)} download={artifact.filename}>TXT 다운로드</a>
        <button type="button" disabled={disabled || busy || unavailable} onClick={() => void save()}>
          {busy ? '저장 확인 중…' : '저장 파일 다시 확인'}
        </button>
      </> : <>
        {retry && !unavailable && <p>“저장 다시 시도”는 이 답변의 본문과 출처를 그대로 사용합니다.</p>}
        <button type="button" disabled={disabled || busy || unavailable} onClick={() => void save()}>
          {busy ? 'TXT 저장 중…' : retry ? '저장 다시 시도' : expired ? 'TXT 다시 만들기' : 'TXT 파일 만들기'}
        </button>
      </>}
    </section>
  );
}
