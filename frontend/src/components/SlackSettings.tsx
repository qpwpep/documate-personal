import { useEffect, useId, useState } from 'react';
import { fetchSlackDefault } from '../api/client';
import type { SlackDefault } from '../api/types';
import { slackRecipientPatterns } from '../slackRecipients';
import type { SlackSettingsValue } from './SlackSettingsState';

const recipientLabels = { channel: '채널', user: '사용자', email: '이메일' };
const recipientFields = {
  channel: { placeholder: 'C0123456789',
    guidance: '채널 이름 대신 C, G 또는 D로 시작하는 채널 ID를 입력하세요.',
    invalid: '채널 이름은 사용할 수 없습니다. C, G 또는 D로 시작하는 채널 ID를 입력하세요.' },
  user: { placeholder: 'U0123456789',
    guidance: '표시 이름 대신 U 또는 W로 시작하는 멤버 ID를 입력하세요. 이메일로도 지정할 수 있습니다.',
    invalid: '표시 이름은 사용할 수 없습니다. U 또는 W로 시작하는 멤버 ID를 입력하세요.' },
  email: { placeholder: 'name@example.com',
    guidance: '받는 사람이 Slack 워크스페이스에 등록한 이메일을 입력하세요.',
    invalid: '이메일 형식을 확인해 주세요. 예: name@example.com' },
};

type DefaultLookup = { baseUrl: string; revision: number; value: SlackDefault | null };
type Props = {
  value: SlackSettingsValue;
  onChange: (value: SlackSettingsValue) => void;
  baseUrl?: string;
  disabled?: boolean;
};

export function SlackSettings({ value, onChange, baseUrl = '', disabled = false }: Props) {
  const fieldId = useId();
  const [revision, setRevision] = useState(0);
  const [lookup, setLookup] = useState<DefaultLookup | null>(null);
  const loading = lookup?.baseUrl !== baseUrl || lookup.revision !== revision;

  useEffect(() => {
    const controller = new AbortController();
    fetchSlackDefault(baseUrl, { signal: controller.signal }).then(
      (result) => { if (!controller.signal.aborted) setLookup({ baseUrl, revision, value: result }); },
      () => { if (!controller.signal.aborted) setLookup({ baseUrl, revision, value: null }); },
    );
    return () => controller.abort();
  }, [baseUrl, revision]);

  const kind = value.kind === 'omitted' ? null : value.kind;
  const field = kind ? recipientFields[kind] : null;
  const valid = kind !== null && field !== null && slackRecipientPatterns[kind].test(value.drafts[kind].trim());

  return (
    <details className="slack-settings">
      <summary>Slack 수신자 설정</summary>
      <p>수신자를 설정한 뒤 질문 입력창에 “방금 답변을 Slack으로 보내줘”라고 요청하세요. 설정만으로 전송하지 않습니다.</p>
      <fieldset className="settings-options" disabled={disabled}>
        <legend>수신자 유형</legend>
        {(['omitted', 'channel', 'user', 'email'] as const).map((kind) => <label key={kind}>
          <input type="radio" name={`${fieldId}-kind`} value={kind} checked={value.kind === kind}
            onChange={() => onChange({ ...value, kind })} />
          {kind === 'omitted' ? '직접 지정 안 함' : recipientLabels[kind]}
        </label>)}
      </fieldset>
      {kind && field ? <>
        <p id={`${fieldId}-guidance`}>{field.guidance}</p>
        <label htmlFor={`${fieldId}-recipient`}>Slack 수신자</label>
        <input id={`${fieldId}-recipient`} value={value.drafts[kind]} placeholder={field.placeholder}
          disabled={disabled} spellCheck={false} autoComplete="off" aria-invalid={!valid}
          aria-describedby={`${fieldId}-guidance ${fieldId}-feedback`}
          onChange={(event) => onChange({ ...value, drafts: { ...value.drafts, [kind]: event.target.value } })} />
        <p id={`${fieldId}-feedback`} role="status">
          {valid ? '수신자 형식을 확인했습니다. 실제 수신 가능 여부는 전송할 때 확인합니다.'
            : `${field.invalid} 일반 질문은 계속할 수 있습니다.`}
        </p>
        {kind !== 'email' && <details>
          <summary>수신자 ID 찾는 방법</summary>
          <p>{kind === 'channel'
            ? 'Slack에서 대상 채널을 열고 상단 채널 이름을 누르세요. 채널 상세 화면 맨 아래의 채널 ID를 복사해 입력하세요.'
            : 'Slack에서 받는 사람의 프로필을 열고 더 보기 → 멤버 ID 복사를 선택하세요. ID를 찾기 어렵다면 이메일로 지정할 수 있습니다.'}</p>
        </details>}
      </> : <>
        {loading ? <p role="status">기본 수신자를 확인하고 있습니다.</p>
          : <DefaultRecipient value={lookup?.value ?? null} />}
        <button type="button" disabled={disabled || loading} onClick={() => setRevision((previous) => previous + 1)}>
          기본 수신자 다시 확인
        </button>
      </>}
    </details>
  );
}

function DefaultRecipient({ value }: { value: SlackDefault | null }) {
  if (value === null) return <p role="status">기본 수신자를 확인하지 못했습니다. 기본 수신자가 없는 상태와는 다릅니다. 수신자를 직접 지정하거나 다시 확인해 주세요.</p>;
  if (value.failure) return <>
    <p role="status">{value.failure.message}</p>
    <p>채널·사용자·이메일로 수신자를 직접 지정하면 기본 설정을 사용하지 않습니다.</p>
  </>;
  if (!value.selector) return <p role="status">기본 수신자가 없습니다. 채널·사용자·이메일을 선택하거나 질문에 수신자를 지정해 주세요. 수신자가 정해져야 Slack으로 전송할 수 있습니다.</p>;
  return <>
    <p role="status">기본 수신자: {recipientLabels[value.selector.kind]} <strong>{value.selector.value}</strong></p>
    <p>질문에도 수신자를 지정하지 않은 새 전송은 이 대상으로 보냅니다. 사용자·이메일 수신자에게는 개인 메시지로 전송합니다.</p>
    <p>보류 중인 전송은 앞서 확정한 수신자를 유지합니다. 실제 수신 가능 여부는 전송할 때 확인합니다.</p>
  </>;
}
