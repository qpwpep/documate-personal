// @vitest-environment node
import { describe, expect, it } from 'vitest';

import fixtures from '../../../tests/fixtures/web_contracts/agent_responses.json';
import { parseAnswerExport, parseSlackDefault } from './delivery-validation';
import { parseAgentResponse } from './validation';

const failure = {
  stage: 'selection', code: 'configuration_error', message: '기본 수신자 설정을 확인해 주세요.',
  next_action: 'fix_configuration', retry_after_seconds: null, slack_error: null,
};
const operation = {
  operation_id: 'save-operation-1', session_id: 'session-1', request_id: 'source-request',
  contract_revision: 1, target_kind: 'copy_answer',
  source_hash: fixtures.responses.completed.response.content_hash,
  answer_hash: fixtures.responses.completed.response.content_hash,
  export_profile: 'answer-text-v1/utf-8-sig', payload_sha256: 'a'.repeat(64), byte_count: 42,
};
const artifact = {
  artifact_id: 'b'.repeat(64), filename: 'answer.txt', sha256: 'a'.repeat(64),
  byte_count: 42, created_at: 1000, expires_at: 2000,
};
const receipt = {
  invocation_id: null, kind: 'save_text', status: 'success', file_path: 'output/save_text/answer.txt',
  slack: null, previous_slack: null, message: '저장했습니다.', error: null, operation, artifact,
  verification: 'verified', error_code: null,
};
const completed = {
  status: 'completed', answer_id: 'answer-1', request_id: 'export-request-1', message: '저장했습니다.',
  response: { ...fixtures.responses.completed.response, actions: [receipt] },
};

const selectedRecipient = {
  request_id: 'slack-request-1', source: 'user_text', selector: { kind: 'channel', value: 'C123' },
};
const sentDelivery = {
  invocation_id: 'slack-call-1',
  operation: { operation_id: 'slack-operation-1', request_id: 'slack-request-1', payload_sha256: 'c'.repeat(64), byte_count: 42 },
  intent: { state: 'explicit', selector: selectedRecipient.selector, evidence_ids: ['request-evidence-1'] },
  selection: selectedRecipient, resolved_user_id: null, target: { channel_id: 'C123', user_id: null },
  status: 'sent', failure: null, message_ts: '100.200',
};
const uncertainDelivery = {
  ...sentDelivery, status: 'unknown', message_ts: null,
  failure: { stage: 'send', code: 'delivery_unknown', message: '전달 여부를 확인해 주세요.',
    next_action: 'verify_delivery', retry_after_seconds: null, slack_error: null },
};
const slackReceipt = {
  ...receipt, kind: 'slack_notify', file_path: null, slack: sentDelivery,
  operation: null, artifact: null, verification: null, message: '전송했습니다.',
};

describe('Slack default response', () => {
  it.each([
    { selector: null, failure: null },
    { selector: { kind: 'channel', value: 'C123' }, failure: null },
    { selector: { kind: 'user', value: 'U123' }, failure: null },
    { selector: { kind: 'email', value: 'person@example.com' }, failure: null },
    { selector: null, failure },
  ])('preserves a confirmed default state', (value) => {
    expect(parseSlackDefault(value)).toEqual(value);
  });

  it.each([
    {},
    { selector: null },
    { selector: { kind: 'user', value: 'wrong' }, failure: null },
    { selector: { kind: 'channel', value: '' }, failure: null },
    { selector: { kind: 'email', value: 'one@example.com;two@example.com' }, failure: null },
    { selector: { kind: 'user', value: 'U123' }, failure },
  ])('rejects an unknown or contradictory default instead of treating it as missing', (value) => {
    expect(() => parseSlackDefault(value)).toThrowError(expect.objectContaining({ code: 'invalid_response' }));
  });

  it('rejects a noncanonical confirmed selector instead of repairing the response', () => {
    expect(() => parseSlackDefault({ selector: { kind: 'user', value: ' U123 ' }, failure: null }))
      .toThrowError(expect.objectContaining({ code: 'invalid_response' }));
  });
});

describe('answer export response', () => {
  it.each([null, 'unverifiable'])('rejects a legacy successful save with verification %s in a final response', (verification) => {
    const result = {
      ...fixtures.responses.completed,
      response: { ...completed.response, actions: [{ ...receipt, verification, artifact: null, operation: null }] },
    };

    expect(() => parseAgentResponse(result)).toThrowError(expect.objectContaining({ code: 'invalid_response' }));
  });

  it('preserves the selected answer and its verified save identity', () => {
    expect(parseAnswerExport(completed, 'answer-1')).toEqual(completed);
  });

  it('retains a failed save receipt and its original answer', () => {
    const failed = {
      ...completed,
      status: 'failed',
      response: {
        ...completed.response,
        actions: [{ ...receipt, status: 'error', file_path: null, artifact: null, verification: 'failed', error: '저장 실패' }],
      },
    };
    expect(parseAnswerExport(failed, 'answer-1')).toEqual(failed);
  });

  it('rejects a response belonging to another selected answer', () => {
    expect(() => parseAnswerExport(completed, 'answer-2')).toThrowError(expect.objectContaining({ code: 'invalid_response' }));
  });

  it.each([
    { ...completed, response: { ...completed.response, actions: [] } },
    { ...completed, response: { ...completed.response, actions: [receipt, receipt] } },
    { ...completed, status: 'failed' },
    { ...completed, response: { ...completed.response, actions: [{ ...receipt, artifact: null }] } },
    { ...completed, response: { ...completed.response, actions: [{ ...receipt, operation: null }] } },
    { ...completed, response: { ...completed.response, actions: [{ ...receipt, verification: 'unverifiable' }] } },
    { ...completed, response: { ...completed.response, actions: [{ ...receipt, file_path: null }] } },
    { ...completed, response: { ...completed.response, actions: [{ ...receipt, artifact: { ...artifact, sha256: 'c'.repeat(64) } }] } },
    { ...completed, response: { ...completed.response, actions: [{ ...receipt, artifact: { ...artifact, byte_count: 41 } }] } },
  ])('rejects contradictory completion or save binding', (value) => {
    expect(() => parseAnswerExport(value, 'answer-1')).toThrowError(expect.objectContaining({ code: 'invalid_response' }));
  });

  it.each([
    { ...artifact, filename: '../answer.txt' },
    { ...artifact, filename: 'folder\\answer.txt' },
    { ...artifact, filename: 'C:answer.txt' },
    { ...artifact, filename: 'answer.html' },
    { ...artifact, expires_at: artifact.created_at },
  ])('rejects an invalid download artifact location or retention interval', (value) => {
    const result = { ...completed, response: { ...completed.response, actions: [{ ...receipt, artifact: value }] } };
    expect(() => parseAnswerExport(result, 'answer-1')).toThrowError(expect.objectContaining({ code: 'invalid_response' }));
  });
});

describe('Slack delivery response', () => {
  it.each([
    { name: 'confirmed delivery', action: slackReceipt },
    { name: 'uncertain delivery', action: { ...slackReceipt, status: 'unknown', slack: uncertainDelivery } },
    { name: 'previous uncertain delivery', action: { ...slackReceipt, status: 'skipped', slack: null,
      previous_slack: uncertainDelivery, message: '이전 전송을 확인하기 전에는 새 답변을 보내지 않습니다.' } },
    { name: 'invalid recipient', action: { ...slackReceipt, status: 'skipped', slack: {
      ...sentDelivery, status: 'not_sent', message_ts: null, failure,
    } } },
  ])('preserves $name and its bound operation', ({ action }) => {
    const result = { ...fixtures.responses.completed, response: { ...completed.response, actions: [action] } };

    expect(parseAgentResponse(result)).toEqual(result);
  });

  it.each([
    { name: 'sent without operation', action: { ...slackReceipt, slack: { ...sentDelivery, operation: null } } },
    { name: 'unknown without operation', action: { ...slackReceipt, status: 'unknown', slack: { ...uncertainDelivery, operation: null } } },
    { name: 'operation for another request', action: { ...slackReceipt, slack: {
      ...sentDelivery, operation: { ...sentDelivery.operation, request_id: 'other-request' },
    } } },
    { name: 'operation without selection', action: { ...slackReceipt, slack: { ...sentDelivery, selection: null } } },
    { name: 'sent without target', action: { ...slackReceipt, slack: { ...sentDelivery, target: null } } },
    { name: 'sent without timestamp', action: { ...slackReceipt, slack: { ...sentDelivery, message_ts: ' ' } } },
    { name: 'sent with failure', action: { ...slackReceipt, slack: { ...sentDelivery, failure } } },
    { name: 'unknown without failure', action: { ...slackReceipt, status: 'unknown', slack: { ...uncertainDelivery, failure: null } } },
    { name: 'unknown without target', action: { ...slackReceipt, status: 'unknown', slack: { ...uncertainDelivery, target: null } } },
    { name: 'unknown before send', action: { ...slackReceipt, status: 'unknown', slack: {
      ...uncertainDelivery, failure: { ...uncertainDelivery.failure, stage: 'lookup' },
    } } },
    { name: 'unknown with retry instruction', action: { ...slackReceipt, status: 'unknown', slack: {
      ...uncertainDelivery, failure: { ...uncertainDelivery.failure, next_action: 'retry_same_target' },
    } } },
    { name: 'unconfirmed timestamp', action: { ...slackReceipt, status: 'unknown', slack: { ...uncertainDelivery, message_ts: '100.200' } } },
    { name: 'uncertainty disguised as failure', action: { ...slackReceipt, status: 'error', slack: { ...uncertainDelivery, status: 'not_sent' } } },
    { name: 'not sent without failure', action: { ...slackReceipt, status: 'error', slack: {
      ...sentDelivery, status: 'not_sent', message_ts: null,
    } } },
    { name: 'successful receipt for unknown delivery', action: { ...slackReceipt, slack: uncertainDelivery } },
    { name: 'failed receipt for sent delivery', action: { ...slackReceipt, status: 'error' } },
    { name: 'receipt without delivery', action: { ...slackReceipt, slack: null } },
    { name: 'receipt with pending delivery', action: { ...slackReceipt, status: 'skipped', slack: {
      ...sentDelivery, status: 'pending', message_ts: null,
    } } },
    { name: 'send failure marked skipped', action: { ...slackReceipt, status: 'skipped', slack: {
      ...sentDelivery, status: 'not_sent', message_ts: null,
      failure: { ...failure, stage: 'send' },
    } } },
    { name: 'save containing delivery', action: { ...receipt, slack: sentDelivery } },
    { name: 'save containing previous delivery', action: { ...receipt, previous_slack: uncertainDelivery } },
    { name: 'previous delivery on successful receipt', action: { ...slackReceipt, slack: null, previous_slack: uncertainDelivery } },
    { name: 'previous and current delivery together', action: { ...slackReceipt, status: 'skipped', previous_slack: uncertainDelivery } },
    { name: 'previous confirmed delivery', action: { ...slackReceipt, status: 'skipped', slack: null, previous_slack: sentDelivery } },
    { name: 'previous delivery without message', action: { ...slackReceipt, status: 'skipped', slack: null,
      previous_slack: uncertainDelivery, message: '' } },
    { name: 'previous delivery without operation', action: { ...slackReceipt, status: 'skipped', slack: null,
      previous_slack: { ...uncertainDelivery, operation: null } } },
  ])('rejects $name instead of accepting contradictory action state', ({ action }) => {
    const result = { ...fixtures.responses.completed, response: { ...completed.response, actions: [action] } };

    expect(() => parseAgentResponse(result)).toThrowError(expect.objectContaining({ code: 'invalid_response' }));
  });
});
