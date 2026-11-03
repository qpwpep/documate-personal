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
  slack: null, message: '저장했습니다.', error: null, operation, artifact,
  verification: 'verified', error_code: null,
};
const completed = {
  status: 'completed', answer_id: 'answer-1', request_id: 'export-request-1', message: '저장했습니다.',
  response: { ...fixtures.responses.completed.response, actions: [receipt] },
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

  it('normalizes the confirmed selector the same way as the server', () => {
    expect(parseSlackDefault({ selector: { kind: 'user', value: ' U123 ' }, failure: null })).toEqual({
      selector: { kind: 'user', value: 'U123' }, failure: null,
    });
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
