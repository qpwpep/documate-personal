import { ApiError } from './errors';
import type { AnswerExportResult, SlackDefault } from './types';
import { modelValidator, normalizeAnswer } from './validation';
import { slackRecipientPatterns } from '../slackRecipients';

const slackDefaultValidator = modelValidator('SlackDefault');
const answerExportValidator = modelValidator('AnswerExportResult');

export function parseSlackDefault(value: unknown): SlackDefault {
  if (!slackDefaultValidator(value) || !('selector' in value) || !('failure' in value)) {
    throw invalidResponse();
  }
  const selector = value.selector == null ? null : {
    ...value.selector,
    value: value.selector.value.trim(),
  };
  const failure = value.failure ?? null;
  if ((selector !== null && failure !== null)
    || (selector !== null && !slackRecipientPatterns[selector.kind].test(selector.value))) {
    throw invalidResponse();
  }
  return { selector, failure };
}

export function parseAnswerExport(value: unknown, answerId: string): AnswerExportResult {
  if (!answerExportValidator(value) || value.answer_id !== answerId) throw invalidResponse();
  const response = normalizeAnswer(value.response);
  const receipts = response.actions.filter((receipt) => receipt.kind === 'save_text');
  if (receipts.length !== 1) throw invalidResponse();
  const receipt = receipts[0];
  const artifact = receipt.artifact;
  const operation = receipt.operation;
  const complete = receipt.status === 'success' && receipt.verification === 'verified'
    && artifact != null && operation != null;
  if ((value.status === 'completed') !== complete) throw invalidResponse();
  if (artifact != null && (
    /[/\\:]/.test(artifact.filename) || !artifact.filename.endsWith('.txt')
    || artifact.expires_at <= artifact.created_at
  )) throw invalidResponse();
  if (receipt.verification === 'verified' && (
    !complete || !receipt.file_path || artifact == null || operation == null
    || artifact.sha256 !== operation.payload_sha256 || artifact.byte_count !== operation.byte_count
  )) throw invalidResponse();
  return { ...value, response };
}

function invalidResponse(): ApiError {
  return new ApiError('invalid_response', '서버의 전달 결과를 확인하지 못했습니다.');
}
