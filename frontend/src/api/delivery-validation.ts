import { ApiError } from './errors';
import type { AnswerExportResult, SlackDefault } from './types';
import { modelValidator, validateActionReceipts } from './validation';
import { slackRecipientPatterns } from '../slackRecipients';

const slackDefaultValidator = modelValidator('SlackDefault');
const answerExportValidator = modelValidator('AnswerExportResult');

export function parseSlackDefault(value: unknown): SlackDefault {
  if (!slackDefaultValidator(value)) {
    throw invalidResponse();
  }
  const selector = value.selector;
  const failure = value.failure;
  if ((selector !== null && failure !== null)
    || (selector !== null && !slackRecipientPatterns[selector.kind].test(selector.value))) {
    throw invalidResponse();
  }
  return value;
}

export function parseAnswerExport(value: unknown, answerId: string): AnswerExportResult {
  if (!answerExportValidator(value) || value.answer_id !== answerId) throw invalidResponse();
  const receipt = value.receipt;
  validateActionReceipts([receipt]);
  if (receipt.kind !== 'save_text') throw invalidResponse();
  const complete = receipt.status === 'success';
  if ((value.status === 'completed') !== complete) throw invalidResponse();
  return value;
}

function invalidResponse(): ApiError {
  return new ApiError('invalid_response', '서버의 전달 결과를 확인하지 못했습니다.');
}
