import Ajv2020, { type ValidateFunction } from 'ajv/dist/2020';

import type { components } from './generated';
import openapi from './openapi.json';
import { ApiError } from './errors';
import type { AgentResponse, AgentStreamEvent, AnswerResponse, ClientConfig, StagedUpload, UploadManifest } from './types';

// OpenAPI annotations are not JSON Schema assertions. Keep every schema
// constraint, but accept the document's OpenAPI-specific annotation keywords.
const validator = new Ajv2020({ strictSchema: false, strictTypes: false });
validator.addSchema(openapi, 'documate');

export function modelValidator<Name extends keyof components['schemas']>(name: Name): ValidateFunction<components['schemas'][Name]> {
  return validator.compile<components['schemas'][Name]>({ $ref: `documate#/components/schemas/${name}` });
}

const configValidator = modelValidator('ClientConfig');
const manifestValidator = modelValidator('UploadManifest');
const stagedUploadValidator = modelValidator('StagedUpload');
const responseValidator = modelValidator('AgentResponse');
const eventSchemas = openapi.paths['/agent/stream'].post.responses['200'].content['text/event-stream']['x-sse-events'];
const eventValidators = new Map<string, ValidateFunction>(
  Object.entries(eventSchemas).filter(([event]) => event !== 'final_response').map(([event, schema]) => [event, validator.compile({
    ...schema,
    components: openapi.components,
  })]),
);

export function parseClientConfig(value: unknown): ClientConfig {
  if (!configValidator(value)) throw invalidResponse();
  return value;
}

export function parseUploadManifest(value: unknown): UploadManifest {
  if (!manifestValidator(value)) throw invalidResponse();
  return value;
}

export function parseStagedUpload(value: unknown): StagedUpload {
  if (!stagedUploadValidator(value)) throw invalidResponse();
  return value;
}

export function parseAgentEvent(event: string, value: unknown): AgentStreamEvent {
  if (event === 'final_response') return { event, data: parseAgentResponse(value) };
  const validate = eventValidators.get(event);
  if (!validate || !validate(value)) throw invalidResponse();
  // The selected server schema validates this discriminated event at the boundary.
  return { event, data: value } as AgentStreamEvent;
}

export function parseAgentResponse(value: unknown): AgentResponse {
  if (!responseValidator(value)) throw invalidResponse();
  const response = value;
  const hasAnswer = response.status === 'completed' || response.status === 'partial';
  const hasProblem = response.status === 'failed' || response.status === 'partial' || response.status === 'refused';
  if (hasAnswer !== (response.response !== null)
    || hasProblem !== (response.problem !== null)
    || (response.answer_id !== null && response.response === null)
    || (response.status === 'needs_input' && !response.message.trim())
    || (response.status === 'refused' && response.problem?.code !== 'model_refusal')
    || (!['failed', 'refused'].includes(response.status) && response.upload_manifest === null)) {
    throw invalidResponse();
  }
  if (response.response !== null) validateSaveReceipts(response.response);
  return response;
}

export function validateSaveReceipts(response: AnswerResponse): void {
  for (const receipt of response.actions) {
    const artifact = receipt.artifact;
    const operation = receipt.operation;
    if (artifact !== null && (
      /[/\\:]/.test(artifact.filename) || !artifact.filename.endsWith('.txt')
      || artifact.expires_at <= artifact.created_at
    )) throw invalidResponse();
    if ((receipt.kind === 'save_text' && receipt.status === 'success') || receipt.verification === 'verified') {
      if (receipt.kind !== 'save_text' || receipt.status !== 'success' || receipt.verification !== 'verified'
        || !receipt.file_path || artifact === null || operation === null
        || artifact.sha256 !== operation.payload_sha256 || artifact.byte_count !== operation.byte_count) {
        throw invalidResponse();
      }
    }
  }
}

function invalidResponse(): ApiError {
  return new ApiError('invalid_response', '서버 응답의 형식을 확인하지 못했습니다.');
}
