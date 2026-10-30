import Ajv2020, { type ValidateFunction } from 'ajv/dist/2020';

import type { components } from './generated';
import openapi from './openapi.json';
import { ApiError } from './errors';
import type { AgentResponse, AgentStreamEvent, AnswerResponse, ClientConfig, UploadManifest } from './types';

// OpenAPI annotations are not JSON Schema assertions. Keep every schema
// constraint, but accept the document's OpenAPI-specific annotation keywords.
const validator = new Ajv2020({ strictSchema: false, strictTypes: false });
validator.addSchema(openapi, 'documate');

const configValidator = validator.compile<ClientConfig>({ $ref: 'documate#/components/schemas/ClientConfig' });
const manifestValidator = validator.compile<UploadManifest>({ $ref: 'documate#/components/schemas/UploadManifest' });
const responseValidator = validator.compile<components['schemas']['AgentResponse']>({
  $ref: 'documate#/components/schemas/AgentResponse',
});
const eventSchemas = openapi.paths['/agent/stream'].post.responses['200'].content['text/event-stream']['x-sse-events'];
const eventValidators = new Map<string, ValidateFunction>(
  Object.entries(eventSchemas).map(([event, schema]) => [event, validator.compile({
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

export function parseAgentEvent(event: string, value: unknown): AgentStreamEvent {
  const validate = eventValidators.get(event);
  if (!validate || !validate(value)) throw invalidResponse();
  if (event === 'final_response') return { event, data: parseAgentResponse(value) };
  // The selected server schema validates this discriminated event at the boundary.
  return { event, data: value } as AgentStreamEvent;
}

function parseAgentResponse(value: unknown): AgentResponse {
  if (!responseValidator(value)) throw invalidResponse();
  const response: AgentResponse = {
    ...value,
    status: value.status ?? 'completed',
    response: value.response == null ? null : normalizeAnswer(value.response),
    answer_id: value.answer_id ?? null,
    message: value.message ?? '',
    problem: value.problem ?? null,
    request_id: value.request_id ?? '',
    missing_slots: value.missing_slots ?? [],
    debug: value.debug ?? null,
    upload_manifest: value.upload_manifest ?? null,
  };
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
  return response;
}

function normalizeAnswer(value: components['schemas']['AnswerResponse']): AnswerResponse {
  return {
    ...value,
    content: { blocks: value.content?.blocks ?? [] },
    citations: value.citations ?? [],
    checks: value.checks ?? [],
    issues: value.issues ?? [],
    actions: value.actions ?? [],
    content_hash: value.content_hash ?? '',
    retrieval_required: value.retrieval_required ?? false,
    semantic_validation: value.semantic_validation ?? null,
  };
}

function invalidResponse(): ApiError {
  return new ApiError('invalid_response', '서버 응답의 형식을 확인하지 못했습니다.');
}
