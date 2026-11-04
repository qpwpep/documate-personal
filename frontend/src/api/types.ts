import type { AgentStreamEventData, components } from './generated';

export type ClientConfig = components['schemas']['ClientConfig'];
export type UploadManifest = components['schemas']['UploadManifest'];
export type StagedUpload = components['schemas']['StagedUpload'];
export type BrowserUploadSyncRequest = components['schemas']['BrowserUploadSyncRequest'];
export type AgentRequest = components['schemas']['AgentRequest'];
export type RecipientInput = components['schemas']['RecipientInput'];
export type SlackDefault = components['schemas']['SlackDefault'];
export type AnswerExportResult = components['schemas']['AnswerExportResult'];
export type AnswerResponse = components['schemas']['SourcedAnswerPayload'];
export type AgentResponse = components['schemas']['AgentResponse'];

export type AgentStreamEvent = {
  [Name in keyof AgentStreamEventData]: { event: Name; data: AgentStreamEventData[Name] }
}[keyof AgentStreamEventData];
