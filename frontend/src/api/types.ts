import type { AgentStreamEventData, components } from './generated';

export type ClientConfig = components['schemas']['ClientConfig'];
export type UploadManifest = components['schemas']['UploadManifest'];
export type StagedUpload = components['schemas']['StagedUpload'];
export type BrowserUploadSyncRequest = components['schemas']['BrowserUploadSyncRequest'];
export type AgentRequest = components['schemas']['AgentRequest'];
export type RecipientInput = components['schemas']['RecipientInput'];
export type SlackDefault = Required<components['schemas']['SlackDefault']>;
export type AnswerExportResult = Omit<components['schemas']['AnswerExportResult'], 'response'> & {
  response: AnswerResponse;
};
export type AnswerResponse = Omit<Required<components['schemas']['AnswerResponse']>, 'content'> & {
  content: Required<components['schemas']['AnswerDocument']>;
};
export type AgentResponse = Omit<Required<components['schemas']['AgentResponse']>, 'response'> & {
  response: AnswerResponse | null;
};

type EventData = Omit<AgentStreamEventData, 'final_response'> & {
  final_response: AgentResponse;
};

export type AgentStreamEvent = {
  [Name in keyof EventData]: { event: Name; data: EventData[Name] }
}[keyof EventData];
