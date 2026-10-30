import Ajv2020 from 'ajv/dist/2020';
import { ApiError } from './api/errors';
import { parseAnswerExport } from './api/delivery-validation';
import type { AgentResponse, AnswerExportResult, UploadManifest } from './api/types';
import { parseAgentResponse, parseUploadManifest } from './api/validation';
import { createAttachmentState, type AttachmentState } from './attachments/state';
import { createSlackSettings, type SlackSettingsValue } from './components/SlackSettingsState';

export const STORAGE_KEY = 'documate.react.conversations.v1';

export interface ChatTurn {
  id: string;
  query: string;
  result: AgentResponse | null;
  notices: string[];
  error: string | null;
  requestId: string | null;
  interrupted: boolean;
  receiving: boolean;
  exportResult: AnswerExportResult | null;
  exportError: string | null;
}

export interface Conversation {
  id: string;
  draft: string;
  turns: ChatTurn[];
  attachments: AttachmentState;
  slack: SlackSettingsValue;
  confirmedEpoch: string | null;
  manifest: UploadManifest | null;
  connection: 'unknown' | 'checking' | 'ready' | 'expired';
  connectionError: string | null;
  busy: 'chat' | 'upload' | 'save' | null;
}

export interface ConversationState {
  activeId: string;
  conversations: Conversation[];
  theme: 'system' | 'light' | 'dark';
}

interface StoredTurn extends Omit<ChatTurn, 'result' | 'exportResult'> {
  result: object | null;
  exportResult: object | null;
}

interface StoredConversation {
  id: string;
  draft: string;
  turns: StoredTurn[];
  attachments: { pendingPrompt: string | null; restoreFilesRequired: boolean };
  slack: SlackSettingsValue;
  confirmedEpoch: string | null;
  manifest: object | null;
  expired: boolean;
}

interface StoredSnapshot {
  version: 1;
  activeId: string;
  conversations: StoredConversation[];
  theme: ConversationState['theme'];
}

const text = { type: 'string' };
const nullableText = { type: ['string', 'null'] };
const nullableObject = { type: ['object', 'null'] };
const identity = { type: 'string', pattern: '^[A-Za-z0-9_-]{1,128}$' };
const validateSnapshot = new Ajv2020().compile<StoredSnapshot>({
  type: 'object', additionalProperties: false,
  required: ['version', 'activeId', 'conversations', 'theme'],
  properties: {
    version: { const: 1 }, activeId: identity,
    theme: { enum: ['system', 'light', 'dark'] },
    conversations: {
      type: 'array', minItems: 1,
      items: {
        type: 'object', additionalProperties: false,
        required: ['id', 'draft', 'turns', 'attachments', 'slack', 'confirmedEpoch', 'manifest', 'expired'],
        properties: {
          id: identity, draft: text, confirmedEpoch: nullableText, manifest: nullableObject,
          expired: { type: 'boolean' },
          attachments: {
            type: 'object', additionalProperties: false,
            required: ['pendingPrompt', 'restoreFilesRequired'],
            properties: { pendingPrompt: nullableText, restoreFilesRequired: { type: 'boolean' } },
          },
          slack: {
            type: 'object', additionalProperties: false, required: ['kind', 'drafts'],
            properties: {
              kind: { enum: ['omitted', 'channel', 'user', 'email'] },
              drafts: {
                type: 'object', additionalProperties: false, required: ['channel', 'user', 'email'],
                properties: { channel: text, user: text, email: text },
              },
            },
          },
          turns: {
            type: 'array',
            items: {
              type: 'object', additionalProperties: false,
              required: ['id', 'query', 'result', 'notices', 'error', 'requestId', 'interrupted', 'receiving', 'exportResult', 'exportError'],
              properties: {
                id: identity, query: text, result: nullableObject,
                notices: { type: 'array', items: text }, error: nullableText,
                requestId: nullableText, interrupted: { type: 'boolean' }, receiving: { type: 'boolean' },
                exportResult: nullableObject, exportError: nullableText,
              },
            },
          },
        },
      },
    },
  },
});

export function createConversation(): Conversation {
  return {
    id: crypto.randomUUID(), draft: '', turns: [], attachments: createAttachmentState(),
    slack: createSlackSettings(), confirmedEpoch: null, manifest: null,
    connection: 'unknown', connectionError: null, busy: null,
  };
}

export function createConversationState(): ConversationState {
  const conversation = createConversation();
  return { activeId: conversation.id, conversations: [conversation], theme: 'system' };
}

export function conversationTitle(conversation: Conversation): string {
  const title = conversation.turns[0]?.query.trim().replace(/\s+/g, ' ');
  return title ? Array.from(title).slice(0, 36).join('') : '새 대화';
}

export function serializeConversations(state: ConversationState): string {
  const snapshot: StoredSnapshot = {
    version: 1, activeId: state.activeId, theme: state.theme,
    conversations: state.conversations.map((conversation) => ({
      id: conversation.id, draft: conversation.draft, turns: conversation.turns,
      attachments: {
        pendingPrompt: conversation.attachments.pendingPrompt,
        restoreFilesRequired: conversation.attachments.restoreFilesRequired
          || conversation.attachments.files.length > 0 || conversation.busy === 'upload',
      },
      slack: conversation.slack, confirmedEpoch: conversation.confirmedEpoch,
      manifest: conversation.manifest, expired: conversation.connection === 'expired',
    })),
  };
  return JSON.stringify(snapshot);
}

export function restoreConversations(snapshot: string): ConversationState {
  let value: unknown;
  try {
    value = JSON.parse(snapshot);
  } catch (error) {
    if (error instanceof SyntaxError) throw invalidSnapshot();
    throw error;
  }
  if (!validateSnapshot(value)) throw invalidSnapshot();
  const ids = new Set(value.conversations.map((conversation) => conversation.id));
  if (ids.size !== value.conversations.length || !ids.has(value.activeId)) throw invalidSnapshot();
  try {
    return {
      activeId: value.activeId, theme: value.theme,
      conversations: value.conversations.map((conversation) => {
        if (new Set(conversation.turns.map((turn) => turn.id)).size !== conversation.turns.length) throw invalidSnapshot();
        return {
          id: conversation.id, draft: conversation.draft,
          turns: conversation.turns.map(restoreTurn),
          attachments: { ...createAttachmentState(), ...conversation.attachments },
          slack: conversation.slack,
          confirmedEpoch: conversation.confirmedEpoch,
          // This is a display snapshot. The HTTP client must GET a fresh manifest.
          manifest: conversation.manifest === null ? null : parseUploadManifest(conversation.manifest),
          connection: conversation.expired ? 'expired' : 'unknown',
          connectionError: null, busy: null,
        };
      }),
    };
  } catch (error) {
    if (error instanceof ApiError) throw invalidSnapshot();
    throw error;
  }
}

function restoreTurn(turn: StoredTurn): ChatTurn {
  const result = turn.result === null ? null : parseAgentResponse(turn.result);
  let exportResult: AnswerExportResult | null = null;
  if (turn.exportResult !== null) {
    if (!result?.answer_id) throw invalidSnapshot();
    exportResult = parseAnswerExport(turn.exportResult, result.answer_id);
  }
  const interrupted = turn.interrupted || (turn.receiving && result === null);
  return {
    ...turn,
    result,
    exportResult,
    receiving: false,
    interrupted,
    error: interrupted && turn.error === null
      ? '응답 수신이 중단되었습니다. 서버 작업은 계속될 수 있습니다. 질문을 자동으로 다시 보내지 않습니다.'
      : turn.error,
  };
}

function invalidSnapshot(): Error {
  return new Error('저장된 대화 기록을 읽지 못했습니다. 새 대화를 시작할 수 있습니다.');
}
