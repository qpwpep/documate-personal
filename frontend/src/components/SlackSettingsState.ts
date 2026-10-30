import type { RecipientInput } from '../api/types';

export type SlackSettingsValue = {
  kind: 'omitted' | RecipientInput['kind'];
  drafts: Record<RecipientInput['kind'], string>;
};

export function createSlackSettings(): SlackSettingsValue {
  return { kind: 'omitted', drafts: { channel: '', user: '', email: '' } };
}

export function selectedSlackRecipient(value: SlackSettingsValue): RecipientInput | null {
  // Invalid or blank explicit input is still a choice, never the configured default.
  return value.kind === 'omitted' ? null : { kind: value.kind, value: value.drafts[value.kind] };
}
