import { expectTypeOf, it } from 'vitest';

import type { AgentStreamEventData } from './generated';
import type { AgentResponse, AgentStreamEvent } from './types';

it('keeps server event requirements and stage names in the generated types', () => {
  expectTypeOf<AgentStreamEventData['request_started']>().toExtend<{
    execution_id: string;
    session_id: string;
  }>();
  expectTypeOf<AgentStreamEventData['stage_started']['stage']>().not.toEqualTypeOf<string>();
  expectTypeOf<AgentStreamEventData['stage_started']['attempt']>().toEqualTypeOf<number>();
  expectTypeOf<Extract<AgentStreamEvent, { event: 'final_response' }>['data']>().toEqualTypeOf<AgentResponse>();
});
