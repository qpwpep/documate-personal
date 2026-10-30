export interface RawSseEvent {
  event: string;
  data: unknown;
}

/** Decode complete data frames; the client decides whether an event ends a turn. */
export async function* parseSseStream(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<RawSseEvent> {
  let event = 'message';
  let dataLines: string[] = [];

  for await (const line of readSseLines(stream)) {
    if (line === '') {
      if (dataLines.length > 0) {
        const data: unknown = JSON.parse(dataLines.join('\n'));
        if (data === null || typeof data !== 'object' || Array.isArray(data)) {
          throw new Error(`SSE event '${event}' data must be an object`);
        }
        yield { event, data };
      }
      event = 'message';
      dataLines = [];
      continue;
    }
    if (line.startsWith(':')) continue;

    const separator = line.indexOf(':');
    const field = separator === -1 ? line : line.slice(0, separator);
    const rawValue = separator === -1 ? '' : line.slice(separator + 1);
    const value = rawValue.startsWith(' ') ? rawValue.slice(1) : rawValue;
    if (field === 'event') event = value || 'message';
    if (field === 'data') dataLines.push(value);
  }

  if (dataLines.length > 0) {
    throw new Error(`Incomplete SSE event '${event}'`);
  }
}

async function* readSseLines(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = stream.getReader();
  const sourceFailure = reader.closed.then(
    () => null,
    (error: unknown) => ({ error }),
  );
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let line = '';
  let skipLf = false;
  let ended = false;

  try {
    while (!ended) {
      const chunk = await reader.read();
      ended = chunk.done;
      const text = ended ? decoder.decode() : decoder.decode(chunk.value, { stream: true });
      for (const character of text) {
        if (skipLf) {
          skipLf = false;
          if (character === '\n') continue;
        }
        if (character === '\r' || character === '\n') {
          yield line;
          line = '';
          skipLf = character === '\r';
        } else {
          line += character;
        }
      }
    }
    if (line !== '') yield line;
  } finally {
    try {
      if (!ended) {
        await reader.cancel().catch(async (error: unknown) => {
          // cancel() repeats an already-errored stream's failure. Cleanup must
          // not retract a received event, but a separate cancellation error is real.
          const failure = await sourceFailure;
          if (failure === null || failure.error !== error) throw error;
        });
      }
    } finally {
      reader.releaseLock();
    }
  }
}
