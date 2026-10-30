// @vitest-environment node

import { describe, expect, it } from 'vitest';
import { parseSseStream } from './sse';

const encoder = new TextEncoder();

function byteStream(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
}

async function receive(stream: ReadableStream<Uint8Array>) {
  const events = [];
  for await (const event of parseSseStream(stream)) events.push(event);
  return events;
}

describe('parseSseStream', () => {
  it('preserves Korean and emoji across one-byte HTTP chunks', async () => {
    const bytes = encoder.encode('event: final_response\ndata: {"message":"한글 답변 🧪"}\n\n');
    const chunks = Array.from(bytes, (byte) => Uint8Array.of(byte));

    await expect(receive(byteStream(chunks))).resolves.toEqual([
      { event: 'final_response', data: { message: '한글 답변 🧪' } },
    ]);
  });

  it('handles split CRLF, bare CR, comments, BOM and multiline data', async () => {
    const stream = byteStream([
      encoder.encode('\ufeff: keepalive\r'),
      encoder.encode('\nevent: progress_snapshot\r\ndata: {"stage":"planner",\r'),
      encoder.encode('\ndata: "summary":"계획 중"}\r\n\r'),
      encoder.encode('\nevent:\rdata: {}\r\rid: ignored\n\ndata: {"message":"next"}\n\n'),
    ]);

    await expect(receive(stream)).resolves.toEqual([
      { event: 'progress_snapshot', data: { stage: 'planner', summary: '계획 중' } },
      { event: 'message', data: {} },
      { event: 'message', data: { message: 'next' } },
    ]);
  });

  it('returns no events for empty or comment-only streams', async () => {
    await expect(receive(byteStream([]))).resolves.toEqual([]);
    await expect(receive(byteStream([encoder.encode(': keepalive\n\nevent: done\n\n')]))).resolves.toEqual([]);
  });

  it.each([
    ['malformed JSON', 'event: final_response\ndata: {\n\n'],
    ['array payload', 'data: []\n\n'],
    ['null payload', 'data: null\n\n'],
    ['primitive payload', 'data: "text"\n\n'],
    ['missing frame delimiter', 'data: {}\n'],
    ['missing line delimiter', 'data: {}'],
  ])('rejects %s instead of emitting an event', async (_name, text) => {
    await expect(receive(byteStream([encoder.encode(text)]))).rejects.toThrow();
  });

  it('rejects an incomplete UTF-8 sequence at the end of the stream', async () => {
    const bytes = encoder.encode('data: {"message":"🧪');
    await expect(receive(byteStream([bytes.slice(0, -1)]))).rejects.toThrow();
  });

  it('cancels the reader and releases its lock when the consumer stops', async () => {
    let cancelled = false;
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('event: final_response\ndata: {}\n\n'));
      },
      cancel() {
        cancelled = true;
      },
    });

    for await (const event of parseSseStream(stream)) {
      expect(event.event).toBe('final_response');
      break;
    }

    expect(cancelled).toBe(true);
    expect(stream.locked).toBe(false);
  });

  it('releases the lock after a stream transport failure', async () => {
    const failure = new Error('connection interrupted');
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.error(failure);
      },
    });

    await expect(receive(stream)).rejects.toBe(failure);
    expect(stream.locked).toBe(false);
  });

  it.each([
    new TypeError('connection closed after final_response'),
    new DOMException('Request aborted', 'AbortError'),
  ])('keeps a received final event when the source subsequently errors: %s', async (failure) => {
    let source!: ReadableStreamDefaultController<Uint8Array>;
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        source = controller;
        controller.enqueue(encoder.encode('event: final_response\ndata: {}\n\n'));
      },
    });
    const events = parseSseStream(stream);

    await expect(events.next()).resolves.toMatchObject({ value: { event: 'final_response', data: {} } });
    source.error(failure);

    await expect(events.return(undefined)).resolves.toEqual({ done: true, value: undefined });
    expect(stream.locked).toBe(false);
  });

  it('still propagates a source failure before the final event', async () => {
    const failure = new TypeError('connection closed before final_response');
    let source!: ReadableStreamDefaultController<Uint8Array>;
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        source = controller;
        controller.enqueue(encoder.encode('event: heartbeat\ndata: {}\n\n'));
      },
    });
    const events = parseSseStream(stream);

    await expect(events.next()).resolves.toMatchObject({ value: { event: 'heartbeat' } });
    source.error(failure);

    await expect(events.next()).rejects.toBe(failure);
    expect(stream.locked).toBe(false);
  });

  it('propagates a separate failure in the source cancellation algorithm', async () => {
    const failure = new DOMException('Cancellation failed', 'AbortError');
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('event: final_response\ndata: {}\n\n'));
      },
      cancel() {
        throw failure;
      },
    });
    const events = parseSseStream(stream);

    await expect(events.next()).resolves.toMatchObject({ value: { event: 'final_response' } });
    await expect(events.return(undefined)).rejects.toBe(failure);
    expect(stream.locked).toBe(false);
  });
});
