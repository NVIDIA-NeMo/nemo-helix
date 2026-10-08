// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseSseEvent, type SseEvent, streamSse } from '@studio/util/sseStream';

describe('parseSseEvent', () => {
  it('parses data and id, stripping the single leading space', () => {
    expect(parseSseEvent('id: 11\ndata: {"m":2}')).toEqual({ data: '{"m":2}', id: '11' });
  });

  it('skips blank lines and keepalive comments', () => {
    expect(parseSseEvent(': keepalive 2026-06-03\n')).toEqual({ data: '' });
  });

  it('joins multiple data fields with newlines', () => {
    expect(parseSseEvent('data: line1\ndata: line2')).toEqual({ data: 'line1\nline2' });
  });

  it('tolerates CRLF line endings', () => {
    expect(parseSseEvent('id: 3\r\ndata: hi\r')).toEqual({ data: 'hi', id: '3' });
  });
});

const sseResponse = (body: string): Response => {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(body));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { 'content-type': 'text/event-stream' },
  });
};

describe('streamSse', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('emits one event per data block and tracks ids', async () => {
    const body = 'id: 6\ndata: {"m":1}\n\n: keepalive\n\nid: 11\ndata: {"m":2}\n\n';
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => sseResponse(body))
    );

    const controller = new AbortController();
    const received: SseEvent[] = [];
    await streamSse('https://example/stream', {
      signal: controller.signal,
      onEvent: (event) => {
        received.push(event);
        // Stop after the stream's two data events so we don't reconnect.
        if (received.length === 2) controller.abort();
      },
    });

    expect(received.map((e) => e.data)).toEqual(['{"m":1}', '{"m":2}']);
    expect(received.map((e) => e.id)).toEqual(['6', '11']);
  });

  it('sends Authorization and resumes with Last-Event-ID on reconnect', async () => {
    const fetchMock = vi
      .fn()
      // First connect: deliver one event, then the stream ends → triggers reconnect.
      .mockResolvedValueOnce(sseResponse('id: 6\ndata: {"m":1}\n\n'))
      // Second connect: deliver another event.
      .mockResolvedValueOnce(sseResponse('id: 11\ndata: {"m":2}\n\n'));
    vi.stubGlobal('fetch', fetchMock);

    const controller = new AbortController();
    const received: SseEvent[] = [];
    await streamSse('https://example/stream', {
      signal: controller.signal,
      headers: { Authorization: 'Bearer tok' },
      onEvent: (event) => {
        received.push(event);
        if (received.length === 2) controller.abort();
      },
    });

    expect(received.map((e) => e.data)).toEqual(['{"m":1}', '{"m":2}']);
    // First request carries the auth header and no resume cursor.
    const firstHeaders = new Headers((fetchMock.mock.calls[0][1] as RequestInit).headers);
    expect(firstHeaders.get('Authorization')).toBe('Bearer tok');
    expect(firstHeaders.has('Last-Event-ID')).toBe(false);
    // Reconnect resumes from the last delivered id.
    const secondHeaders = new Headers((fetchMock.mock.calls[1][1] as RequestInit).headers);
    expect(secondHeaders.get('Last-Event-ID')).toBe('6');
  });

  it('passes credentials through to fetch for cookie-backed streams', async () => {
    const fetchMock = vi.fn();
    fetchMock.mockResolvedValue(sseResponse('id: 1\ndata: {"m":1}\n\n'));
    vi.stubGlobal('fetch', fetchMock);

    const controller = new AbortController();
    await streamSse('https://example/stream', {
      signal: controller.signal,
      credentials: 'include',
      headers: new Headers({ 'X-Source': 'NeMo Studio' }),
      onEvent: () => controller.abort(),
    });

    const request = fetchMock.mock.calls[0][1] as RequestInit;
    const headers = new Headers(request.headers);
    expect(request.credentials).toBe('include');
    expect(headers.get('X-Source')).toBe('NeMo Studio');
    expect(headers.get('Accept')).toBe('text/event-stream');
  });

  it('sends initialLastEventId on the first connect to bridge a tail handoff', async () => {
    const fetchMock = vi.fn();
    fetchMock.mockResolvedValue(sseResponse('id: 42\ndata: {"m":1}\n\n'));
    vi.stubGlobal('fetch', fetchMock);

    const controller = new AbortController();
    await streamSse('https://example/stream', {
      signal: controller.signal,
      initialLastEventId: '40',
      onEvent: () => controller.abort(),
    });

    const firstHeaders = new Headers((fetchMock.mock.calls[0][1] as RequestInit).headers);
    expect(firstHeaders.get('Last-Event-ID')).toBe('40');
  });

  it('does not retry on a fatal 4xx status', async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 404 }));
    vi.stubGlobal('fetch', fetchMock);

    const controller = new AbortController();
    let errorCalls = 0;
    await streamSse('https://example/stream', {
      signal: controller.signal,
      onEvent: () => {},
      onError: () => {
        errorCalls += 1;
      },
    });

    // One attempt, one error report, no reconnect loop.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(errorCalls).toBe(1);
  });
});
