// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Fetch-based SSE reader: native EventSource can't send request headers or
// configure credentials, and we need auth + Last-Event-ID resume.

export interface SseEvent {
  data: string;
  id?: string;
}

export interface StreamSseOptions {
  signal: AbortSignal;
  headers?: HeadersInit;
  credentials?: RequestCredentials;
  onEvent: (event: SseEvent) => void;
  onError?: (error: unknown) => void;
  // Sent as Last-Event-ID on the *first* connect so the server resumes from a
  // known cursor (e.g. the end of an initial REST tail) instead of EOF.
  initialLastEventId?: string;
}

export const parseSseEvent = (raw: string): SseEvent => {
  let data = '';
  let id: string | undefined;
  for (const rawLine of raw.split('\n')) {
    const line = rawLine.replace(/\r$/, '');
    if (line === '' || line.startsWith(':')) continue; // blank / keepalive comment
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? '' : line.slice(colon + 1).replace(/^ /, ''); // strip one leading space
    if (field === 'data') data += data ? `\n${value}` : value;
    else if (field === 'id') id = value;
  }
  return { data, id };
};

const INITIAL_RETRY_MS = 1000;
const MAX_RETRY_MS = 30000;

// Non-retryable HTTP statuses — retrying these just hammers the endpoint
// (e.g. 401 after token expiry). 408/429 stay retryable.
const isFatalStatus = (status: number): boolean =>
  status >= 400 && status < 500 && status !== 408 && status !== 429;

const abortableDelay = (ms: number, signal: AbortSignal): Promise<void> =>
  new Promise((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException('Aborted', 'AbortError'));
    };
    const timer = setTimeout(() => {
      // Drop the abort listener so backoff cycles don't accumulate listeners.
      signal.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal.addEventListener('abort', onAbort, { once: true });
  });

export const streamSse = async (url: string, options: StreamSseOptions): Promise<void> => {
  const { signal, headers, credentials, onEvent, onError, initialLastEventId } = options;
  let lastEventId: string | undefined = initialLastEventId;
  let retryMs = INITIAL_RETRY_MS;

  while (!signal.aborted) {
    try {
      const requestHeaders = new Headers(headers);
      requestHeaders.set('Accept', 'text/event-stream');
      if (lastEventId) {
        requestHeaders.set('Last-Event-ID', lastEventId);
      }

      const response = await fetch(url, {
        signal,
        credentials,
        headers: requestHeaders,
      });
      if (!response.ok || !response.body) {
        const err = new Error(`SSE request failed: ${response.status}`);
        if (!response.ok && isFatalStatus(response.status)) {
          onError?.(err);
          return; // don't retry client errors (auth, not-found, bad request)
        }
        throw err;
      }
      retryMs = INITIAL_RETRY_MS; // reset backoff on successful connect

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        // Normalize CRLF so boundary indices match the sliced buffer.
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
        let boundary = buffer.indexOf('\n\n');
        while (boundary !== -1) {
          const rawEvent = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const event = parseSseEvent(rawEvent);
          if (event.id !== undefined) lastEventId = event.id;
          if (event.data) onEvent(event);
          boundary = buffer.indexOf('\n\n');
        }
      }
    } catch (error) {
      if (signal.aborted) return;
      onError?.(error);
    }
    if (signal.aborted) return;
    try {
      await abortableDelay(retryMs, signal);
    } catch {
      return; // aborted during backoff
    }
    retryMs = Math.min(retryMs * 2, MAX_RETRY_MS);
  }
};
