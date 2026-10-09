// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SampleAgentResponse } from '@nemo/sdk/generated/agents/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import {
  parseSampleAgentFrame,
  streamCreateSampleAgent,
} from '@studio/routes/WorkspaceDashboardHomeRoute/CreateSampleAgentModal/streamCreateSampleAgent';
import { http, HttpResponse } from 'msw';

const authMocks = vi.hoisted(() => ({
  platformFetch: vi.fn(),
}));

vi.mock('@nemo/sdk/src/utils/platformRequest', () => ({
  platformFetch: authMocks.platformFetch,
}));

const URL = `${PLATFORM_BASE_URL}/apis/agents/v2/sample-agent`;

const RESULT: SampleAgentResponse = {
  status: 'created',
  workspace: 'sample-1a2b3c4d',
  studio_url: '/studio/workspaces/sample-1a2b3c4d/dashboard',
  agent: 'email-security-triage',
  deployment: 'email-security-triage',
  deployment_status: 'pending',
};

const ndjson = (...frames: unknown[]) =>
  frames.map((frame) => (typeof frame === 'string' ? frame : JSON.stringify(frame))).join('\n');

const respondWith = (body: string, status = 201) =>
  server.use(
    http.post(
      URL,
      () => new HttpResponse(body, { status, headers: { 'Content-Type': 'application/x-ndjson' } })
    )
  );

const run = () =>
  streamCreateSampleAgent({ model: 'my-ws/my-model' }, 'token', new AbortController().signal);

beforeEach(() => {
  authMocks.platformFetch.mockReset();
  authMocks.platformFetch.mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
    const headers = new Headers(init?.headers);
    headers.set('X-Source', 'NeMo Studio');
    return fetch(input, { ...init, headers });
  });
});

describe('parseSampleAgentFrame', () => {
  it('reads known frame kinds', () => {
    expect(parseSampleAgentFrame('{"kind":"progress","component":"agent"}')).toEqual({
      kind: 'progress',
      component: 'agent',
    });
  });

  it('ignores blank lines, malformed JSON, and unknown frame kinds', () => {
    expect(parseSampleAgentFrame('  ')).toBeUndefined();
    expect(parseSampleAgentFrame('{not json')).toBeUndefined();
    expect(parseSampleAgentFrame('[1,2]')).toBeUndefined();
    expect(parseSampleAgentFrame('{"kind":"heartbeat"}')).toBeUndefined();
  });
});

describe('streamCreateSampleAgent', () => {
  it('posts the model with auth headers, skips progress and garbage, and resolves with the done result', async () => {
    let request: Request | undefined;
    server.use(
      http.post(URL, ({ request: req }) => {
        request = req.clone();
        return new HttpResponse(
          ndjson(
            { kind: 'progress', component: 'workspace', status: 'created' },
            'garbage',
            { kind: 'progress', component: 'agent', status: 'created' },
            { kind: 'done', result: RESULT }
          ),
          { status: 201, headers: { 'Content-Type': 'application/x-ndjson' } }
        );
      })
    );

    await expect(run()).resolves.toEqual(RESULT);

    expect(await request?.json()).toEqual({ model: 'my-ws/my-model' });
    expect(request?.headers.get('Authorization')).toBe('Bearer token');
    expect(request?.headers.get('X-Source')).toBe('NeMo Studio');
  });

  it('uses server-session credentials without stale bearer auth', async () => {
    let request: Request | undefined;
    authMocks.platformFetch.mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const headers = new Headers(init?.headers);
      headers.delete('Authorization');
      headers.set('X-Source', 'NeMo Studio');
      return fetch(input, { ...init, credentials: 'include', headers });
    });
    server.use(
      http.post(URL, ({ request: req }) => {
        request = req.clone();
        return new HttpResponse(ndjson({ kind: 'done', result: RESULT }), {
          status: 201,
          headers: { 'Content-Type': 'application/x-ndjson' },
        });
      })
    );

    await expect(run()).resolves.toEqual(RESULT);

    expect(authMocks.platformFetch).toHaveBeenCalled();
    expect(request?.credentials).toBe('include');
    expect(request?.headers.has('Authorization')).toBe(false);
    expect(request?.headers.get('X-Source')).toBe('NeMo Studio');
  });

  it('rejects with the message of an in-stream error frame, even on a 2xx response', async () => {
    respondWith(
      ndjson(
        { kind: 'progress', component: 'workspace', status: 'existing' },
        { kind: 'error', message: 'Already deployed with another model', status_code: 409 }
      ),
      200
    );

    await expect(run()).rejects.toThrow('Already deployed with another model');
  });

  it('rejects with the FastAPI detail when the request fails before streaming', async () => {
    server.use(
      http.post(URL, () =>
        HttpResponse.json({ detail: "Model 'my-ws/my-model' not found" }, { status: 404 })
      )
    );

    await expect(run()).rejects.toThrow("Model 'my-ws/my-model' not found");
  });

  it('joins pydantic validation messages', async () => {
    server.use(
      http.post(URL, () =>
        HttpResponse.json({ detail: [{ msg: 'Bad ref.' }, { msg: 'Try again.' }] }, { status: 422 })
      )
    );

    await expect(run()).rejects.toThrow('Bad ref. Try again.');
  });

  it('rejects when the stream ends without a done frame', async () => {
    respondWith(ndjson({ kind: 'progress', component: 'workspace', status: 'created' }));

    await expect(run()).rejects.toThrow('The connection closed before setup finished.');
  });

  it.each([
    ['an HTML proxy page', 502, '<html><body><h1>502 Bad Gateway</h1></body></html>', 'text/html'],
    ['a plain-text body', 500, 'Internal Server Error', 'text/plain'],
    ['an empty body', 503, '', 'text/plain'],
  ])('never shows %s, only the status', async (_case, status, body, contentType) => {
    server.use(
      http.post(
        URL,
        () => new HttpResponse(body, { status, headers: { 'Content-Type': contentType } })
      )
    );

    await expect(run()).rejects.toThrow(
      `The platform couldn't handle the request (status ${status}). Try again in a moment.`
    );
  });

  it('names the status of a 4xx without a FastAPI detail', async () => {
    server.use(http.post(URL, () => new HttpResponse('nope', { status: 403 })));

    await expect(run()).rejects.toThrow('The request was rejected (status 403).');
  });

  it('replaces the browser wording when the request never gets a response', async () => {
    server.use(http.post(URL, () => HttpResponse.error()));

    await expect(run()).rejects.toThrow("Couldn't reach the platform.");
  });

  it('lets an abort through unchanged', async () => {
    const controller = new AbortController();
    controller.abort();

    await expect(
      streamCreateSampleAgent({ model: 'my-ws/my-model' }, 'token', controller.signal)
    ).rejects.toThrow(expect.objectContaining({ name: 'AbortError' }));
  });
});
