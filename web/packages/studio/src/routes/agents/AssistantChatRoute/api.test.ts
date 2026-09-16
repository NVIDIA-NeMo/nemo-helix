// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BASE_URL } from '@studio/constants/environment';
import {
  AssistantSessionNotFoundError,
  createAssistantSession,
  deleteAssistantSessionHistory,
  getAssistantSessionHistory,
  listAssistantHistorySessions,
  listAssistantSkills,
  resolveAssistantInput,
  resolveAssistantPermission,
  streamAssistantMessage,
} from '@studio/routes/agents/AssistantChatRoute/api';

const getExpectedStudioBaseUrl = (): string => {
  const normalizedBaseUrl = BASE_URL.replace(/\/+$/, '');
  const basePath = normalizedBaseUrl && normalizedBaseUrl !== '/' ? normalizedBaseUrl : '';
  return `${window.location.origin}${basePath}`;
};

describe('Assistant API helpers', () => {
  afterEach(() => {
    localStorage.clear();
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it('adds the active OIDC bearer token to Assistant requests', async () => {
    const authority = 'https://auth.example.test';
    const clientId = 'studio-client';
    vi.stubEnv('VITE_AUTH_AUTHORITY', authority);
    vi.stubEnv('VITE_AUTH_CLIENT_ID', clientId);
    localStorage.setItem(
      `oidc.user:${authority}:${clientId}`,
      JSON.stringify({
        access_token: 'assistant-token',
        expires_at: Math.floor(Date.now() / 1000) + 3600,
        profile: { sub: 'user-1' },
        token_type: 'Bearer',
      })
    );
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify([]), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await listAssistantSkills();

    const requestInit = fetchMock.mock.calls[0]?.[1] as RequestInit | undefined;
    expect(new Headers(requestInit?.headers).get('Authorization')).toBe('Bearer assistant-token');
  });

  it('scopes session creation and history requests to the active workspace', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ session_id: 'session-1' }), { status: 200 })
      )
      .mockResolvedValueOnce(new Response(JSON.stringify([]), { status: 200 }))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ session_id: 'session-1', items: [], chat_artifacts: {} }), {
          status: 200,
        })
      )
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);

    await createAssistantSession('team-a');
    await listAssistantHistorySessions('team-a');
    await getAssistantSessionHistory('session-1', 'team-a');
    await deleteAssistantSessionHistory('session-1', 'team-a');

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      expect.stringContaining('/sessions?workspace=team-a'),
      expect.stringContaining('/history/sessions?workspace=team-a'),
      expect.stringContaining('/history/sessions/session-1?workspace=team-a'),
      expect.stringContaining('/history/sessions/session-1?workspace=team-a'),
    ]);
    expect(fetchMock.mock.calls[3]?.[1]).toEqual({ method: 'DELETE' });
  });

  it('identifies a missing session history response', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify({ detail: 'no such session history' }), { status: 404 })
        )
    );

    await expect(getAssistantSessionHistory('missing-session')).rejects.toBeInstanceOf(
      AssistantSessionNotFoundError
    );
  });

  it('reads model-generated titles from history sessions', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify([
            {
              session_id: 'session-1',
              mtime: 123,
              title: 'Create Spam Detector Agent',
              first_prompt: 'I want to create an agent that does spam detection.',
              message_count: 2,
              token_count: 10,
              tool_call_count: 0,
              tool_calls: [],
              chat_artifacts: {},
            },
          ]),
          { status: 200 }
        )
      )
    );

    const sessions = await listAssistantHistorySessions();

    expect(sessions[0]?.title).toBe('Create Spam Detector Agent');
  });

  it('posts messages with the active workspace', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response('event: done\ndata: \n\n', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await streamAssistantMessage({
      sessionId: 'session-1',
      message: 'list agents',
      workspace: 'default',
      signal: new AbortController().signal,
      handlers: {
        onAssistantEvent: vi.fn(),
        onInputRequest: vi.fn(),
        onPermissionRequest: vi.fn(),
        onDone: vi.fn(),
        onError: vi.fn(),
      },
    });

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining('/sessions/session-1/messages'),
      expect.objectContaining({
        body: JSON.stringify({
          message: 'list agents',
          studio_base_url: getExpectedStudioBaseUrl(),
          studio_pathname: window.location.pathname,
          workspace: 'default',
        }),
        method: 'POST',
      })
    );
  });

  it('keeps historical chat artifact selections when loading a session', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          session_id: 'session-1',
          items: [],
          chat_artifacts: {
            selections: [
              { label: 'Agent', value: 'cat-identifier' },
              { label: 'Model', value: '`cloud, nvidia/example` - good for this agent' },
            ],
            files: [],
            links: [{ label: 'Agents', destination: 'agents', href: '/workspaces/default/agents' }],
            jobs: [
              {
                name: 'agent-eval-1',
                job_type: 'agent_evaluation',
                source: 'evaluator',
                href: '/workspaces/default/agents/evaluations/agent-eval-1',
              },
            ],
            tools: [],
          },
        }),
        { status: 200 }
      )
    );
    vi.stubGlobal('fetch', fetchMock);

    const history = await getAssistantSessionHistory('session-1');

    expect(history.chat_artifacts.selections).toEqual([
      { label: 'Agent', value: 'cat-identifier' },
      { label: 'Model', value: 'cloud, nvidia/example' },
    ]);
    expect(history.chat_artifacts.links).toEqual([
      { label: 'Agents', destination: 'agents', href: '/workspaces/default/agents' },
    ]);
    expect(history.chat_artifacts.jobs).toEqual([
      {
        name: 'agent-eval-1',
        job_type: 'agent_evaluation',
        source: 'evaluator',
        href: '/workspaces/default/agents/evaluations/agent-eval-1',
      },
    ]);
  });

  it('parses a stored thinking part when loading a session', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          session_id: 'session-1',
          items: [
            {
              kind: 'assistant',
              parts: [
                { type: 'thinking', thinking: 'The user wants a capital city.' },
                { type: 'thinking', thinking: '' },
                { type: 'text', text: 'Paris.' },
              ],
            },
          ],
          chat_artifacts: {},
        }),
        { status: 200 }
      )
    );
    vi.stubGlobal('fetch', fetchMock);

    const history = await getAssistantSessionHistory('session-1');

    expect(history.items).toEqual([
      {
        kind: 'assistant',
        parts: [
          { type: 'thinking', thinking: 'The user wants a capital city.' },
          { type: 'text', text: 'Paris.' },
        ],
      },
    ]);
  });

  it('preserves legacy copilot model artifacts when loading a session', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          session_id: 'session-1',
          items: [],
          chat_artifacts: {
            model: 'nvidia/legacy-model',
            model_source: 'copilot',
            copilot_model: 'nvidia/legacy-model',
          },
        }),
        { status: 200 }
      )
    );
    vi.stubGlobal('fetch', fetchMock);

    const history = await getAssistantSessionHistory('session-1');

    expect(history.chat_artifacts.assistant_model).toBe('nvidia/legacy-model');
    expect(history.chat_artifacts.model).toBeUndefined();
    expect(history.chat_artifacts.model_source).toBeUndefined();
  });

  it('emits structured permission requests from SSE events', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          [
            'event: permission_request',
            'data: {"request_id":"request-1","tool_name":"Bash","input":{"command":"ls"},"tool_use_id":"tool-1"}',
            '',
            'event: done',
            'data: ',
            '',
          ].join('\n'),
          { status: 200 }
        )
      );
    vi.stubGlobal('fetch', fetchMock);

    const onPermissionRequest = vi.fn();

    await streamAssistantMessage({
      sessionId: 'session-1',
      message: 'list files',
      signal: new AbortController().signal,
      handlers: {
        onAssistantEvent: vi.fn(),
        onInputRequest: vi.fn(),
        onPermissionRequest,
        onDone: vi.fn(),
        onError: vi.fn(),
      },
    });

    expect(onPermissionRequest).toHaveBeenCalledWith({
      requestId: 'request-1',
      toolName: 'Bash',
      input: { command: 'ls' },
      toolUseId: 'tool-1',
    });
  });

  it('fails closed when permission requests omit required fields', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          [
            'event: permission_request',
            'data: {"request_id":"request-1","input":{"command":"ls"}}',
            '',
          ].join('\n'),
          { status: 200 }
        )
      );
    vi.stubGlobal('fetch', fetchMock);

    const onPermissionRequest = vi.fn();
    const onError = vi.fn();

    await streamAssistantMessage({
      sessionId: 'session-1',
      message: 'list files',
      signal: new AbortController().signal,
      handlers: {
        onAssistantEvent: vi.fn(),
        onInputRequest: vi.fn(),
        onPermissionRequest,
        onDone: vi.fn(),
        onError,
      },
    });

    expect(onPermissionRequest).not.toHaveBeenCalled();
    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'NeMo Assistant permission request was malformed' })
    );
  });

  it('fails closed when permission request input is an array', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          [
            'event: permission_request',
            'data: {"request_id":"request-1","tool_name":"Bash","input":[]}',
            '',
          ].join('\n'),
          { status: 200 }
        )
      );
    vi.stubGlobal('fetch', fetchMock);

    const onPermissionRequest = vi.fn();
    const onError = vi.fn();

    await streamAssistantMessage({
      sessionId: 'session-1',
      message: 'list files',
      signal: new AbortController().signal,
      handlers: {
        onAssistantEvent: vi.fn(),
        onInputRequest: vi.fn(),
        onPermissionRequest,
        onDone: vi.fn(),
        onError,
      },
    });

    expect(onPermissionRequest).not.toHaveBeenCalled();
    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'NeMo Assistant permission request was malformed' })
    );
  });

  it('calls onPermissionExpired when a permission_expired event arrives', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          [
            'event: permission_expired',
            'data: {"request_id":"request-1"}',
            '',
            'event: done',
            'data: ',
            '',
          ].join('\n'),
          { status: 200 }
        )
      );
    vi.stubGlobal('fetch', fetchMock);

    const onPermissionExpired = vi.fn();

    await streamAssistantMessage({
      sessionId: 'session-1',
      message: 'list files',
      signal: new AbortController().signal,
      handlers: {
        onAssistantEvent: vi.fn(),
        onInputRequest: vi.fn(),
        onPermissionRequest: vi.fn(),
        onPermissionExpired,
        onDone: vi.fn(),
        onError: vi.fn(),
      },
    });

    expect(onPermissionExpired).toHaveBeenCalledWith('request-1');
  });

  it('calls onInputExpired when an input_expired event arrives', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          [
            'event: input_expired',
            'data: {"request_id":"request-2"}',
            '',
            'event: done',
            'data: ',
            '',
          ].join('\n'),
          { status: 200 }
        )
      );
    vi.stubGlobal('fetch', fetchMock);

    const onInputExpired = vi.fn();

    await streamAssistantMessage({
      sessionId: 'session-1',
      message: 'list files',
      signal: new AbortController().signal,
      handlers: {
        onAssistantEvent: vi.fn(),
        onInputRequest: vi.fn(),
        onPermissionRequest: vi.fn(),
        onInputExpired,
        onDone: vi.fn(),
        onError: vi.fn(),
      },
    });

    expect(onInputExpired).toHaveBeenCalledWith('request-2');
  });

  it('posts approval decisions using the backend permission shape', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await resolveAssistantPermission({
      sessionId: 'session-1',
      requestId: 'request-1',
      decision: {
        approved: true,
        updatedInput: { command: 'ls' },
      },
    });

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining('/sessions/session-1/permissions/request-1'),
      expect.objectContaining({
        body: JSON.stringify({
          approved: true,
          updated_input: { command: 'ls' },
        }),
        method: 'POST',
      })
    );
  });

  it('loads Assistant skill metadata', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify([
          {
            name: 'inference',
            claude_name: 'nemo-inference',
            description: 'Use NeMo Platform inference.',
            source: 'nemo-platform',
            source_path: 'packages/nemo_platform_ext/src/nemo_platform_ext/skills/inference',
            install_path: '.claude/skills/nemo-inference/SKILL.md',
            installed: false,
          },
        ]),
        { status: 200 }
      )
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(listAssistantSkills()).resolves.toEqual([
      {
        name: 'inference',
        claude_name: 'nemo-inference',
        description: 'Use NeMo Platform inference.',
        source: 'nemo-platform',
        source_path: 'packages/nemo_platform_ext/src/nemo_platform_ext/skills/inference',
        install_path: '.claude/skills/nemo-inference/SKILL.md',
        installed: false,
      },
    ]);
  });

  it('emits structured blocking input requests from SSE events', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          [
            'event: input_request',
            'data: {"request_id":"request-1","kind":"agent","input":{"title":"Pick agent"}}',
            '',
          ].join('\n'),
          { status: 200 }
        )
      );
    vi.stubGlobal('fetch', fetchMock);

    const onInputRequest = vi.fn();

    await streamAssistantMessage({
      sessionId: 'session-1',
      message: 'pick agent',
      signal: new AbortController().signal,
      handlers: {
        onAssistantEvent: vi.fn(),
        onInputRequest,
        onPermissionRequest: vi.fn(),
        onDone: vi.fn(),
        onError: vi.fn(),
      },
    });

    expect(onInputRequest).toHaveBeenCalledWith({
      requestId: 'request-1',
      kind: 'agent',
      input: { title: 'Pick agent' },
    });
  });

  it('emits structured dataset file input requests from SSE events', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          [
            'event: input_request',
            'data: {"request_id":"request-1","kind":"dataset_file","input":{"title":"Pick dataset"}}',
            '',
          ].join('\n'),
          { status: 200 }
        )
      );
    vi.stubGlobal('fetch', fetchMock);

    const onInputRequest = vi.fn();

    await streamAssistantMessage({
      sessionId: 'session-1',
      message: 'pick dataset',
      signal: new AbortController().signal,
      handlers: {
        onAssistantEvent: vi.fn(),
        onInputRequest,
        onPermissionRequest: vi.fn(),
        onDone: vi.fn(),
        onError: vi.fn(),
      },
    });

    expect(onInputRequest).toHaveBeenCalledWith({
      requestId: 'request-1',
      kind: 'dataset_file',
      input: { title: 'Pick dataset' },
    });
  });

  it('emits structured model input requests from SSE events', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          [
            'event: input_request',
            'data: {"request_id":"request-1","kind":"model","input":{"title":"Pick model"}}',
            '',
          ].join('\n'),
          { status: 200 }
        )
      );
    vi.stubGlobal('fetch', fetchMock);

    const onInputRequest = vi.fn();

    await streamAssistantMessage({
      sessionId: 'session-1',
      message: 'pick model',
      signal: new AbortController().signal,
      handlers: {
        onAssistantEvent: vi.fn(),
        onInputRequest,
        onPermissionRequest: vi.fn(),
        onDone: vi.fn(),
        onError: vi.fn(),
      },
    });

    expect(onInputRequest).toHaveBeenCalledWith({
      requestId: 'request-1',
      kind: 'model',
      input: { title: 'Pick model' },
    });
  });

  it('posts blocking input decisions using the backend input shape', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await resolveAssistantInput({
      sessionId: 'session-1',
      requestId: 'request-1',
      decision: {
        value: { agent: 'react-agent' },
      },
    });

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining('/sessions/session-1/inputs/request-1'),
      expect.objectContaining({
        body: JSON.stringify({
          skipped: undefined,
          value: { agent: 'react-agent' },
        }),
        method: 'POST',
      })
    );
  });

  it('preserves tool use ids from session history', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          session_id: 'session-1',
          items: [
            {
              kind: 'assistant',
              parts: [
                {
                  type: 'tool_use',
                  id: 'toolu_job',
                  name: 'job_progress',
                  input: { job_name: 'studio-job-1' },
                },
              ],
            },
          ],
        }),
        { status: 200 }
      )
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(getAssistantSessionHistory('session-1')).resolves.toEqual({
      session_id: 'session-1',
      chat_artifacts: {
        agent: undefined,
        assistant_model: undefined,
        files: [],
        links: [],
        jobs: [{ name: 'studio-job-1' }],
        model: undefined,
        model_source: undefined,
        selections: [],
        tools: ['job_progress'],
        workspace: undefined,
      },
      items: [
        {
          kind: 'assistant',
          parts: [
            {
              type: 'tool_use',
              id: 'toolu_job',
              name: 'job_progress',
              input: { job_name: 'studio-job-1' },
            },
          ],
        },
      ],
    });
  });
});
