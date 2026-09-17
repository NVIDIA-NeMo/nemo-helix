// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { agentsCloseSession, agentsCreateSession } from '@nemo/sdk/generated/agents/agent-sessions';
import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import type { AgentSession } from '@nemo/sdk/generated/agents/schema/AgentSession';
import { useAgentChatSession } from '@studio/api/agents/useAgentChatSession';
import { act, renderHook, waitFor } from '@testing-library/react';

vi.mock('@nemo/sdk/generated/agents/agent-sessions', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/agents/agent-sessions')>()),
  agentsCreateSession: vi.fn(),
  agentsCloseSession: vi.fn(),
}));

vi.mock('@nemo/common/src/utils/logger', () => ({ handleGenericError: vi.fn() }));

const entityFields = {
  created_at: '2026-01-01T00:00:00Z',
  created_by: 'mschwab',
  updated_at: '2026-01-01T00:00:00Z',
  updated_by: 'mschwab',
  entity_id: 'entity-1',
  parent: '',
  db_version: 1,
};

const deployment = (configFormat: string): AgentDeployment => ({
  ...entityFields,
  auth_context: null,
  id: 'deployment-1',
  name: 'calc-1',
  workspace: 'ws',
  config: { config_format: configFormat },
});

const session = (id: string): AgentSession => ({
  ...entityFields,
  id,
  entity_id: id,
  name: `session-${id}`,
  workspace: 'ws',
  deployment_id: 'deployment-1',
});

beforeEach(() => {
  vi.mocked(agentsCreateSession).mockResolvedValue(session('abc'));
  vi.mocked(agentsCloseSession).mockResolvedValue(session('abc'));
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('useAgentChatSession', () => {
  it('creates a session for a Fabric deployment and sends its id as a header', async () => {
    const { result } = renderHook(() =>
      useAgentChatSession('ws', deployment('nemo-agents-spec-v1'))
    );

    await waitFor(() =>
      expect(result.current.extraHeaders).toEqual({ 'X-Nemo-Session-Id': 'abc' })
    );
    expect(agentsCreateSession).toHaveBeenCalledWith('ws', { deployment_id: 'deployment-1' });
    expect(result.current.isPending).toBe(false);
  });

  it('sends no header for a NAT deployment', async () => {
    const { result } = renderHook(() => useAgentChatSession('ws', deployment('nat-workflow-v1')));

    await waitFor(() => expect(result.current.isPending).toBe(false));
    expect(agentsCreateSession).not.toHaveBeenCalled();
    expect(result.current.extraHeaders).toBeUndefined();
  });

  it('closes the session when the deployment changes', async () => {
    const { rerender, result } = renderHook(
      ({ target }: { target: AgentDeployment }) => useAgentChatSession('ws', target),
      { initialProps: { target: deployment('nemo-agents-spec-v1') } }
    );
    await waitFor(() => expect(result.current.extraHeaders).toBeDefined());

    vi.mocked(agentsCreateSession).mockResolvedValue(session('def'));
    rerender({
      target: { ...deployment('nemo-agents-spec-v1'), id: 'deployment-2' },
    });

    await waitFor(() => expect(agentsCloseSession).toHaveBeenCalledWith('ws', 'session-abc'));
    await waitFor(() =>
      expect(result.current.extraHeaders).toEqual({ 'X-Nemo-Session-Id': 'def' })
    );
  });

  it('recovers from a rejection naming the current session', async () => {
    const { result } = renderHook(() =>
      useAgentChatSession('ws', deployment('nemo-agents-spec-v1'))
    );
    await waitFor(() => expect(result.current.extraHeaders).toBeDefined());

    vi.mocked(agentsCreateSession).mockResolvedValue(session('def'));
    let recovered = false;
    act(() => {
      recovered = result.current.recoverFromError(
        new Error("Session ID 'abc' has expired and cannot be invoked.")
      );
    });

    expect(recovered).toBe(true);
    await waitFor(() =>
      expect(result.current.extraHeaders).toEqual({ 'X-Nemo-Session-Id': 'def' })
    );
  });

  it('leaves an unrelated failure to the caller', async () => {
    const { result } = renderHook(() =>
      useAgentChatSession('ws', deployment('nemo-agents-spec-v1'))
    );
    await waitFor(() => expect(result.current.extraHeaders).toBeDefined());

    let recovered = true;
    act(() => {
      recovered = result.current.recoverFromError(new Error('Agent returned 500'));
    });

    expect(recovered).toBe(false);
    expect(agentsCreateSession).toHaveBeenCalledTimes(1);
  });

  it('keeps the chat usable when session creation fails', async () => {
    vi.mocked(agentsCreateSession).mockRejectedValue(new Error('boom'));

    const { result } = renderHook(() =>
      useAgentChatSession('ws', deployment('nemo-agents-spec-v1'))
    );

    await waitFor(() => expect(result.current.isPending).toBe(false));
    expect(result.current.extraHeaders).toBeUndefined();
  });
});
