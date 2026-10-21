// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { handleGenericError } from '@nemo/common/src/utils/logger';
import { agentsCloseSession, agentsCreateSession } from '@nemo/sdk/generated/agents/agent-sessions';
import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import type { AgentSession } from '@nemo/sdk/generated/agents/schema/AgentSession';
import { useCallback, useEffect, useMemo, useState } from 'react';

/** Read by the agents gateway, which forwards it to the deployment's Fabric server. */
const SESSION_ID_HEADER = 'X-Nemo-Session-Id';

/** Only Fabric-backed deployments serve logical sessions; NAT workflows have no notion of one. */
const FABRIC_CONFIG_FORMAT = 'nemo-agents-spec-v1';

export interface AgentChatSession {
  extraHeaders?: Record<string, string>;
  isPending: boolean;
  resetSession: () => void;
  /** Recreates the session when `error` is the gateway rejecting it; reports whether it did. */
  recoverFromError: (error: Error) => boolean;
}

const isFabricDeployment = (deployment?: AgentDeployment): boolean =>
  deployment?.config?.config_format === FABRIC_CONFIG_FORMAT;

const closeQuietly = async (workspace: string, session: AgentSession): Promise<void> => {
  if (!session.name) return;
  try {
    await agentsCloseSession(workspace, session.name);
  } catch {
    // The idle sweep reclaims it.
  }
};

/** Created eagerly: a message sent before a session exists is the case this prevents. */
export function useAgentChatSession(
  workspace: string,
  deployment?: AgentDeployment
): AgentChatSession {
  const deploymentId = isFabricDeployment(deployment) ? deployment?.id : undefined;
  const [generation, setGeneration] = useState(0);
  const [sessionId, setSessionId] = useState<string>();
  const [isPending, setIsPending] = useState(false);

  useEffect(() => {
    setSessionId(undefined);
    if (!deploymentId) {
      setIsPending(false);
      return;
    }

    let cancelled = false;
    let created: AgentSession | undefined;
    setIsPending(true);

    void (async () => {
      try {
        const session = await agentsCreateSession(workspace, { deployment_id: deploymentId });
        created = session;
        if (cancelled) {
          void closeQuietly(workspace, session);
          return;
        }
        setSessionId(session.id);
      } catch (error) {
        // Chat still works unsessioned.
        handleGenericError(error instanceof Error ? error : new Error(String(error)));
      } finally {
        if (!cancelled) setIsPending(false);
      }
    })();

    return () => {
      cancelled = true;
      if (created) void closeQuietly(workspace, created);
    };
  }, [workspace, deploymentId, generation]);

  const extraHeaders = useMemo(
    () => (sessionId ? { [SESSION_ID_HEADER]: sessionId } : undefined),
    [sessionId]
  );

  const resetSession = useCallback(() => setGeneration((value) => value + 1), []);

  // Matching our own id keeps an unrelated agent failure from discarding a healthy session.
  const recoverFromError = useCallback(
    (error: Error) => {
      if (!sessionId || !error.message.includes(`Session ID '${sessionId}'`)) return false;
      resetSession();
      return true;
    },
    [resetSession, sessionId]
  );

  return { extraHeaders, isPending, resetSession, recoverFromError };
}
