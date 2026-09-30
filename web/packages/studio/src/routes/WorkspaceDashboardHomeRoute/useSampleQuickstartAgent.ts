// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isNotFoundError } from '@nemo/common/src/api/common/utils';
import { JOB_POLLING_INTERVAL_LONG, JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { useAgentsListDeployments } from '@nemo/sdk/generated/agents/agent-deployments';
import { useAgentsGetAgent } from '@nemo/sdk/generated/agents/agents';
import type { QuickstartSampleAgent } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSamplePanel';
import { useMemo } from 'react';

/** Mirrors `_SAMPLE_WORKSPACE_NAME` in `nemo setup` (`nemo_helix_ext/cli/commands/setup.py`). */
export const SAMPLE_WORKSPACE = 'sample';
/** Mirrors `_SAMPLE_AGENT_NAME` in `nemo setup`, which deploys it into the sample workspace. */
export const SAMPLE_AGENT_NAME = 'email-security-triage';

/** Deployment statuses the controller is still moving through, so the panel polls faster. */
const TRANSITIONAL_STATUSES = new Set(['pending', 'starting', 'deleting']);

/** Not routable yet, but will be: worth naming in the chat command over the placeholder. */
const ROUTABLE_SOON_STATUSES = new Set(['pending', 'starting']);

/**
 * The sample agent `nemo setup` provisions, shaped for the QuickstartSamplePanel. Undefined while
 * loading, when disabled, or when the agent is missing, all of which hide the panel.
 */
export const useSampleQuickstartAgent = (
  workspace: string,
  enabled: boolean
): QuickstartSampleAgent | undefined => {
  const { data: agent } = useAgentsGetAgent(workspace, SAMPLE_AGENT_NAME, {
    query: {
      enabled,
      // A missing agent is an answer, not a transient failure. Keep looking while it is absent:
      // a re-run of `nemo setup` creates it, and the panel should appear without a reload.
      retry: (failureCount, error) => !isNotFoundError(error) && failureCount < 3,
      refetchInterval: (query) => (query.state.data ? false : JOB_POLLING_INTERVAL_LONG),
    },
  });

  const { data: deploymentsResponse, isFetched: isDeploymentsFetched } = useAgentsListDeployments(
    workspace,
    { page_size: 100 },
    {
      query: {
        enabled: enabled && !!agent,
        // The badge and the chat command both follow the deployment, which is often still
        // starting right after `nemo setup`; matches the agent page's cadence.
        refetchInterval: (query) =>
          (query.state.data?.data ?? []).some(
            (d) => d.agent === SAMPLE_AGENT_NAME && TRANSITIONAL_STATUSES.has(d.status ?? '')
          )
            ? JOB_POLLING_INTERVAL_MS
            : JOB_POLLING_INTERVAL_LONG,
      },
    }
  );

  return useMemo(() => {
    // Held back until the first deployments response, rather than flashing "Unknown" and a
    // placeholder command. `isFetched`, not `isPending`: a query that errored without data goes
    // back to pending on every poll, which would unmount the panel each time.
    if (!enabled || !agent || !isDeploymentsFetched) return undefined;

    const deployments = (deploymentsResponse?.data ?? []).filter(
      (d) => d.agent === SAMPLE_AGENT_NAME
    );
    // Only a deployment chat can reach, or soon will, goes in the command. A failed or deleting
    // one would copy a command that errors, so those leave the placeholder instead.
    const reachable =
      deployments.find((d) => d.status === 'running') ??
      deployments.find((d) => ROUTABLE_SOON_STATUSES.has(d.status ?? ''));
    const shown = reachable ?? deployments[0];

    return {
      name: agent.name ?? SAMPLE_AGENT_NAME,
      description: agent.description ?? '',
      status: shown?.status ?? 'unknown',
      statusLabel: deployments.length === 0 ? 'No deployments' : undefined,
      deploymentName: reachable?.name,
    };
  }, [enabled, agent, isDeploymentsFetched, deploymentsResponse]);
};
