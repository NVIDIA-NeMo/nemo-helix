// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LogViewer } from '@nemo/common/src/components/LogViewer';
import { logger } from '@nemo/common/src/utils/logger';
import {
  getAgentsStreamDeploymentLogsQueryKey,
  useAgentsGetDeploymentLogs,
} from '@nemo/sdk/generated/agents/agent-deployments';
import type { AgentDeployment, AgentDeploymentStatus } from '@nemo/sdk/generated/agents/schema';
import type { HelixJobLog } from '@nemo/sdk/generated/platform/schema';
import { withHelixBrowserAuthRequest } from '@nemo/sdk/src/utils/platformRequest';
import { Block, Stack, Text } from '@nvidia/foundations-react-core';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { useOidcBearerToken } from '@studio/providers/auth/useOidcBearerToken';
import { streamSse } from '@studio/util/sseStream';
import { type FC, useEffect, useMemo, useState } from 'react';

interface DeploymentLogsViewProps {
  workspace: string;
  /** The header's selection, any status. */
  deployment?: AgentDeployment;
}

export const DeploymentLogsView: FC<DeploymentLogsViewProps> = ({ workspace, deployment }) =>
  deployment?.name ? (
    <Block className="h-full min-h-0 w-full" paddingX="2">
      <LogsForDeployment
        workspace={workspace}
        deploymentName={deployment.name}
        status={deployment.status}
      />
    </Block>
  ) : (
    <Stack gap="density-md" padding="density-md">
      <Text kind="body/regular/sm" className="text-subtle">
        No deployments yet for this agent. Deploy first to capture logs.
      </Text>
    </Stack>
  );

interface LogsForDeploymentProps {
  workspace: string;
  deploymentName: string;
  status?: AgentDeploymentStatus;
}

const TAIL_LINES = 500;
const MAX_STREAMED_LINES = 5000;

const LogsForDeployment: FC<LogsForDeploymentProps> = ({ workspace, deploymentName, status }) => {
  const { data, isLoading } = useAgentsGetDeploymentLogs(
    workspace,
    deploymentName,
    { tail: TAIL_LINES },
    { query: { staleTime: 5000 } }
  );

  const [streamedLines, setStreamedLines] = useState<HelixJobLog[]>([]);
  // streamSse gives up on non-retryable responses (e.g. 401/403); the log is no longer live then.
  const [streamStopped, setStreamStopped] = useState(false);

  const accessToken = useOidcBearerToken();
  const tailOffset = data?.next_offset;

  // Also keyed on tailOffset: a refetch re-baselines `data` to the latest tail and
  // reopens the stream from the new offset, so lines buffered against the old offset
  // would otherwise be rendered a second time after the ones `data` already carries.
  useEffect(() => {
    setStreamedLines([]);
  }, [deploymentName, tailOffset]);

  useEffect(() => {
    if (!deploymentName || isLoading) return;
    const url = `${PLATFORM_BASE_URL}${getAgentsStreamDeploymentLogsQueryKey(workspace, deploymentName)[0]}`;
    const controller = new AbortController();
    setStreamStopped(false);
    void (async () => {
      try {
        const request = await withHelixBrowserAuthRequest({
          headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : undefined,
        });
        if (controller.signal.aborted) return;

        await streamSse(url, {
          signal: controller.signal,
          credentials: request.credentials,
          headers: request.headers,
          initialLastEventId: tailOffset != null ? String(tailOffset) : undefined,
          onEvent: (event) => {
            try {
              const parsed = JSON.parse(event.data) as HelixJobLog;
              setStreamedLines((prev) => {
                const next = [...prev, parsed];
                return next.length > MAX_STREAMED_LINES
                  ? next.slice(next.length - MAX_STREAMED_LINES)
                  : next;
              });
            } catch {
              // ignore malformed lines
            }
          },
          onError: (err) => {
            logger.warn(`Log stream interrupted for deployment ${deploymentName}; retrying`, err);
          },
        });
        if (!controller.signal.aborted) setStreamStopped(true);
      } catch (error) {
        if (!controller.signal.aborted) {
          logger.warn(`Unable to start log stream for deployment ${deploymentName}`, error);
          setStreamStopped(true);
        }
      }
    })();
    return () => controller.abort();
  }, [workspace, deploymentName, accessToken, isLoading, tailOffset]);

  const logs = useMemo<HelixJobLog[]>(() => {
    const initial = (data?.data ?? []).map(
      (line): HelixJobLog => ({
        timestamp: line.timestamp,
        message: line.message,
        job: '',
        job_step: '',
        job_task: '',
      })
    );
    return [...initial, ...streamedLines];
  }, [data, streamedLines]);

  return (
    <LogViewer
      logs={logs}
      isLoading={isLoading && logs.length === 0}
      // The server keeps the SSE open for failed/deleting deployments, so the status decides.
      logStatus={
        streamStopped ? undefined : status === 'failed' || status === 'deleting' ? 'ended' : 'live'
      }
      downloadFilename={`${deploymentName}.log`}
      rows={40}
      emptyMessage="No log output yet — the deployment may not have started writing."
    />
  );
};
