// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { CodeEditor } from '@nemo/common/src/components/CodeEditor';
import { ContentType } from '@nemo/common/src/components/CodeEditor/constants';
import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { Refresh } from '@nemo/icons/icons';
import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import {
  Flex,
  SidePanelContent,
  SidePanelDialog,
  SidePanelHeading,
  SidePanelRoot,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import {
  type AgentSpecSource,
  agentSpecSource,
  useRefreshAgentSpecFileset,
} from '@studio/api/agents/useAgentSpecFileset';
import { isSensitiveKey, MASK } from '@studio/routes/agents/AgentDetailRoute/configFormat';
import { shortRevision } from '@studio/routes/agents/AgentDetailRoute/helpers';
import { type FC, useMemo } from 'react';

interface AgentConfigPanelProps {
  open: boolean;
  workspace: string;
  agent?: Agent;
  specSource?: AgentSpecSource;
  onClose: () => void;
}

/**
 * Read-only JSON of the agent's stored config, with secrets masked. An agent whose spec fileset
 * tracks a ref can be moved to that ref's latest commit from here.
 */
export const AgentConfigPanel: FC<AgentConfigPanelProps> = ({
  open,
  workspace,
  agent,
  specSource,
  onClose,
}) => {
  const toast = useToast();
  const { mutate: refresh, isPending } = useRefreshAgentSpecFileset(workspace, agent?.name);
  const content = useMemo(
    () =>
      JSON.stringify(
        agent?.config ?? {},
        (key, value) => (key && isSensitiveKey(key) && value != null ? MASK : value),
        2
      ),
    [agent?.config]
  );

  const onRefresh = () =>
    refresh(undefined, {
      onSuccess: (updated) => {
        const revision = agentSpecSource(updated)?.revision ?? '';
        toast.success(
          revision === specSource?.revision
            ? 'Already on the latest commit'
            : `Updated to commit ${shortRevision(revision)}`
        );
      },
      onError: (failure) => toast.error(getErrorMessage(failure, 'Could not refresh')),
    });

  return (
    <SidePanelRoot open={open} onOpenChange={(next) => !next && onClose()} modal>
      <SidePanelDialog>
        <SidePanelContent className="w-[720px]" bordered>
          <SidePanelHeading>Agent config</SidePanelHeading>
          <Stack gap="density-md" padding="4" className="h-full min-h-0">
            {specSource?.trackedRevision ? (
              <Flex align="center" justify="between" gap="density-md">
                <Text kind="body/regular/sm" className="text-secondary">
                  Tracking {specSource.trackedRevision} at {shortRevision(specSource.revision)}.
                  Running deployments keep the commit they were created with.
                </Text>
                <LoadingButton
                  kind="secondary"
                  size="small"
                  height={32}
                  className="shrink-0"
                  onClick={onRefresh}
                  loading={isPending}
                >
                  <Refresh size="14" aria-hidden />
                  Update to latest
                </LoadingButton>
              </Flex>
            ) : null}
            <CodeEditor
              content={content}
              contentType={ContentType.JSON}
              readOnly
              className="h-full min-h-0"
            />
          </Stack>
        </SidePanelContent>
      </SidePanelDialog>
    </SidePanelRoot>
  );
};
