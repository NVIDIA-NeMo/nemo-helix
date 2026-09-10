// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { KVPair } from '@nemo/common/src/components/KVPair';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { Anchor, Button, Stack, Text } from '@nvidia/foundations-react-core';
import {
  agentSpecSource,
  useAgentSpecFileset,
  useRefreshAgentSpecFileset,
} from '@studio/api/agents/useAgentSpecFileset';
import { shortRevision } from '@studio/routes/agents/AgentDetailRoute/helpers';
import { DetailPanel } from '@studio/routes/agents/AgentDetailRoute/overview/DetailPanel';
import { RefreshCw } from 'lucide-react';
import { type FC, useEffect, useRef } from 'react';
import { useLocation } from 'react-router';

/** Anchor for the header's commit badge, which deep-links here. */
export const SOURCE_PANEL_ID = 'agent-source';

interface SourcePanelProps {
  workspace: string;
  agentName?: string;
}

/**
 * Where a repository-backed agent's files come from, and which commit it is pinned to.
 *
 * Renders nothing for an agent whose fileset is not repository-backed: an uploaded
 * directory has no revision to show and nothing to re-resolve.
 */
export const SourcePanel: FC<SourcePanelProps> = ({ workspace, agentName }) => {
  const toast = useToast();
  const { hash } = useLocation();
  const panelRef = useRef<HTMLDivElement>(null);
  const { data: fileset, isLoading } = useAgentSpecFileset(workspace, agentName);
  const { mutate: refresh, isPending } = useRefreshAgentSpecFileset(workspace, agentName);

  const source = agentSpecSource(fileset);

  // The panel mounts once the fileset resolves, which is after the hash lands.
  useEffect(() => {
    if (hash === `#${SOURCE_PANEL_ID}` && source) {
      panelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [hash, source]);

  if (isLoading || !source) return null;

  const onRefresh = () =>
    refresh(undefined, {
      onSuccess: (updated) => {
        const revision = agentSpecSource(updated)?.revision ?? '';
        toast.success(
          revision === source.revision
            ? `Already on the latest commit of ${source.trackedRevision}`
            : `Updated to ${shortRevision(revision)}`
        );
      },
      onError: (error) => toast.error(getErrorMessage(error as Error) || 'Could not refresh'),
    });

  return (
    <div ref={panelRef} id={SOURCE_PANEL_ID} className="scroll-mt-6">
      <DetailPanel
        title="Source"
        slotAction={
          source.trackedRevision ? (
            <Button
              kind="primary"
              color="brand"
              size="small"
              type="button"
              onClick={onRefresh}
              disabled={isPending}
            >
              <RefreshCw size={14} aria-hidden />
              {isPending ? 'Updating…' : 'Update to latest'}
            </Button>
          ) : undefined
        }
      >
        <Stack gap="2">
          <KVPair
            label="Repository"
            value={
              <Anchor
                href={source.webUrl}
                target="_blank"
                rel="noreferrer noopener"
                textKind="body/semibold/md"
                underline
                className="text-brand"
              >
                {source.repository}
              </Anchor>
            }
          />
          {source.trackedRevision && <KVPair label="Tracking" value={source.trackedRevision} />}
          <KVPair label="Revision" value={shortRevision(source.revision)} truncate />
          <Text kind="body/regular/sm">
            Files are read from GitHub on demand at this commit. Deployments stage the commit they
            were created with, so updating here leaves running deployments where they are.
          </Text>
        </Stack>
      </DetailPanel>
    </div>
  );
};
