// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage, isNotFoundError } from '@nemo/common/src/api/common/utils';
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
import { type FC, useEffect, useMemo, useRef } from 'react';
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
  const { hash, key } = useLocation();
  const panelRef = useRef<HTMLDivElement>(null);
  const scrolledForKey = useRef<string | null>(null);
  const {
    data: fileset,
    isLoading,
    error,
    isFetching,
    refetch,
  } = useAgentSpecFileset(workspace, agentName);
  const { mutate: refresh, isPending } = useRefreshAgentSpecFileset(workspace, agentName);

  const source = useMemo(() => agentSpecSource(fileset), [fileset]);
  // A 404 answers the question: the agent has no spec fileset. Any other failure leaves the
  // source unknown, which is not the same as absent, so it gets said rather than hidden.
  const unreadable = Boolean(error) && !isNotFoundError(error);
  const shown = Boolean(source) || unreadable;

  // The panel mounts once the fileset resolves, which is after the hash lands. Keyed on the
  // navigation rather than the hash so a repeat click scrolls again but a re-render does not.
  useEffect(() => {
    if (hash !== `#${SOURCE_PANEL_ID}` || !shown || scrolledForKey.current === key) return;
    scrolledForKey.current = key;
    panelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, [hash, key, shown]);

  if (isLoading || !shown) return null;

  const onRefresh = () =>
    refresh(undefined, {
      onSuccess: (updated) => {
        const revision = agentSpecSource(updated)?.revision ?? '';
        toast.success(
          revision === source?.revision
            ? 'Already on the latest commit'
            : `Updated to commit ${shortRevision(revision)}`
        );
      },
      onError: (failure) => toast.error(getErrorMessage(failure as Error) || 'Could not refresh'),
    });

  return (
    <div ref={panelRef} id={SOURCE_PANEL_ID} className="scroll-mt-6">
      <DetailPanel
        title="Source"
        slotAction={
          source?.trackedRevision ? (
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
        {!source ? (
          <Stack gap="2" align="start">
            <Text kind="body/regular/sm">
              Could not read where this agent&apos;s files come from.{' '}
              {error ? getErrorMessage(error) : null}
            </Text>
            <Button
              kind="secondary"
              size="small"
              type="button"
              onClick={() => void refetch()}
              disabled={isFetching}
            >
              <RefreshCw size={14} aria-hidden />
              {isFetching ? 'Retrying…' : 'Retry'}
            </Button>
          </Stack>
        ) : (
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
            {source.trackedRevision ? (
              <KVPair label="Tracking" value={source.trackedRevision} />
            ) : null}
            <KVPair label="Revision" value={shortRevision(source.revision)} truncate />
            <Text kind="body/regular/sm">
              Files are read from GitHub on demand at this commit. Deployments stage the commit they
              were created with, so updating here leaves running deployments where they are.
            </Text>
          </Stack>
        )}
      </DetailPanel>
    </div>
  );
};
