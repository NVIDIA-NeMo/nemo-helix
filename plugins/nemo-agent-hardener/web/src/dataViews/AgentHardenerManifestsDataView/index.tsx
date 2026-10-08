// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useRunWarGame } from '@agent-hardener/components/useRunWarGame';
import {
  getAgentHardenerListManifestsQueryKey,
  useAgentHardenerDeleteManifest,
  useAgentHardenerListManifests,
} from '@agent-hardener/generated/api';
import type { AgentHardenerManifest } from '@agent-hardener/generated/schema';
import { useNotify, useToast, useWorkspace } from '@agent-hardener/host';
import {
  getAgentHardenerManifestDetailRoute,
  getNewAgentHardenerManifestRoute,
} from '@agent-hardener/paths';
import { DeleteConfirmationModal, QuickActionsMenuRoot, RelativeTime, StudioDataView, TableEmptyState, getSortParam, useStudioDataViewState } from '@nemo/common';
import { Button, Text } from '@nvidia/foundations-react-core';
import { keepPreviousData, useQueryClient } from '@tanstack/react-query';
import { ComponentProps, FC, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router';

type AgentHardenerManifestWithId = AgentHardenerManifest & { id: string };

export const AgentHardenerManifestsDataView: FC = () => {
  const navigate = useNavigate();
  const workspace = useWorkspace();
  const toast = useToast();
  const notify = useNotify();
  const queryClient = useQueryClient();
  const dataViewState = useStudioDataViewState({ defaultSort: [{ id: 'created_at', desc: true }] });
  const [toDelete, setToDelete] = useState<AgentHardenerManifestWithId | null>(null);

  const { data: response, isLoading } = useAgentHardenerListManifests(
    workspace,
    {
      sort: getSortParam(dataViewState.sorting.state),
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
    },
    { query: { placeholderData: keepPreviousData, refetchOnMount: 'always', retry: false } }
  );

  const invalidate = () =>
    queryClient.invalidateQueries({ queryKey: getAgentHardenerListManifestsQueryKey(workspace) });

  const runWarGame = useRunWarGame(workspace);
  const deleteManifest = useAgentHardenerDeleteManifest();

  // A war-game replays the manifest's benign suite, so require one first: with none, send the user to the
  // manifest page to generate it rather than silently kicking off an inline interview mid-run.
  const startRun = (manifest: AgentHardenerManifestWithId) => {
    if (!manifest.name) return;
    if (!manifest.benign_suite?.length) {
      toast.error(
        'No benign suite for this manifest yet — generate it first, then run the war-game.'
      );
      navigate(getAgentHardenerManifestDetailRoute(workspace, manifest.name));
      return;
    }
    runWarGame.mutate({
      workspace,
      data: { spec: { manifest_id: manifest.name, driver: 'service' } },
    });
  };

  const manifests = useMemo<AgentHardenerManifestWithId[]>(() => {
    const rows = (response?.data ?? []) as AgentHardenerManifest[];
    return rows.map((m) => ({ ...m, id: m.id || `${m.workspace ?? ''}/${m.name ?? ''}` }));
  }, [response]);

  const total =
    (response?.pagination as { total_results?: number } | undefined)?.total_results ??
    manifests.length;

  const makeColumns: ComponentProps<
    typeof StudioDataView<AgentHardenerManifestWithId>
  >['makeColumns'] = ({ accessor }, { rowActionsColumn }) => [
    accessor('name', { header: 'Manifest', cell: ({ row }) => row.original.name ?? '-' }),
    accessor('agent', {
      header: 'Agent',
      cell: ({ row }) => (
        <Text className="truncate" style={{ maxWidth: 240 }} kind="body/regular/md">
          {row.original.agent || '-'}
        </Text>
      ),
    }),
    accessor('created_at', {
      id: 'created_at',
      header: 'Created',
      enableSorting: true,
      size: 160,
      cell: ({ row }) =>
        row.original.created_at ? <RelativeTime datetime={row.original.created_at} /> : null,
    }),
    rowActionsColumn({
      size: 70,
      cell: ({ row }) => (
        <QuickActionsMenuRoot
          actions={[
            { label: 'Run war-game', onSelect: () => startRun(row.original) },
            {
              label: 'Edit',
              onSelect: () =>
                row.original.name &&
                navigate(getAgentHardenerManifestDetailRoute(workspace, row.original.name)),
            },
            { label: 'Delete', onSelect: () => setToDelete(row.original) },
          ]}
        />
      ),
    }),
  ];

  return (
    <>
      <StudioDataView<AgentHardenerManifestWithId>
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        onRowClick={(row) =>
          row.name && navigate(getAgentHardenerManifestDetailRoute(workspace, row.name))
        }
        attributes={{
          DataViewRoot: {
            data: manifests,
            totalCount: total,
            requestStatus: isLoading && !response ? 'loading' : undefined,
          },
          DataViewTableContent: {
            renderEmptyState: () => (
              <TableEmptyState
                header="No manifests yet"
                emptyMessage="Create a manifest from a registered agent, then run the war-game against it. An agent that is not on the platform — bespoke orchestration, its own image — is war-gamed with the agent-hardener CLI instead."
                actions={
                  <Button asChild color="brand">
                    <Link to={getNewAgentHardenerManifestRoute(workspace)}>New Manifest</Link>
                  </Button>
                }
              />
            ),
          },
        }}
      />
      <DeleteConfirmationModal
        onNotify={notify}
        open={!!toDelete}
        onClose={() => setToDelete(null)}
        title={`Delete ${toDelete?.name ?? 'manifest'}?`}
        description="This permanently deletes the manifest and its cached benign suite."
        successText="Manifest deleted."
        errorText="Failed to delete the manifest."
        onDelete={async () => {
          if (!toDelete?.name) return false;
          await deleteManifest.mutateAsync({ workspace, name: toDelete.name });
          invalidate();
          return true;
        }}
      />
    </>
  );
};
