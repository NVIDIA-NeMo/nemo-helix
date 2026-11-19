// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import {
  ROW_ACTIONS_COLUMN_SIZE,
  StudioDataView,
} from '@nemo/common/src/components/DataView/StudioDataView';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { QuickActionsMenuRoot } from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { TableEmptyState } from '@nemo/common/src/components/TableEmptyState';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  useEnablePluginApisPluginsInstalledNamePatch,
  useInstalledApisPluginsInstalledGet,
  useUninstallPluginApisPluginsInstalledNameDelete,
} from '@nemo/sdk/generated/platform/external-plugins';
import type { ExternalPlugin } from '@nemo/sdk/generated/platform/schema';
import { Badge, Button, Stack, Text } from '@nvidia/foundations-react-core';
import { invalidatePluginCaches } from '@studio/routes/CommunityPluginsRoute/pluginQueries';
import { useQueryClient } from '@tanstack/react-query';
import { type ComponentProps, type FC, useCallback, useMemo, useState } from 'react';

interface InstalledPluginsDataViewProps {
  onAdd: () => void;
  attributes?: { Stack?: ComponentProps<typeof Stack> };
}

export const InstalledPluginsDataView: FC<InstalledPluginsDataViewProps> = ({
  onAdd,
  attributes,
}) => {
  const toast = useToast();
  const queryClient = useQueryClient();
  const dataViewState = useStudioDataViewState();
  const [pluginToRemove, setPluginToRemove] = useState<ExternalPlugin>();

  const { data, error, isFetching } = useInstalledApisPluginsInstalledGet();
  const enablePlugin = useEnablePluginApisPluginsInstalledNamePatch();
  const removePlugin = useUninstallPluginApisPluginsInstalledNameDelete();

  // The registry has no paging or search: it returns every registration (at most 256).
  const plugins = useMemo(() => data?.plugins ?? [], [data?.plugins]);

  const setEnabled = useCallback(
    async (plugin: ExternalPlugin, enabled: boolean) => {
      try {
        await enablePlugin.mutateAsync({ name: plugin.name, data: { enabled } });
        await invalidatePluginCaches(queryClient);
      } catch (err) {
        toast.error(getErrorMessage(err as Error, `Failed to update ${plugin.title}`));
      }
    },
    [enablePlugin, queryClient, toast]
  );

  const handleRemove = async () => {
    if (!pluginToRemove) return false;
    try {
      await removePlugin.mutateAsync({ name: pluginToRemove.name });
      await invalidatePluginCaches(queryClient);
      return true;
    } catch {
      return false;
    }
  };

  const makeColumns: ComponentProps<typeof StudioDataView<ExternalPlugin>>['makeColumns'] =
    useCallback(
      ({ display }, { rowActionsColumn }) => [
        display({
          id: 'plugin',
          header: 'Plugin',
          size: 260,
          cell: ({ row }) => (
            <Stack gap="density-xs">
              <Text kind="label/bold/md">{row.original.title}</Text>
              <Text kind="body/regular/xs" className="text-subtle">
                {row.original.name}
              </Text>
            </Stack>
          ),
        }),
        display({
          id: 'status',
          header: 'Status',
          size: 110,
          cell: ({ row }) => (
            <Badge color={row.original.enabled === false ? 'gray' : 'green'} kind="solid">
              {row.original.enabled === false ? 'Disabled' : 'Enabled'}
            </Badge>
          ),
        }),
        display({
          id: 'endpoints',
          header: 'Endpoints',
          size: 100,
          cell: ({ row }) => <Text>{row.original.endpoints.length}</Text>,
        }),
        display({
          id: 'ui',
          header: 'Studio UI',
          size: 110,
          cell: ({ row }) => <Text>{row.original.bundle ? 'Yes' : '—'}</Text>,
        }),
        display({
          id: 'credential',
          header: 'Secret',
          size: 180,
          cell: ({ row }) => <Text>{row.original.credential?.secret_name ?? '—'}</Text>,
        }),
        rowActionsColumn({
          size: ROW_ACTIONS_COLUMN_SIZE,
          enableResizing: false,
          cell: ({ row: { original: plugin } }) => (
            <QuickActionsMenuRoot
              actions={[
                {
                  label: plugin.enabled === false ? 'Enable' : 'Disable',
                  onSelect: () => void setEnabled(plugin, plugin.enabled === false),
                },
              ]}
              deleteAction={{ label: 'Remove', onSelect: () => setPluginToRemove(plugin) }}
            />
          ),
        }),
      ],
      [setEnabled]
    );

  return (
    <Stack gap="density-2xl" {...attributes?.Stack}>
      <StudioDataView
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        attributes={{
          DataViewRoot: {
            data: plugins,
            totalCount: plugins.length,
            reactTableOptions: { getRowId: (plugin) => plugin.name },
            requestStatus: error ? 'error' : isFetching ? 'loading' : undefined,
          },
          DataViewTableContent: {
            renderEmptyState: () => (
              <TableEmptyState
                header="No community plugins"
                emptyMessage="Add a plugin from its manifest URL to proxy an external API and show its UI in Studio."
                actions={
                  <Button color="brand" onClick={onAdd}>
                    Add Plugin
                  </Button>
                }
              />
            ),
            renderErrorState: () => (
              <ErrorPanel
                errorMessage={getErrorMessage(error ?? new Error('Failed to load plugins'))}
              />
            ),
          },
        }}
      />
      {pluginToRemove && (
        <DeleteConfirmationModal
          open
          simpleConfirm
          title={`Remove: ${pluginToRemove.title}`}
          description="This removes the registration and its API routes. Workspace secrets are kept. Browsers that already loaded the plugin's UI keep running it until they reload."
          successText="Plugin removed"
          errorText="Failed to remove the plugin"
          onDelete={handleRemove}
          onClose={() => setPluginToRemove(undefined)}
        />
      )}
    </Stack>
  );
};
