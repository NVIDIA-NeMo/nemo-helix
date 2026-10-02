/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import {
  getGuardrailsGetGuardrailConfigQueryKey,
  useGuardrailsDeleteConfig,
} from '@nemo/sdk/generated/platform/guardrails';
import type { GuardrailConfig } from '@nemo/sdk/generated/platform/schema';
import { Button, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { GuardrailsDataView } from '@studio/components/dataViews/GuardrailsDataView';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { CreateGuardrailModal } from '@studio/routes/guardrails/CreateGuardrailModal';
import { getGuardrailDetailRoute, getGuardrailsRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useCallback, useState } from 'react';

export const GuardrailsRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const queryClient = useQueryClient();
  const openRow = useRowNavigation();

  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [configToDuplicate, setConfigToDuplicate] = useState<GuardrailConfig | null>(null);
  const [configToDelete, setConfigToDelete] = useState<GuardrailConfig | null>(null);
  const [configsToDelete, setConfigsToDelete] = useState<GuardrailConfig[]>([]);

  const { mutateAsync: deleteConfig } = useGuardrailsDeleteConfig();

  useBreadcrumbs({
    items: [{ href: getGuardrailsRoute(workspace), slotLabel: 'Guardrails' }],
  });

  // Errors propagate deliberately — see the matching handler in GuardrailDetailActions. The
  // service's 409 for a config still applied by a VirtualModel names the routes blocking the
  // delete, and ConfirmationModal toasts a thrown error's message.
  const handleDelete = useCallback(async (): Promise<boolean> => {
    if (!configToDelete?.name) return false;
    await deleteConfig({ workspace, name: configToDelete.name });
    // Invalidate by URL prefix — matches all pages/sorts for this workspace
    await queryClient.invalidateQueries({
      queryKey: [`/apis/guardrails/v2/workspaces/${workspace}/configs`],
    });
    return true;
  }, [configToDelete, deleteConfig, queryClient, workspace]);

  const handleBulkDelete = useCallback(async (): Promise<boolean> => {
    if (!configsToDelete.length) return false;
    if (configsToDelete.some((c) => !c.name)) return false;

    const results = await Promise.allSettled(
      configsToDelete.map((c) => deleteConfig({ workspace, name: c.name! }))
    );

    if (results.some((r) => r.status === 'fulfilled')) {
      await queryClient.invalidateQueries({
        queryKey: [`/apis/guardrails/v2/workspaces/${workspace}/configs`],
      });
    }

    return results.every((r) => r.status === 'fulfilled');
  }, [configsToDelete, deleteConfig, queryClient, workspace]);

  return (
    <AccessibleTitle title="Guardrails">
      <Stack className="h-full overflow-auto" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Guardrail Configs"
          slotDescription="Manage NeMo Guardrails configurations for your workspace."
          slotActions={
            <Button color="brand" onClick={() => setIsCreateOpen(true)}>
              Create Guardrail
            </Button>
          }
        />
        <GuardrailsDataView
          workspace={workspace}
          onRowClick={(config, event) => {
            if (!config.name) return;
            queryClient.setQueryData(
              getGuardrailsGetGuardrailConfigQueryKey(workspace, config.name),
              config
            );
            openRow(event, getGuardrailDetailRoute(workspace, config.name));
          }}
          onRequestDuplicate={setConfigToDuplicate}
          onRequestDelete={setConfigToDelete}
          onCreate={() => setIsCreateOpen(true)}
          onRequestBulkDelete={setConfigsToDelete}
        />
      </Stack>

      <CreateGuardrailModal open={isCreateOpen} onClose={() => setIsCreateOpen(false)} />

      {configToDuplicate ? (
        <CreateGuardrailModal
          open
          sourceConfig={configToDuplicate}
          onClose={() => setConfigToDuplicate(null)}
        />
      ) : null}

      {configToDelete ? (
        <DeleteConfirmationModal
          open
          simpleConfirm
          title={`Delete guardrail config: ${configToDelete.name}`}
          successText="Guardrail config deleted successfully."
          onDelete={handleDelete}
          onClose={() => setConfigToDelete(null)}
        />
      ) : null}

      {configsToDelete.length > 0 ? (
        <DeleteConfirmationModal
          open
          simpleConfirm
          title={`Delete ${configsToDelete.length} guardrail config${configsToDelete.length === 1 ? '' : 's'}?`}
          description={`This will permanently delete ${configsToDelete.length} guardrail config${configsToDelete.length === 1 ? '' : 's'}. This action cannot be undone.`}
          successText={`${configsToDelete.length} guardrail config${configsToDelete.length === 1 ? '' : 's'} deleted successfully.`}
          errorText="Failed to delete some guardrail configs. Please try again."
          onDelete={handleBulkDelete}
          onClose={() => setConfigsToDelete([])}
        />
      ) : null}
    </AccessibleTitle>
  );
};
