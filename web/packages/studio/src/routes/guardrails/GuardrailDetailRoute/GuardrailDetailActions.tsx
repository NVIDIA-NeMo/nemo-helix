// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import {
  type QuickActionItem,
  QuickActionsMenuRoot,
} from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { useGuardrailsDeleteConfig } from '@nemo/sdk/generated/platform/guardrails';
import type { GuardrailConfig } from '@nemo/sdk/generated/platform/schema';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { CreateGuardrailModal } from '@studio/routes/guardrails/CreateGuardrailModal';
import { getGuardrailsRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useCallback, useMemo, useState } from 'react';
import { useNavigate } from 'react-router';

interface GuardrailDetailActionsProps {
  config: GuardrailConfig;
}

export const GuardrailDetailActions: FC<GuardrailDetailActionsProps> = ({ config }) => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [showDuplicateModal, setShowDuplicateModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  const { mutateAsync: deleteConfig } = useGuardrailsDeleteConfig();

  // Errors propagate deliberately: the service refuses to delete a config a VirtualModel
  // still applies, and its 409 names them ("...is applied by default/my-vm. Detach it from
  // those virtual models before deleting it."). ConfirmationModal toasts a thrown error's
  // message, so letting it through says which routes are blocking; returning false would
  // instead show a fixed "please try again", which is wrong — retrying cannot succeed.
  const handleDelete = useCallback(async (): Promise<boolean> => {
    if (!config.name) return false;
    await deleteConfig({ workspace, name: config.name });
    await queryClient.invalidateQueries({
      queryKey: [`/apis/guardrails/v2/workspaces/${workspace}/configs`],
    });
    void navigate(getGuardrailsRoute(workspace));
    return true;
  }, [config.name, deleteConfig, navigate, queryClient, workspace]);

  const actions = useMemo<QuickActionItem[]>(
    () => [
      {
        label: 'Duplicate',
        onSelect: () => setShowDuplicateModal(true),
      },
      {
        label: 'Delete',
        onSelect: () => setShowDeleteModal(true),
        danger: true,
      },
    ],
    []
  );

  return (
    <>
      <QuickActionsMenuRoot actions={actions} />
      {showDuplicateModal ? (
        <CreateGuardrailModal
          open
          sourceConfig={config}
          onClose={() => setShowDuplicateModal(false)}
        />
      ) : null}
      {showDeleteModal ? (
        <DeleteConfirmationModal
          open
          simpleConfirm
          title={`Delete guardrail config: ${config.name}`}
          successText="Guardrail config deleted successfully."
          onDelete={handleDelete}
          onClose={() => setShowDeleteModal(false)}
        />
      ) : null}
    </>
  );
};
