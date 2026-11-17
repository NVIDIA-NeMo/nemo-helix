// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import {
  getEntitiesListWorkspacesQueryKey,
  useEntitiesDeleteWorkspace,
} from '@nemo/sdk/generated/platform/entity-store';
import type { WorkspacesPage } from '@nemo/sdk/generated/platform/schema';
import { queryClient } from '@studio/api/queryClient';
import { useRecentWorkspaces } from '@studio/components/WorkspaceDropdown/useRecentWorkspaces';
import { getWorkspaceDetailsDefaultRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { useNavigate } from 'react-router';

interface DeleteWorkspaceModalProps {
  workspace: string;
  open: boolean;
  onClose: () => void;
}

export const DeleteWorkspaceModal: FC<DeleteWorkspaceModalProps> = ({
  workspace,
  open,
  onClose,
}) => {
  const navigate = useNavigate();
  const { mutateAsync: deleteWorkspace } = useEntitiesDeleteWorkspace();
  const { removeRecentWorkspace } = useRecentWorkspaces();

  const handleDelete = async () => {
    try {
      await deleteWorkspace({ name: workspace });

      queryClient.setQueriesData<WorkspacesPage>(
        { queryKey: getEntitiesListWorkspacesQueryKey() },
        (oldData) => {
          if (!oldData) return oldData;
          return {
            ...oldData,
            data: oldData.data.filter((w) => w.name !== workspace),
          };
        }
      );
      queryClient.invalidateQueries({ queryKey: getEntitiesListWorkspacesQueryKey() });

      removeRecentWorkspace(workspace);
      navigate(getWorkspaceDetailsDefaultRoute(DEFAULT_WORKSPACE));
      return true;
    } catch {
      return false;
    }
  };

  return (
    <DeleteConfirmationModal
      open={open}
      onClose={onClose}
      entity="workspace"
      ids={[workspace]}
      description="The workspace and everything in it are deleted and can't be restored."
      onDelete={handleDelete}
    />
  );
};
