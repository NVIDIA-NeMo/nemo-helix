// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getBaseModelsQueryKeyPrefix } from '@nemo/common/src/api/entity-store/useBaseModels';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { getURNFromNamedEntityRef } from '@nemo/common/src/namedEntity';
import {
  getModelsListModelsQueryKey,
  useModelsDeleteModel,
} from '@nemo/sdk/generated/platform/models';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Button, Flex } from '@nvidia/foundations-react-core';
import { CUSTOMIZER_ENABLED } from '@studio/constants/environment';
import { canFineTuneModel } from '@studio/hooks/useModelCustomizationEligibility';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getNewCustomizationFormRoute, getWorkspaceBaseModelsRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useState } from 'react';
import { Link, useNavigate } from 'react-router';

export interface DetailsActionsProps {
  model: ModelEntity;
  canDelete: boolean;
}

export const DetailsActions: FC<DetailsActionsProps> = ({ model, canDelete }) => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { mutateAsync: deleteModel } = useModelsDeleteModel();
  const [deleteOpen, setDeleteOpen] = useState(false);

  const handleDelete = async () => {
    try {
      await deleteModel({ workspace: model.workspace, name: model.name });
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: getModelsListModelsQueryKey(model.workspace, {}),
        }),
        queryClient.invalidateQueries({ queryKey: getBaseModelsQueryKeyPrefix(model.workspace) }),
      ]);
      navigate(getWorkspaceBaseModelsRoute(workspace), { replace: true });
      return true;
    } catch {
      return false;
    }
  };

  if (!CUSTOMIZER_ENABLED && !canDelete) return null;

  return (
    <Flex gap="density-sm">
      {canDelete && (
        <>
          <Button kind="secondary" color="danger" onClick={() => setDeleteOpen(true)}>
            Delete
          </Button>
          {deleteOpen && (
            <DeleteConfirmationModal
              open
              title={`Delete model: ${model.name}`}
              description="This permanently removes the model entity from the workspace. Related deployments and resources may be affected."
              confirmationText={model.name}
              simpleConfirm
              successText={`Successfully deleted model ${model.name}`}
              errorText="Failed to delete the model. Try again or check your permissions."
              onClose={() => setDeleteOpen(false)}
              onDelete={handleDelete}
            />
          )}
        </>
      )}
      {CUSTOMIZER_ENABLED &&
        (canFineTuneModel(model) ? (
          <Button color="brand" asChild>
            <Link
              to={getNewCustomizationFormRoute(workspace, {
                model: getURNFromNamedEntityRef(model),
              })}
            >
              Customize
            </Link>
          </Button>
        ) : (
          <Button color="brand" disabled>
            Customize
          </Button>
        ))}
    </Flex>
  );
};
