// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { getModelsGetModelQueryKey } from '@nemo/sdk/generated/platform/models';
import type { ModelDeployment, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Button } from '@nvidia/foundations-react-core';
import { useDeleteDeploymentAndConfig } from '@studio/routes/DeploymentsListRoute/useDeleteDeploymentAndConfig';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useState } from 'react';

export interface DeleteDeploymentButtonProps {
  model: ModelEntity;
  deployment: ModelDeployment;
}

export const DeleteDeploymentButton: FC<DeleteDeploymentButtonProps> = ({ model, deployment }) => {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const { deleteDeploymentAndConfig } = useDeleteDeploymentAndConfig(deployment.workspace);

  const handleDelete = async () => {
    try {
      await deleteDeploymentAndConfig(deployment);
      await queryClient.invalidateQueries({
        queryKey: getModelsGetModelQueryKey(model.workspace, model.name),
      });
      return true;
    } catch {
      return false;
    }
  };

  return (
    <>
      <Button kind="secondary" color="danger" type="button" onClick={() => setOpen(true)}>
        Delete
      </Button>
      {open && (
        <DeleteConfirmationModal
          open
          title={`Delete deployment: ${deployment.name}`}
          description={`This deletes the deployment and its configuration. ${model.name} stops being served until it is deployed again.`}
          confirmationText={deployment.name}
          simpleConfirm
          successText={`Deleting deployment ${deployment.name}`}
          errorText="Failed to delete the deployment. Try again or check your permissions."
          onClose={() => setOpen(false)}
          onDelete={handleDelete}
        />
      )}
    </>
  );
};
