// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { useAgentsListAgentImages } from '@nemo/sdk/generated/agents/agent-images';
import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import { Checkbox, Stack, Text } from '@nvidia/foundations-react-core';
import { isPlatformBuiltImage } from '@studio/routes/agents/AgentDetailRoute/helpers';
import { type FC, useState } from 'react';

interface DeleteDeploymentModalProps {
  workspace: string;
  deployment: AgentDeployment;
  onDelete: (removeImage: boolean) => Promise<boolean>;
  onClose: () => void;
}

/** Confirms a deployment delete, offering to remove its image when the platform built it. */
export const DeleteDeploymentModal: FC<DeleteDeploymentModalProps> = ({
  workspace,
  deployment,
  onDelete,
  onClose,
}) => {
  const image = deployment.image ?? '';
  const offerImage = isPlatformBuiltImage(image);
  const { data } = useAgentsListAgentImages(workspace, undefined, {
    query: { enabled: offerImage },
  });
  const otherUsers =
    data?.data
      .find((candidate) => candidate.tags.includes(image))
      ?.deployments?.filter((name) => name !== deployment.name) ?? [];
  // Until the user picks, follow whether anything else still uses the image.
  const [choice, setChoice] = useState<boolean>();
  const removeImage = offerImage && (choice ?? otherUsers.length === 0);

  return (
    <DeleteConfirmationModal
      open
      title="Delete Deployment"
      successText="Successfully queued deployment for deletion."
      onDelete={() => onDelete(removeImage)}
      onClose={onClose}
      simpleConfirm
    >
      {offerImage && (
        <Stack gap="density-xs">
          <Checkbox
            checked={removeImage}
            onChange={(event) => setChoice(event.target.checked)}
            slotLabel="Also remove the image"
          />
          <Text kind="body/regular/xs" className="break-all font-mono text-secondary">
            {image}
          </Text>
          {otherUsers.length > 0 && (
            <Text kind="body/regular/xs" className="text-secondary">
              Also used by {otherUsers.join(', ')}. The image is kept while another deployment uses
              it.
            </Text>
          )}
        </Stack>
      )}
    </DeleteConfirmationModal>
  );
};
