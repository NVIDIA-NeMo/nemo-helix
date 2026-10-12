// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelsGetModel } from '@nemo/sdk/generated/platform/models';
import { Anchor, Stack, Text } from '@nvidia/foundations-react-core';
import { useModelChatAvailability } from '@studio/hooks/useModelChatAvailability';
import { getCustomizationJobListRoute } from '@studio/routes/utils';
import type { FC } from 'react';

interface OutputModelValueProps {
  workspace: string;
  /** Name the run registers its output model under. */
  modelName: string | undefined;
}

/** What still stands between the model and a user who wants to use it. */
const readinessNote = (status: string): string => {
  switch (status) {
    case 'enabled':
      return 'Ready — open it to test or evaluate it';
    case 'pending':
      return 'Deploying — open it to follow the deployment';
    default:
      return 'Not deployed yet — open it to deploy or test it';
  }
};

/**
 * The model a fine-tuning run produces, linked to its entry once it exists.
 *
 * Until the run finishes there is no entity to link to, and saying so matters: the gap
 * between a run's training steps ending and its model appearing is exactly where someone
 * goes looking for a model that is not there yet, with nothing to tell them whether to
 * keep waiting.
 */
export const OutputModelValue: FC<OutputModelValueProps> = ({ workspace, modelName }) => {
  const { data: model, isLoading } = useModelsGetModel(workspace, modelName ?? '', undefined, {
    query: { enabled: Boolean(modelName), retry: false },
  });
  const { modelChatStatus, isLoading: isStatusLoading } = useModelChatAvailability(model);

  if (!modelName) return <Text>-</Text>;

  if (!model) {
    return (
      <Stack gap="0" className="min-w-0">
        <Text className="truncate">{modelName}</Text>
        {!isLoading && (
          <Text kind="body/regular/sm" className="text-secondary">
            Registered when this run completes
          </Text>
        )}
      </Stack>
    );
  }

  return (
    <Stack gap="0" className="min-w-0">
      <Anchor
        href={getCustomizationJobListRoute(workspace, { model: modelName })}
        className="truncate text-success"
      >
        {modelName}
      </Anchor>
      {!isStatusLoading && (
        <Text kind="body/regular/sm" className="text-secondary">
          {readinessNote(modelChatStatus)}
        </Text>
      )}
    </Stack>
  );
};
