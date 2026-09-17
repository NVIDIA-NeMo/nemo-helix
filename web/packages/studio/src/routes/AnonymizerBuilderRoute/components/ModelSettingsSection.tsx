// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSearchableSelect } from '@nemo/common/src/components/form/ControlledSearchableSelect';
import { ParamsDropdown } from '@nemo/common/src/components/ModelSelectV2/ParamsDropdown';
import type { InferenceParams } from '@nemo/sdk/generated/platform/schema';
import { Divider, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import {
  activeRolesForStrategy,
  ANONYMIZER_PARAM_METADATA,
  DETECTOR_GROUP_LABELS,
  DETECTOR_MODEL_HINT,
  DETECTOR_ROLE,
  ROLE_LABELS,
  supportsSamplingParams,
} from '@studio/routes/AnonymizerBuilderRoute/constants';
import type { AnonymizerFormData } from '@studio/routes/AnonymizerBuilderRoute/schema';
import { useAnonymizerModels } from '@studio/routes/AnonymizerBuilderRoute/useAnonymizerModels';
import { detectorOptions } from '@studio/routes/AnonymizerBuilderRoute/utils';
import { useMemo, useState, type FC } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

export const ModelSettingsSection: FC = () => {
  const { control, setValue } = useFormContext<AnonymizerFormData>();
  const strategy = useWatch({ control, name: 'strategy' });
  const roleModelsValue = useWatch({ control, name: 'roleModels' });
  const [openParamsRole, setOpenParamsRole] = useState<string | null>(null);

  const roles = useMemo(() => activeRolesForStrategy(strategy), [strategy]);
  const { models, items, isLoading, applyModel } = useAnonymizerModels();

  const detectorItems = useMemo(() => detectorOptions(items, models), [items, models]);

  return (
    <Stack gap="density-2xl">
      {roles.map((role, index) => (
        <Stack key={role} gap="density-lg" data-testid={`role-settings-${role}`}>
          {index > 0 && <Divider orientation="horizontal" width="small" />}
          <Text kind="label/bold/lg">{ROLE_LABELS[role] ?? role}</Text>
          <Flex gap="density-md" align="end">
            <div className="grow">
              <ControlledSearchableSelect
                aria-label={ROLE_LABELS[role] ?? role}
                options={role === DETECTOR_ROLE ? detectorItems : items}
                groupLabels={role === DETECTOR_ROLE ? DETECTOR_GROUP_LABELS : undefined}
                isLoading={isLoading}
                triggerPlaceholder="Select a model"
                searchPlaceholder="Search models..."
                emptyMessage={isLoading ? 'Loading models...' : 'No models in this workspace.'}
                onChange={(value) => applyModel(role, value)}
                useControllerProps={{
                  name: `roleModels.${role}.modelId`,
                  control,
                }}
                formFieldProps={{
                  slotLabel: 'Model',
                  required: true,
                  ...(role === DETECTOR_ROLE && { slotInfo: DETECTOR_MODEL_HINT }),
                }}
              />
            </div>
            {supportsSamplingParams(role) && (
              <ParamsDropdown
                open={openParamsRole === role}
                onOpenChange={(next) => setOpenParamsRole(next ? role : null)}
                inferenceParams={roleModelsValue?.[role]?.params as Partial<InferenceParams>}
                onInferenceParamsChange={(params) =>
                  setValue(`roleModels.${role}.params`, params as Record<string, unknown>)
                }
                fieldMetadata={ANONYMIZER_PARAM_METADATA}
              />
            )}
          </Flex>
        </Stack>
      ))}
    </Stack>
  );
};
