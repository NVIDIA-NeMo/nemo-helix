// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSearchableSelect } from '@nemo/common/src/components/form/ControlledSearchableSelect';
import { ModelSelectV2 } from '@nemo/common/src/components/ModelSelectV2';
import type { InferenceParams } from '@nemo/sdk/generated/platform/schema';
import { Divider, FormField, Stack, Text } from '@nvidia/foundations-react-core';
import {
  activeRolesForStrategy,
  ANONYMIZER_PARAM_METADATA,
  DETECTOR_GROUP_LABELS,
  DETECTOR_MODEL_HINT,
  ROLE_LABELS,
  supportsSamplingParams,
} from '@studio/routes/AnonymizerBuilderRoute/constants';
import type { AnonymizerFormData } from '@studio/routes/AnonymizerBuilderRoute/schema';
import { useAnonymizerModels } from '@studio/routes/AnonymizerBuilderRoute/useAnonymizerModels';
import { detectorOptions } from '@studio/routes/AnonymizerBuilderRoute/utils';
import { useMemo, type FC } from 'react';
import { useFormContext, useFormState, useWatch } from 'react-hook-form';

/**
 * Sampling roles use the shared model + params picker, fed from the workspace's providers so every
 * option carries the served name the job sends. The entity detector keeps a searchable select: it
 * takes no sampling params, defaults to the built-in GLiNER2 (a sentinel, not a model), and groups
 * recognised NER families as suggestions, none of which the shared picker models.
 */
export const ModelSettingsSection: FC = () => {
  const { control, setValue } = useFormContext<AnonymizerFormData>();
  const strategy = useWatch({ control, name: 'strategy' });
  const roleModelsValue = useWatch({ control, name: 'roleModels' });
  const { errors } = useFormState({ control, name: 'roleModels' });

  const roles = useMemo(() => activeRolesForStrategy(strategy), [strategy]);
  const { models, items, groups, isLoading, applyModel } = useAnonymizerModels();

  const detectorItems = useMemo(() => detectorOptions(items, models), [items, models]);

  return (
    <Stack gap="density-2xl">
      {roles.map((role, index) => {
        const label = ROLE_LABELS[role] ?? role;
        const roleModel = roleModelsValue?.[role];
        const error = errors.roleModels?.[role]?.modelId?.message;
        return (
          <Stack key={role} gap="density-lg" data-testid={`role-settings-${role}`}>
            {index > 0 && <Divider orientation="horizontal" width="small" />}
            <Text kind="label/bold/lg">{label}</Text>
            {supportsSamplingParams(role) ? (
              <FormField
                slotLabel="Model"
                required
                slotError={error}
                status={error ? 'error' : undefined}
              >
                <ModelSelectV2
                  groups={groups}
                  loading={isLoading}
                  emptyMessage="No models in this workspace."
                  value={roleModel?.modelId ? { model: roleModel.modelId } : null}
                  onValueChange={(selection) => {
                    applyModel(role, selection.model);
                    setValue(`roleModels.${role}.modelId`, selection.model, {
                      shouldValidate: true,
                    });
                  }}
                  placeholder="Select a model"
                  fullWidth
                  showParams
                  paramsFieldMetadata={ANONYMIZER_PARAM_METADATA}
                  inferenceParams={roleModel?.params as Partial<InferenceParams> | undefined}
                  onInferenceParamsChange={(params) =>
                    setValue(`roleModels.${role}.params`, params as Record<string, unknown>)
                  }
                  aria-label={label}
                />
              </FormField>
            ) : (
              <ControlledSearchableSelect
                aria-label={label}
                options={detectorItems}
                groupLabels={DETECTOR_GROUP_LABELS}
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
                  slotInfo: DETECTOR_MODEL_HINT,
                }}
              />
            )}
          </Stack>
        );
      })}
    </Stack>
  );
};
