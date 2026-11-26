// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSearchableSelect } from '@nemo/common/src/components/form/ControlledSearchableSelect';
import type { ModelSelection } from '@nemo/common/src/components/ModelSelectV2/types';
import { WorkspaceModelSelect } from '@nemo/common/src/components/ModelSelectV2/WorkspaceModelSelect';
import { hasModelProvider } from '@nemo/common/src/utils/models';
import type { InferenceParams } from '@nemo/sdk/generated/platform/schema';
import { Divider, FormField, Stack, Text } from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import {
  activeRolesForStrategy,
  ANONYMIZER_PARAM_METADATA,
  DETECTOR_GROUP_LABELS,
  DETECTOR_MODEL_HINT,
  DETECTOR_ROLE,
  ROLE_LABELS,
} from '@studio/routes/AnonymizerBuilderRoute/constants';
import type { AnonymizerFormData } from '@studio/routes/AnonymizerBuilderRoute/schema';
import { useAnonymizerModels } from '@studio/routes/AnonymizerBuilderRoute/useAnonymizerModels';
import { detectorOptions } from '@studio/routes/AnonymizerBuilderRoute/utils';
import { useMemo, type FC } from 'react';
import { useController, useFormContext, useWatch } from 'react-hook-form';

interface RoleModelSelectProps {
  role: string;
  applyModel: (role: string, id: string) => void;
}

const RoleModelSelect: FC<RoleModelSelectProps> = ({ role, applyModel }) => {
  const { control, setValue } = useFormContext<AnonymizerFormData>();
  const workspace = useWorkspaceFromPath();
  const { field, fieldState } = useController({ control, name: `roleModels.${role}.modelId` });
  const params = useWatch({ control, name: `roleModels.${role}.params` });
  const value = useMemo(() => (field.value ? { model: field.value } : null), [field.value]);
  const error = fieldState.error?.message;

  const handleValueChange = (selection: ModelSelection) => {
    field.onChange(selection.model);
    applyModel(role, selection.model);
  };

  return (
    <FormField slotLabel="Model" required slotError={error} status={error ? 'error' : undefined}>
      <WorkspaceModelSelect
        workspace={workspace}
        value={value}
        onValueChange={handleValueChange}
        include={hasModelProvider}
        hideAdapters
        placeholder="Select a model"
        fullWidth
        dropdownSide="bottom"
        aria-label={ROLE_LABELS[role] ?? role}
        showParams
        paramsFieldMetadata={ANONYMIZER_PARAM_METADATA}
        inferenceParams={params as Partial<InferenceParams>}
        onInferenceParamsChange={(next) =>
          setValue(`roleModels.${role}.params`, next as Record<string, unknown>)
        }
      />
    </FormField>
  );
};

/**
 * The entity detector keeps a plain select: it lists a built-in GLiNER2 option that is not a model
 * and groups suggested NER models first. The job sends each model's served name, so every pick
 * goes through `applyModel`, which resolves it from the providers' served models.
 */
export const ModelSettingsSection: FC = () => {
  const { control } = useFormContext<AnonymizerFormData>();
  const strategy = useWatch({ control, name: 'strategy' });

  const roles = useMemo(() => activeRolesForStrategy(strategy), [strategy]);
  const { models, items, isLoading, applyModel } = useAnonymizerModels();

  const detectorItems = useMemo(() => detectorOptions(items, models), [items, models]);

  return (
    <Stack gap="density-2xl">
      {roles.map((role, index) => (
        <Stack key={role} gap="density-lg" data-testid={`role-settings-${role}`}>
          {index > 0 ? <Divider orientation="horizontal" width="small" /> : null}
          <Text kind="label/bold/lg">{ROLE_LABELS[role] ?? role}</Text>
          {role === DETECTOR_ROLE ? (
            <ControlledSearchableSelect
              aria-label={ROLE_LABELS[role] ?? role}
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
          ) : (
            <RoleModelSelect role={role} applyModel={applyModel} />
          )}
        </Stack>
      ))}
    </Stack>
  );
};
