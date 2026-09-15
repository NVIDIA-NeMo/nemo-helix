/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useModelSearch } from '@nemo/common/src/api/models/useModelSearch';
import { FilesetSearchableSelect } from '@nemo/common/src/components/FilesetSearchableSelect';
import { type ModelSelection, ModelSelectV2 } from '@nemo/common/src/components/ModelSelectV2';
import { RadioCard } from '@nemo/common/src/components/RadioCard';
import { Flex, FormField, RadioGroupRoot, Stack } from '@nvidia/foundations-react-core';
import { canFineTuneModel } from '@studio/hooks/useModelCustomizationEligibility';
import { EngineFields } from '@studio/routes/NewDeploymentRoute/EngineFields';
import { GPULoraFields } from '@studio/routes/NewDeploymentRoute/GPULoraFields';
import {
  WORKSPACE_PICKER_FILESET,
  WORKSPACE_PICKER_MODEL,
  type WizardFormValues,
} from '@studio/routes/NewDeploymentRoute/schema';
import { useState, type FC } from 'react';
import { useController, useWatch, type Control, type FieldErrors } from 'react-hook-form';

export type WorkspaceSourceFieldsProps = {
  workspace: string;
  queryEnabled: boolean;
  control: Control<WizardFormValues>;
  errors: FieldErrors<WizardFormValues>;
  onPickerTypeChange: (
    value: typeof WORKSPACE_PICKER_MODEL | typeof WORKSPACE_PICKER_FILESET
  ) => void;
};

export const WorkspaceSourceFields: FC<WorkspaceSourceFieldsProps> = ({
  workspace,
  queryEnabled,
  control,
  errors,
  onPickerTypeChange,
}) => {
  const pickerType = useWatch({ control, name: 'workspacePickerType' });

  return (
    <Stack gap="density-xl">
      <RadioGroupRoot
        name="workspace-picker-type"
        orientation="horizontal"
        className="w-full"
        value={pickerType}
        onValueChange={(v) =>
          onPickerTypeChange(v as typeof WORKSPACE_PICKER_MODEL | typeof WORKSPACE_PICKER_FILESET)
        }
      >
        <Flex gap="density-xl" className="w-full *:flex-1">
          <RadioCard
            value={WORKSPACE_PICKER_MODEL}
            label="Existing model"
            description="Deploy a registered model entity from this workspace."
          />
          <RadioCard
            value={WORKSPACE_PICKER_FILESET}
            label="Existing fileset"
            description="Deploy a fileset of weights; a model entity is registered automatically."
          />
        </Flex>
      </RadioGroupRoot>

      {pickerType === WORKSPACE_PICKER_MODEL ? (
        <WorkspaceModelPicker
          workspace={workspace}
          queryEnabled={queryEnabled}
          control={control}
          errorMessage={errors.modelRef?.message}
        />
      ) : (
        <FilesetSearchableSelect
          workspace={workspace}
          queryEnabled={queryEnabled}
          useControllerProps={{ control, name: 'fileset' }}
          formFieldProps={{
            slotLabel: 'Fileset',
            slotInfo: 'A model entity will be registered for the selected fileset.',
            slotError: errors.fileset?.message,
          }}
        />
      )}

      <EngineFields control={control} errors={errors} />

      <GPULoraFields control={control} errors={errors} />
    </Stack>
  );
};

type WorkspaceModelPickerProps = {
  workspace: string;
  queryEnabled: boolean;
  control: Control<WizardFormValues>;
  errorMessage?: string;
};

const WorkspaceModelPicker: FC<WorkspaceModelPickerProps> = ({
  workspace,
  queryEnabled,
  control,
  errorMessage,
}) => {
  const { field } = useController({ control, name: 'modelRef' });
  const [open, setOpen] = useState(false);

  // Deployment needs local weights: the puller resolves `model.fileset` and fails
  // if it is absent, which is what remote provider catalog entries look like.
  // Same condition as fine-tuning, so the predicate is shared.
  const modelSearch = useModelSearch({
    workspace,
    enabled: queryEnabled && open,
    include: canFineTuneModel,
  });

  const value: ModelSelection | null = field.value ? { model: field.value as string } : null;

  return (
    <FormField
      slotLabel="Model"
      slotInfo="Registered model with weights in this workspace. Models served by a remote provider cannot be deployed."
      slotError={errorMessage}
    >
      <ModelSelectV2
        {...modelSearch}
        value={value}
        onValueChange={(selection) => field.onChange(selection.model)}
        placeholder="Select a model"
        hideAdapters
        fullWidth
        onOpenChange={(nextOpen) => {
          setOpen(nextOpen);
          if (!nextOpen) field.onBlur();
        }}
      />
    </FormField>
  );
};
