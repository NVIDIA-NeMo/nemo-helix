// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelSearch } from '@nemo/common/src/api/models/useModelSearch';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { ModelSelectV2, type ModelSelection } from '@nemo/common/src/components/ModelSelectV2';
import { FormField, Stack } from '@nvidia/foundations-react-core';
import { FormSection } from '@studio/components/NewCustomizationForm/FormSection';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import {
  MODEL_FIELD_BY_BACKEND,
  type CustomizationFormFields,
} from '@studio/util/forms/customization';
import { useState } from 'react';
import { useController, useFormContext } from 'react-hook-form';

export const ModelSelectionSection = () => {
  const workspace = useWorkspaceFromPath();
  const { control, watch, formState } = useFormContext<CustomizationFormFields>();
  const backend = watch('backend');
  const disabled = formState.isSubmitting;

  const modelFieldName = MODEL_FIELD_BY_BACKEND[backend];

  const { field: modelField, fieldState: modelFieldState } = useController({
    control,
    name: modelFieldName,
  });

  const [open, setOpen] = useState(false);
  const modelSearch = useModelSearch({
    workspace,
    enabled: open,
    filter: { fileset: true },
  });

  const selectedValue: ModelSelection | null = modelField.value
    ? { model: modelField.value as string }
    : null;

  const handleModelChange = (selection: ModelSelection) => {
    modelField.onChange(selection.model);
  };

  return (
    <FormSection title="Model">
      <Stack gap="density-md">
        <FormField
          slotLabel="Base Model"
          required
          status={modelFieldState.error ? 'error' : undefined}
          slotError={modelFieldState.error?.message}
        >
          <ModelSelectV2
            {...modelSearch}
            value={selectedValue}
            onValueChange={handleModelChange}
            onOpenChange={setOpen}
            disabled={disabled}
            emptyMessage="No fine-tunable models found"
            hideAdapters
            fullWidth
          />
        </FormField>
        <ControlledTextInput
          useControllerProps={{ name: 'outputName', control }}
          label="Output Model Name"
          required
          disabled={disabled}
        />
        <ControlledTextInput
          useControllerProps={{ name: 'description', control }}
          label="Description"
          disabled={disabled}
        />
      </Stack>
    </FormSection>
  );
};
