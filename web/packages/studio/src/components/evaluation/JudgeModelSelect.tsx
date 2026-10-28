// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelSearch } from '@nemo/common/src/api/models/useModelSearch';
import { ModelSelectV2 } from '@nemo/common/src/components/ModelSelectV2/ModelSelectV2';
import type { ModelSelection } from '@nemo/common/src/components/ModelSelectV2/types';
import { getModelEntityChatStatus } from '@nemo/common/src/utils/models';
import type { InferenceParams, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { FormField } from '@nvidia/foundations-react-core';
import { useSetFieldErrorOnApiError } from '@studio/hooks/evaluation/useSetFieldErrorOnApiError';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { parseEvaluationModelValue } from '@studio/util/evaluations';
import { ReactNode, useState } from 'react';
import { type FieldValues, type Path, useController, useFormContext } from 'react-hook-form';

/** Judging needs a model that can actually serve a chat completion. */
const isChatCapable = (model: ModelEntity): boolean =>
  getModelEntityChatStatus(model) !== 'disabled';

export interface JudgeModelSelectProps<TFieldValues extends FieldValues = FieldValues> {
  /**
   * Offer each model's adapters as selections in their own right. The field then
   * holds `workspace/model::adapter` for an adapter, which
   * {@link parseEvaluationModelValue} splits back apart.
   *
   * Off by default, and off is what every judge field wants: a judge is called
   * as a plain chat endpoint, and listing adapters there only invites a
   * selection the submit path cannot express.
   */
  allowAdapters?: boolean;
  required?: boolean;
  placeholder?: string;
  slotLabel?: ReactNode;
  /** Error to show instead of the field's own, for a problem derived outside the form state. */
  slotError?: string;
  requiredMessage?: string;
  dropdownSide?: 'top' | 'bottom';
  formFieldName: Path<TFieldValues>;
  showParams?: boolean;
  inferenceParams?: Partial<InferenceParams>;
  onInferenceParamsChange?: (params: Partial<InferenceParams>) => void;
}

/**
 * Model selector specifically for judge models in LLM-as-a-Judge evaluation. Models are searched
 * and paged in the API, so nothing is fetched until the dropdown opens.
 */
export const JudgeModelSelect = <TFieldValues extends FieldValues = FieldValues>({
  allowAdapters = false,
  required = false,
  placeholder = 'Select a judge model',
  slotLabel = 'Judge Model',
  slotError,
  requiredMessage = 'Judge model is required',
  dropdownSide,
  formFieldName,
  showParams = false,
  inferenceParams,
  onInferenceParamsChange,
}: JudgeModelSelectProps<TFieldValues>) => {
  const {
    control,
    formState: { disabled, isSubmitting },
  } = useFormContext<TFieldValues>();

  const { field, fieldState } = useController({
    control,
    name: formFieldName,
    rules: required ? { required: requiredMessage } : undefined,
  });

  const workspace = useWorkspaceFromPath();
  const [open, setOpen] = useState(false);
  const { error, ...modelSearch } = useModelSearch({
    workspace: workspace ?? null,
    enabled: open && !disabled,
    include: isChatCapable,
  });

  useSetFieldErrorOnApiError<TFieldValues>(formFieldName, error);

  // Round-trips the adapter through the single string the form field holds. Both
  // halves have to agree: keeping only `selection.model` on the way in is what
  // made picking an adapter silently evaluate its base model instead.
  const parsed = field.value ? parseEvaluationModelValue(field.value as string) : null;
  const value: ModelSelection | null = parsed
    ? { model: parsed.modelUrn, ...(parsed.adapterName ? { adapter: parsed.adapterName } : {}) }
    : null;

  const handleValueChange = (selection: ModelSelection) => {
    field.onChange(
      selection.adapter ? `${selection.model}::${selection.adapter}` : selection.model
    );
  };

  const handleOpenChange = (isOpen: boolean) => {
    setOpen(isOpen);
    if (!isOpen) field.onBlur();
  };

  return (
    <FormField
      slotLabel={slotLabel}
      status={slotError || fieldState.error ? 'error' : undefined}
      slotError={slotError ?? fieldState.error?.message}
      required={required}
    >
      <ModelSelectV2
        {...modelSearch}
        value={value}
        onValueChange={handleValueChange}
        disabled={isSubmitting || disabled}
        placeholder={placeholder}
        showParams={showParams}
        inferenceParams={inferenceParams}
        onInferenceParamsChange={onInferenceParamsChange}
        onOpenChange={handleOpenChange}
        dropdownSide={dropdownSide}
        hideAdapters={!allowAdapters}
        fullWidth
      />
    </FormField>
  );
};
