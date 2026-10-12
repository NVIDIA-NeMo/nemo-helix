// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelSearch } from '@nemo/common/src/api/models/useModelSearch';
import { ModelSelectV2 } from '@nemo/common/src/components/ModelSelectV2/ModelSelectV2';
import type {
  ModelAvailability,
  ModelSelection,
} from '@nemo/common/src/components/ModelSelectV2/types';
import { getModelEntityChatStatus, groupModelsByWorkspace } from '@nemo/common/src/utils/models';
import type { InferenceParams, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { FormField } from '@nvidia/foundations-react-core';
import { useSetFieldErrorOnApiError } from '@studio/hooks/evaluation/useSetFieldErrorOnApiError';
import {
  getPendingJobName,
  toPendingModelEntity,
  usePendingCustomizations,
  type PendingCustomization,
} from '@studio/hooks/usePendingCustomizations';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getFormattedCustomizationStatus } from '@studio/util/customizations';
import { useMemo, useState } from 'react';
import { type FieldValues, type Path, useController, useFormContext } from 'react-hook-form';

/** Judging needs a model that can actually serve a chat completion. */
const isChatCapable = (model: ModelEntity): boolean =>
  getModelEntityChatStatus(model) !== 'disabled';

/**
 * An in-flight run matches on anything the user might have in mind for it: the name the
 * model will take, the fully-qualified identifier, or the name of the job itself -- which
 * is often the only name they have seen, the model not existing yet.
 */
const pendingMatchesSearch = (pending: PendingCustomization, search: string): boolean => {
  const needle = search.trim().toLowerCase();
  if (!needle) return true;
  return [
    pending.outputModelName,
    pending.jobName,
    `${pending.workspace}/${pending.outputModelName}`,
  ].some((candidate) => candidate.toLowerCase().includes(needle));
};

export interface JudgeModelSelectProps<TFieldValues extends FieldValues = FieldValues> {
  required?: boolean;
  placeholder?: string;
  slotLabel?: string;
  /** Error to show instead of the field's own, for a problem derived outside the form state. */
  slotError?: string;
  requiredMessage?: string;
  dropdownSide?: 'top' | 'bottom';
  formFieldName: Path<TFieldValues>;
  showParams?: boolean;
  inferenceParams?: Partial<InferenceParams>;
  onInferenceParamsChange?: (params: Partial<InferenceParams>) => void;
  /**
   * List models that cannot be used yet instead of hiding them -- undeployed models, and
   * fine-tuning runs still training. Each is shown disabled with the reason.
   *
   * The default (hiding them) suits a judge model, where only a servable model is ever a
   * valid answer. It does not suit picking the model a run just produced: a user who has
   * just launched a fine-tune and finds nothing in the list cannot tell whether the run is
   * still going, failed, or was never submitted.
   */
  showUnavailable?: boolean;
}

/**
 * Model selector specifically for judge models in LLM-as-a-Judge evaluation. Models are searched
 * and paged in the API, so nothing is fetched until the dropdown opens.
 */
export const JudgeModelSelect = <TFieldValues extends FieldValues = FieldValues>({
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
  showUnavailable = false,
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
    include: showUnavailable ? undefined : isChatCapable,
  });

  useSetFieldErrorOnApiError<TFieldValues>(formFieldName, error);

  // Polled whether or not the dropdown is open: a run finishing invalidates the model
  // lists, so the model becomes selectable on the next open rather than after a reload.
  const { pending } = usePendingCustomizations(workspace, {
    enabled: showUnavailable && !disabled,
  });

  /** In-flight runs whose model entity does not exist yet, as placeholder rows. */
  const pendingByJobName = useMemo(() => {
    if (!showUnavailable) return new Map<string, PendingCustomization>();
    const existingNames = new Set(modelSearch.models.map((model) => model.name));
    return new Map(
      pending
        .filter(
          (p) =>
            !existingNames.has(p.outputModelName) && pendingMatchesSearch(p, modelSearch.search)
        )
        .map((p) => [p.jobName, p])
    );
  }, [modelSearch.models, modelSearch.search, pending, showUnavailable]);

  const groups = useMemo(() => {
    if (pendingByJobName.size === 0) return modelSearch.groups;
    const placeholders = [...pendingByJobName.values()].map(toPendingModelEntity);
    return groupModelsByWorkspace([...modelSearch.models, ...placeholders], { sort: true });
  }, [modelSearch.groups, modelSearch.models, pendingByJobName]);

  const getAvailability = useMemo(() => {
    if (!showUnavailable) return undefined;
    return (model: ModelEntity): ModelAvailability | undefined => {
      const jobName = getPendingJobName(model);
      if (jobName) {
        const status = pendingByJobName.get(jobName)?.status;
        const formatted = status ? getFormattedCustomizationStatus(status) : undefined;
        return {
          disabled: true,
          reason: formatted
            ? `Fine-tuning ${formatted.toLowerCase()} — not ready to evaluate`
            : 'Fine-tuning in progress — not ready to evaluate',
        };
      }
      if (!isChatCapable(model)) {
        return { disabled: true, reason: 'Not deployed — deploy it from Fine-tuning first' };
      }
      return undefined;
    };
  }, [pendingByJobName, showUnavailable]);

  const value: ModelSelection | null = field.value ? { model: field.value as string } : null;

  const handleValueChange = (selection: ModelSelection) => {
    field.onChange(selection.model);
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
        groups={groups}
        getAvailability={getAvailability}
        value={value}
        onValueChange={handleValueChange}
        disabled={isSubmitting || disabled}
        placeholder={placeholder}
        showParams={showParams}
        inferenceParams={inferenceParams}
        onInferenceParamsChange={onInferenceParamsChange}
        onOpenChange={handleOpenChange}
        dropdownSide={dropdownSide}
        fullWidth
      />
    </FormField>
  );
};
