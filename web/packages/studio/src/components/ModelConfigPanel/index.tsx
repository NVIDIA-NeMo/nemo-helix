// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import type { ModelSelection } from '@nemo/common/src/components/ModelSelectV2/types';
import { WorkspaceModelSelect } from '@nemo/common/src/components/ModelSelectV2/WorkspaceModelSelect';
import { SliderWithTextInput } from '@nemo/common/src/components/SliderWithTextInput';
import type { InferenceParams } from '@nemo/sdk/generated/platform/schema';
import { Button, Flex, FormField, Stack, Text } from '@nvidia/foundations-react-core';
import { CardIconBadge } from '@studio/components/common/SelectableCard';
import {
  DEFAULT_MAX_PARALLEL_REQUESTS,
  MAX_PARALLEL_REQUESTS_MAX,
  MAX_PARALLEL_REQUESTS_MIN,
} from '@studio/constants/constants';
import {
  providerForSelection,
  validateModelAlias,
} from '@studio/routes/DataDesignerJobBuildRoute/models';
import type { JobBuilderFormValues } from '@studio/routes/DataDesignerJobBuildRoute/useJobBuilder';
import { Cpu, Trash2, X } from 'lucide-react';
import type { FC } from 'react';
import { useController, useFormContext, useWatch } from 'react-hook-form';

const EMPTY_INFERENCE_PARAMS: Partial<InferenceParams> = {};

export interface ModelConfigPanelProps {
  modelId: string;
  workspace: string;
  onRemove: () => void;
  onClose: () => void;
}

export const ModelConfigPanel: FC<ModelConfigPanelProps> = ({ modelId, ...props }) => {
  const { control } = useFormContext<JobBuilderFormValues>();
  const modelIds = useWatch({
    control,
    name: 'models',
    compute: (models) => models.map((model) => model.id),
  });
  const modelIndex = modelIds.indexOf(modelId);
  return modelIndex < 0 ? null : (
    <ModelConfigFields key={`${modelId}:${modelIndex}`} {...props} modelIndex={modelIndex} />
  );
};

interface ModelConfigFieldsProps extends Omit<ModelConfigPanelProps, 'modelId'> {
  modelIndex: number;
}

const ModelConfigFields: FC<ModelConfigFieldsProps> = ({
  modelIndex,
  workspace,
  onRemove,
  onClose,
}) => {
  const { control, getValues, setValue } = useFormContext<JobBuilderFormValues>();
  const aliasPath = `models.${modelIndex}.alias` as const;
  const { field: modelField } = useController({ control, name: `models.${modelIndex}.model` });
  const { field: providerField } = useController({
    control,
    name: `models.${modelIndex}.provider`,
  });
  const { field: inferenceParamsField } = useController({
    control,
    name: `models.${modelIndex}.inferenceParams`,
  });
  const modelCount = getValues('models').length;
  const aliases = useWatch({
    control,
    name: Array.from({ length: modelCount }, (_, index) => `models.${index}.alias` as const),
  });
  const alias = useWatch({ control, name: aliasPath }) ?? '';
  const aliasError = validateModelAlias(
    alias,
    new Set(aliases.filter((value, index) => index !== modelIndex && Boolean(value)))
  );
  const modelValue: ModelSelection | null = modelField.value ? { model: modelField.value } : null;

  const { field: aliasField } = useController({ control, name: aliasPath });

  const handleModelChange = (selection: ModelSelection) => {
    const oldAlias = aliasField.value as string;
    const newAlias = selection.model.split('/').pop()?.split('@')[0] ?? selection.model;

    modelField.onChange(selection.model);
    providerField.onChange(providerForSelection(selection));

    if (newAlias && newAlias !== oldAlias) {
      aliasField.onChange(newAlias);
      const columns = getValues('columns');
      const updated = columns.map((col) =>
        col.values?.model_alias === oldAlias
          ? { ...col, values: { ...col.values, model_alias: newAlias } }
          : col
      );
      setValue('columns', updated);
    }
  };

  return (
    <aside
      aria-label={`Configure ${alias || 'model'}`}
      className="flex h-full w-full flex-col bg-surface-base"
    >
      <Flex
        align="start"
        justify="between"
        gap="density-md"
        className="shrink-0 border-b border-base p-density-lg"
      >
        <Flex align="center" gap="density-sm" className="min-w-0">
          <CardIconBadge>
            <Cpu size={16} className="text-accent-teal" aria-hidden />
          </CardIconBadge>
          <Stack gap="density-xxs" className="min-w-0">
            <Text kind="body/bold/md" className="truncate">
              Model
            </Text>
            <Text kind="body/regular/xs" className="text-secondary truncate">
              Referenced by LLM columns via its alias
            </Text>
          </Stack>
        </Flex>
        <Button
          kind="tertiary"
          color="neutral"
          size="small"
          aria-label="Close model config"
          onClick={onClose}
        >
          <X size={16} aria-hidden />
        </Button>
      </Flex>

      <Stack gap="density-lg" padding="density-lg" className="min-h-0 flex-1 overflow-y-auto">
        <ControlledTextInput
          label="Alias"
          required
          useControllerProps={{ name: aliasPath }}
          formFieldProps={{
            slotInfo: 'LLM columns reference this model via their model alias.',
            status: alias && aliasError ? 'error' : undefined,
            slotError: alias ? (aliasError ?? undefined) : undefined,
          }}
          placeholder="e.g. default"
          aria-label="Model alias"
        />

        <FormField slotLabel="Model" required slotInfo="Model and inference parameters.">
          <WorkspaceModelSelect
            workspace={workspace}
            value={modelValue}
            onValueChange={handleModelChange}
            placeholder="Select a model"
            showParams
            fullWidth
            dropdownSide="bottom"
            inferenceParams={inferenceParamsField.value ?? EMPTY_INFERENCE_PARAMS}
            onInferenceParamsChange={inferenceParamsField.onChange}
            aria-label="Model selector"
          />
        </FormField>

        <SliderWithTextInput
          id="max-parallel-requests-slider"
          field={{
            name: 'max_parallel_requests',
            value:
              (inferenceParamsField.value?.max_parallel_requests as number | undefined) ??
              DEFAULT_MAX_PARALLEL_REQUESTS,
            onChange: (value: number) =>
              inferenceParamsField.onChange({
                ...(inferenceParamsField.value ?? EMPTY_INFERENCE_PARAMS),
                max_parallel_requests: Math.round(value),
              }),
          }}
          defaultValue={DEFAULT_MAX_PARALLEL_REQUESTS}
          min={MAX_PARALLEL_REQUESTS_MIN}
          max={MAX_PARALLEL_REQUESTS_MAX}
          step={1}
          size="compact"
          showReset
          formFieldProps={{
            slotLabel: 'Max parallel requests',
            slotInfo:
              'How many generation requests this model may have in flight at once. Lower it if your inference provider rate-limits the job.',
          }}
        />
      </Stack>

      <Flex align="center" justify="start" className="shrink-0 border-t border-base p-density-lg">
        <Button kind="tertiary" color="danger" size="small" onClick={onRemove}>
          <Trash2 size={16} aria-hidden />
          Remove model
        </Button>
      </Flex>
    </aside>
  );
};
