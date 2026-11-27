// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelWorkspaceGroup } from '@nemo/common/src/api/models/modelPages';
import { useModelSearch } from '@nemo/common/src/api/models/useModelSearch';
import { ControlledSearchableSelect } from '@nemo/common/src/components/form/ControlledSearchableSelect';
import { ModelSelect } from '@nemo/common/src/components/ModelSelectV2/ModelSelect';
import type { ModelSelection } from '@nemo/common/src/components/ModelSelectV2/types';
import { toInferenceModelEntityId } from '@nemo/common/src/utils/models';
import type { InferenceParams, ModelEntity } from '@nemo/sdk/generated/platform/schema';
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
import { useCallback, useMemo, useState, type FC } from 'react';
import { useController, useFormContext, useWatch } from 'react-hook-form';

const ADAPTERS_INFIX = '&adapters/';

const toModelSelection = (modelId: string): ModelSelection | null => {
  if (!modelId) {
    return null;
  }
  const infixAt = modelId.indexOf(ADAPTERS_INFIX);
  if (infixAt === -1) {
    return { model: modelId };
  }
  const adapterRef = modelId.slice(infixAt + ADAPTERS_INFIX.length);
  return {
    model: modelId.slice(0, infixAt),
    adapter: adapterRef.slice(adapterRef.lastIndexOf('/') + 1),
  };
};

const withServedAdapters = (
  groups: ModelWorkspaceGroup[],
  servedModelIds: ReadonlySet<string>
): ModelWorkspaceGroup[] =>
  groups.map((group) => ({
    ...group,
    models: group.models.map((model) => ({
      ...model,
      adapters: model.adapters?.filter((adapter) =>
        servedModelIds.has(toInferenceModelEntityId(model, adapter))
      ),
    })),
  }));

interface RoleModelSelectProps {
  role: string;
  servedModelIds: ReadonlySet<string>;
  providersLoading: boolean;
  applyModel: (role: string, id: string, preferredProviders?: readonly string[]) => void;
}

const RoleModelSelect: FC<RoleModelSelectProps> = ({
  role,
  servedModelIds,
  providersLoading,
  applyModel,
}) => {
  const { control, setValue } = useFormContext<AnonymizerFormData>();
  const workspace = useWorkspaceFromPath();
  const { field, fieldState } = useController({ control, name: `roleModels.${role}.modelId` });
  const params = useWatch({ control, name: `roleModels.${role}.params` });
  const [open, setOpen] = useState(false);
  const value = useMemo(() => toModelSelection(field.value), [field.value]);
  const error = fieldState.error?.message;

  const isServed = useCallback(
    (model: ModelEntity) => servedModelIds.has(toInferenceModelEntityId(model)),
    [servedModelIds]
  );
  const modelSearch = useModelSearch({
    workspace,
    include: isServed,
    enabled: open && !providersLoading,
  });
  const groups = useMemo(
    () => withServedAdapters(modelSearch.groups, servedModelIds),
    [modelSearch.groups, servedModelIds]
  );

  const handleValueChange = (selection: ModelSelection) => {
    const adapter = selection.entity?.adapters?.find((entry) => entry.name === selection.adapter);
    let modelId = selection.model;
    if (selection.entity) {
      modelId = toInferenceModelEntityId(selection.entity, adapter);
    }
    field.onChange(modelId);
    applyModel(role, modelId, modelSearch.modelFilters.values.providers);
  };

  return (
    <FormField slotLabel="Model" required slotError={error} status={error ? 'error' : undefined}>
      <ModelSelect
        value={value}
        onValueChange={handleValueChange}
        groups={groups}
        loading={modelSearch.loading || providersLoading}
        error={modelSearch.error}
        onSearchChange={modelSearch.onSearchChange}
        onLoadMore={modelSearch.onLoadMore}
        hasMore={modelSearch.hasMore}
        isLoadingMore={modelSearch.isLoadingMore}
        modelFilters={modelSearch.modelFilters}
        onOpenChange={setOpen}
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

// The entity detector keeps a plain select: its built-in GLiNER2 option is not a model.
export const ModelSettingsSection: FC = () => {
  const { control } = useFormContext<AnonymizerFormData>();
  const strategy = useWatch({ control, name: 'strategy' });

  const roles = useMemo(() => activeRolesForStrategy(strategy), [strategy]);
  const { models, items, isLoading, applyModel } = useAnonymizerModels();

  const detectorItems = useMemo(() => detectorOptions(items, models), [items, models]);
  const servedModelIds = useMemo(() => new Set(models.map((model) => model.id)), [models]);

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
            <RoleModelSelect
              role={role}
              servedModelIds={servedModelIds}
              providersLoading={isLoading}
              applyModel={applyModel}
            />
          )}
        </Stack>
      ))}
    </Stack>
  );
};
