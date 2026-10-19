// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { ControlledSearchableSelect } from '@nemo/common/src/components/form/ControlledSearchableSelect';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getAgentsListAgentsQueryKey,
  useAgentsCreateAgent,
} from '@nemo/sdk/generated/agents/agents';
import { useModelsListModels } from '@nemo/sdk/generated/platform/models';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import {
  applyModelToConfig,
  buildClonedAgentName,
  cloneAgentFormSchema,
  getPrimaryModelName,
} from '@studio/routes/agents/AgentsListRoute/CloneAgentModal/const';
import type {
  CloneAgentFormData,
  CloneAgentModalProps,
} from '@studio/routes/agents/AgentsListRoute/CloneAgentModal/type';
import { getAgentDetailRoute } from '@studio/routes/utils';
import {
  buildSuggestedModelOptions,
  pickDefaultModelName,
  SUGGESTED_MODEL_GROUP_LABELS,
} from '@studio/util/buildSuggestedModelOptions';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useEffect, useRef } from 'react';
import { type SubmitHandler, useForm } from 'react-hook-form';
import { useNavigate } from 'react-router';

export const CloneAgentModal: FC<CloneAgentModalProps> = ({
  open,
  onClose,
  workspace,
  sourceAgent,
}) => {
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const { data: modelsPage, isLoading: isLoadingModels } = useModelsListModels(
    workspace,
    { page_size: DEFAULT_LARGE_PAGE_SIZE },
    { query: { enabled: open && !!workspace } }
  );
  const models = modelsPage?.data ?? [];
  const modelOptions = buildSuggestedModelOptions(models);

  const {
    mutateAsync: createAgent,
    error: createError,
    isPending,
    reset: resetMutation,
  } = useAgentsCreateAgent({
    mutation: {
      onSuccess: (agent) => {
        toast.success(`Agent "${agent.name}" created`);
        void queryClient.invalidateQueries({ queryKey: getAgentsListAgentsQueryKey(workspace) });
        resetAndClose();
        if (agent.name) navigate(getAgentDetailRoute(workspace, agent.name));
      },
    },
  });

  const {
    control,
    reset: resetForm,
    setValue,
    handleSubmit,
    formState: { errors },
  } = useForm({
    resolver: zodResolver(cloneAgentFormSchema),
    defaultValues: { name: '', modelName: '' },
    disabled: isPending,
    mode: 'onChange',
  });

  const seededRef = useRef(false);
  useEffect(() => {
    if (!open) {
      seededRef.current = false;
      resetForm({ name: '', modelName: '' });
      return;
    }
    if (seededRef.current) return;
    const optionValues = new Set(modelOptions.map((o) => o.value));
    const currentModel = getPrimaryModelName(sourceAgent?.config);
    const seededModel =
      currentModel && optionValues.has(currentModel)
        ? currentModel
        : (pickDefaultModelName(models) ?? '');
    if (seededModel) {
      // Seed only the model so a name the user typed before models loaded isn't wiped.
      setValue('modelName', seededModel, { shouldValidate: true });
      seededRef.current = true;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, modelsPage, sourceAgent, setValue]);

  const reset = () => {
    resetMutation();
    resetForm({ name: '', modelName: '' });
  };

  const resetAndClose = () => {
    reset();
    onClose();
  };

  const onSubmit: SubmitHandler<CloneAgentFormData> = async (formData) => {
    if (!sourceAgent) return;
    const trimmed = formData.name.trim();
    const name = trimmed || buildClonedAgentName(sourceAgent.name);
    try {
      await createAgent({
        workspace,
        data: {
          name,
          description: sourceAgent.description,
          config: applyModelToConfig(sourceAgent.config, formData.modelName),
          config_format: sourceAgent.config_format,
        },
      });
    } catch {
      // surfaced via errorText
    }
  };

  const errorMessage = createError
    ? getErrorMessage(createError as Error, 'Failed to clone agent')
    : undefined;

  return (
    <FormModal
      open={open}
      onClose={resetAndClose}
      title="Clone Agent"
      instruction={
        sourceAgent
          ? `Create a copy of "${sourceAgent.name}" with a new name and model.`
          : undefined
      }
      submitButtonText="Clone"
      onSubmit={handleSubmit(onSubmit)}
      disabled={isPending}
      loading={isPending}
      errorText={errorMessage}
    >
      <ControlledTextInput
        useControllerProps={{ control, name: 'name' }}
        label="Name"
        placeholder={
          sourceAgent ? `${sourceAgent.name}-… (leave blank to auto-generate)` : undefined
        }
        formFieldProps={{
          slotError: errors.name?.message,
        }}
      />
      <ControlledSearchableSelect
        useControllerProps={{ control, name: 'modelName' }}
        options={modelOptions}
        groupLabels={SUGGESTED_MODEL_GROUP_LABELS}
        isLoading={isLoadingModels}
        triggerPlaceholder="Select a model"
        searchPlaceholder="Search models..."
        emptyMessage={
          isLoadingModels ? 'Loading models...' : 'No usable chat model in this workspace.'
        }
        formFieldProps={{
          slotLabel: 'Model',
          slotError: errors.modelName?.message,
        }}
      />
    </FormModal>
  );
};
