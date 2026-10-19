// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { ControlledSearchableSelect } from '@nemo/common/src/components/form/ControlledSearchableSelect';
import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getAgentsListAgentsQueryKey,
  useAgentsCreateAgent,
} from '@nemo/sdk/generated/agents/agents';
import type { CreateAgentRequestConfig } from '@nemo/sdk/generated/agents/schema/CreateAgentRequestConfig';
import { useModelsListModels } from '@nemo/sdk/generated/platform/models';
import { loadSampleAgentConfig } from '@studio/api/agents/loadSampleAgentConfig';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import {
  buildSampleAgentName,
  DEFAULT_SAMPLE_AGENT_KEY,
  getSampleAgent,
  isSampleAgentName,
  SAMPLE_AGENTS,
  sampleAgentFormSchema,
} from '@studio/constants/sampleAgents';
import {
  hasShownExampleAgentIntro,
  markAgentWalkthroughPending,
  markExampleAgentIntroShown,
} from '@studio/routes/agents/AgentDetailRoute/walkthroughStorage';
import type {
  CreateExampleAgentModalProps,
  ExampleAgentFormData,
} from '@studio/routes/agents/AgentsListRoute/CreateExampleAgentModal/type';
import { getAgentDetailRoute, getAgentsListRoute } from '@studio/routes/utils';
import {
  buildSuggestedModelOptions,
  pickModelNameForExample,
  SUGGESTED_MODEL_GROUP_LABELS,
} from '@studio/util/buildSuggestedModelOptions';
import { loadSampleAgentModelName } from '@studio/util/sampleAgents';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { type FC, useEffect, useMemo, useState } from 'react';
import { type SubmitHandler, useForm, useWatch } from 'react-hook-form';
import { useNavigate } from 'react-router';

const EXAMPLE_ITEMS = SAMPLE_AGENTS.map((example) => ({
  value: example.key,
  children: example.displayName,
}));

// Since useForm is called in the component itself, key-based remount needs a thin outer wrapper
// otherwise there's nothing to put the key on
export const CreateExampleAgentModal: FC<CreateExampleAgentModalProps> = (props) => (
  <CreateExampleAgentModalInner key={props.open ? props.workspace : 'closed'} {...props} />
);

const CreateExampleAgentModalInner: FC<CreateExampleAgentModalProps> = ({
  open,
  onClose,
  workspace,
  existingAgents,
}) => {
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const { data: modelsPage, isLoading: isLoadingModels } = useModelsListModels(
    workspace,
    { page_size: DEFAULT_LARGE_PAGE_SIZE },
    { query: { enabled: open && !!workspace } }
  );
  const models = useMemo(() => modelsPage?.data ?? [], [modelsPage?.data]);
  const modelOptions = useMemo(() => buildSuggestedModelOptions(models), [models]);

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
        const priorExampleAgentExists = existingAgents.some(
          (existing) =>
            !!existing.name && existing.name !== agent.name && isSampleAgentName(existing.name)
        );
        const onboard = !!agent.name && !hasShownExampleAgentIntro() && !priorExampleAgentExists;
        if (onboard && agent.name) {
          markExampleAgentIntroShown();
          markAgentWalkthroughPending(agent.name);
        }
        resetAndClose();
        navigate(
          onboard && agent.name
            ? getAgentDetailRoute(workspace, agent.name)
            : getAgentsListRoute(workspace)
        );
      },
    },
  });

  const {
    control,
    setValue,
    handleSubmit,
    formState: { errors },
  } = useForm({
    resolver: zodResolver(sampleAgentFormSchema),
    defaultValues: { exampleKey: DEFAULT_SAMPLE_AGENT_KEY, modelName: '' },
    disabled: isPending,
    mode: 'onChange',
  });

  const exampleKey = useWatch({ control, name: 'exampleKey' });

  const { data: preferredModel } = useQuery({
    queryKey: ['sample-agent-model', exampleKey],
    queryFn: () => loadSampleAgentModelName(getSampleAgent(exampleKey).agentConfigPath),
    enabled: open && !!exampleKey,
    staleTime: Infinity,
  });

  const defaultModel = useMemo(
    () => pickModelNameForExample(models, preferredModel),
    [models, preferredModel]
  );

  useEffect(() => {
    if (!defaultModel) return;
    setValue('modelName', defaultModel, { shouldValidate: true });
  }, [defaultModel, setValue]);

  const [loadError, setLoadError] = useState<string | undefined>(undefined);

  const reset = () => {
    resetMutation();
    setLoadError(undefined);
  };

  const resetAndClose = () => {
    reset();
    onClose();
  };

  const onSubmit: SubmitHandler<ExampleAgentFormData> = async (formData) => {
    const example = getSampleAgent(formData.exampleKey);
    setLoadError(undefined);
    let config: CreateAgentRequestConfig;
    try {
      config = await loadSampleAgentConfig(example.agentConfigPath, formData.modelName, workspace);
    } catch (err) {
      setLoadError(getErrorMessage(err as Error, 'Failed to load example agent config'));
      return;
    }
    try {
      await createAgent({
        workspace,
        data: {
          name: buildSampleAgentName(example.namePrefix),
          // The config is the single source of truth for the blurb; the registry no
          // longer carries a copy that could drift from the shipped agent.yml.
          description: typeof config.description === 'string' ? config.description : '',
          config,
          // Omitted for NAT samples (API defaults to nat-workflow-v1); set for
          // Fabric samples so the API validates the config as nemo-agents-spec-v1.
          ...(example.configFormat ? { config_format: example.configFormat } : {}),
        },
      });
    } catch {
      // surfaced via errorText
    }
  };

  const errorMessage =
    loadError ??
    (createError
      ? getErrorMessage(createError as Error, 'Failed to create example agent')
      : undefined);

  return (
    <FormModal
      open={open}
      onClose={resetAndClose}
      title="Create Example Agent"
      submitButtonText="Create"
      onSubmit={handleSubmit(onSubmit)}
      disabled={isPending}
      loading={isPending}
      errorText={errorMessage}
    >
      <ControlledSelect
        useControllerProps={{ control, name: 'exampleKey' }}
        items={EXAMPLE_ITEMS}
        renderValue={(v) => EXAMPLE_ITEMS.find((item) => item.value === v)?.children}
        formFieldProps={{
          slotLabel: 'Example',
          slotError: errors.exampleKey?.message,
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
