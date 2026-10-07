// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import type { RunStrategyJob } from '@nemo/sdk/generated/agent-optimization/schema/RunStrategyJob';
import { getAgentsListAgentsQueryKey, useAgentsGetAgent } from '@nemo/sdk/generated/agents/agents';
import type { CreateAgentRequestConfig } from '@nemo/sdk/generated/agents/schema/CreateAgentRequestConfig';
import { Banner, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { AgentSpecFilesetOrphanError } from '@studio/api/agents/agentSpecFileset';
import { useCreateAgentFromTrial } from '@studio/api/agents/useCreateAgentFromTrial';
import type { Trial } from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';
import {
  type AppliedTrialConfig,
  applyTrialToAgentConfig,
  buildTrialAgentName,
  fetchStudyConfig,
  studyConfigLocation,
  withAgentName,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/trialAgentConfig';
import { agentNameSchema } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { type FC, useEffect, useMemo, useState } from 'react';
import { type SubmitHandler, useForm, useWatch } from 'react-hook-form';
import { useNavigate } from 'react-router';
import { z } from 'zod';

const deployTrialFormSchema = z.object({ name: agentNameSchema });

type DeployTrialFormData = z.infer<typeof deployTrialFormSchema>;

// `ws/name`, or a bare name resolved against the study's workspace, as the optimize job does.
const resolveAgentRef = (
  agentRef: string | undefined,
  workspace: string
): { workspace: string; name: string } | undefined => {
  if (!agentRef) return undefined;
  const [first, ...rest] = agentRef.split('/');
  return rest.length ? { workspace: first!, name: rest.join('/') } : { workspace, name: first! };
};

type ApplyResult = ({ ok: true } & AppliedTrialConfig) | { ok: false; error: string };

export interface DeployTrialModalProps {
  workspace: string;
  job: RunStrategyJob;
  trial: Trial | null;
  onClose: () => void;
}

export const DeployTrialModal: FC<DeployTrialModalProps> = ({ workspace, job, trial, onClose }) => {
  const open = trial !== null;
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { spec } = job;
  const studyWorkspace = spec.workspace ?? workspace;
  const sourceRef = resolveAgentRef(spec.agent, studyWorkspace);
  const configLocation = useMemo(
    () => studyConfigLocation(spec, studyWorkspace),
    [spec, studyWorkspace]
  );
  const [replaceArmedFor, setReplaceArmedFor] = useState<string | null>(null);

  const {
    data: sourceAgent,
    isLoading: isLoadingAgent,
    error: agentError,
  } = useAgentsGetAgent(sourceRef?.workspace ?? '', sourceRef?.name ?? '', {
    query: { enabled: open && !!sourceRef },
  });

  // A study's bundle is staged once and never rewritten, so its config is fetched once.
  const {
    data: studyConfig,
    isLoading: isLoadingStudyConfig,
    error: studyConfigError,
  } = useQuery({
    queryKey: ['optimize-study-config', configLocation],
    queryFn: ({ signal }) => fetchStudyConfig(configLocation!, signal),
    enabled: open && !!configLocation,
    staleTime: Infinity,
    retry: false,
  });

  const {
    mutateAsync: createAgent,
    error: createError,
    isPending,
    reset: resetMutation,
  } = useCreateAgentFromTrial({
    onSuccess: (agent) => {
      toast.success(`Agent "${agent.name}" created from trial ${trial?.number}`);
      void queryClient.invalidateQueries({ queryKey: getAgentsListAgentsQueryKey(workspace) });
      if (agent.name) navigate(getAgentDetailRoute(workspace, agent.name));
    },
  });

  const {
    control,
    reset: resetForm,
    handleSubmit,
    formState: { errors },
  } = useForm<DeployTrialFormData>({
    resolver: zodResolver(deployTrialFormSchema),
    defaultValues: { name: '' },
    disabled: isPending,
    mode: 'onChange',
  });
  const watchedName = useWatch({ control, name: 'name' });
  const replaceOrphan = replaceArmedFor !== null && replaceArmedFor === watchedName?.trim();

  const sourceName = sourceRef?.name;
  useEffect(() => {
    resetForm({ name: trial && sourceName ? buildTrialAgentName(sourceName, trial) : '' });
    setReplaceArmedFor(null);
  }, [trial, sourceName, resetForm]);

  const applied = useMemo((): ApplyResult | undefined => {
    if (!trial || !sourceAgent || !studyConfig) return undefined;
    try {
      return {
        ok: true,
        ...applyTrialToAgentConfig(sourceAgent.config ?? {}, trial, studyConfig),
      };
    } catch (error) {
      return {
        ok: false,
        error: getErrorMessage(error as Error, 'Failed to apply the trial configuration'),
      };
    }
  }, [trial, sourceAgent, studyConfig]);
  const appliedConfig = applied?.ok ? applied : undefined;

  // The study read the agent's config when it ran, so a later edit was never evaluated.
  const agentChangedSinceStudy =
    !!sourceAgent?.updated_at &&
    !!job.created_at &&
    Date.parse(sourceAgent.updated_at) > Date.parse(job.created_at);

  const resetAndClose = () => {
    resetMutation();
    onClose();
  };

  const onSubmit: SubmitHandler<DeployTrialFormData> = async ({ name }) => {
    if (!sourceAgent || !sourceRef || !appliedConfig) return;
    try {
      await createAgent({
        workspace,
        name,
        source: {
          ...sourceRef,
          description: sourceAgent.description,
          config_format: sourceAgent.config_format,
        },
        config: withAgentName(appliedConfig.config, name) as CreateAgentRequestConfig,
        replaceOrphanedFileset: replaceOrphan,
      });
    } catch (error) {
      // An orphaned fileset is recoverable, so the next submit replaces it.
      setReplaceArmedFor(error instanceof AgentSpecFilesetOrphanError ? name : null);
    }
  };

  const loadError = !sourceRef
    ? 'This study has no agent under test, so its trials cannot be deployed.'
    : !configLocation
      ? 'This study has no optimize config fileset, so its trials cannot be deployed.'
      : agentError
        ? getErrorMessage(agentError as Error, `Could not load agent "${sourceRef.name}"`)
        : studyConfigError
          ? getErrorMessage(studyConfigError, 'Could not load the study optimize config')
          : undefined;
  const errorText =
    loadError ??
    (applied?.ok === false ? applied.error : undefined) ??
    (createError ? getErrorMessage(createError, 'Failed to create agent') : undefined);
  const isLoading = isLoadingAgent || isLoadingStudyConfig;

  return (
    <FormModal
      open={open}
      onClose={resetAndClose}
      title="Deploy trial as a new agent"
      instruction={
        trial && sourceRef
          ? `Create a new agent from "${sourceRef.name}" using the configuration from trial ${trial.number}. The original agent is not changed.`
          : undefined
      }
      submitButtonText={replaceOrphan ? 'Replace and deploy' : 'Deploy'}
      onSubmit={handleSubmit(onSubmit)}
      disabled={isPending}
      loading={isPending || isLoading}
      submitDisabled={!!loadError || !appliedConfig}
      errorText={errorText}
    >
      {appliedConfig && agentChangedSinceStudy && (
        <Banner kind="inline" status="warning">
          &quot;{sourceRef?.name}&quot; was changed after this study ran. The trial&apos;s values
          are applied to its current configuration, which the study did not evaluate.
        </Banner>
      )}
      {appliedConfig?.modelDifferences.map(({ modelKey, fields, studyModel, agentModel }) => (
        <Banner key={modelKey} kind="inline" status="warning">
          The study ran the &quot;{modelKey}&quot; model with different settings than this agent (
          {fields.join(', ')}).{' '}
          {studyModel !== undefined && (
            <>
              It ran <code>{studyModel}</code>; this agent uses <code>{agentModel}</code>.{' '}
            </>
          )}
          The new agent keeps this agent&apos;s settings, with the trial&apos;s values applied.
        </Banner>
      ))}
      <ControlledTextInput
        useControllerProps={{ control, name: 'name' }}
        label="New agent name"
        formFieldProps={{ slotError: errors.name?.message }}
      />
      {trial && trial.params.length > 0 && (
        <Stack gap="2">
          <Text kind="label/semibold/md">Configuration from trial {trial.number}</Text>
          <Stack gap="1" data-testid="deploy-trial-params">
            {trial.params.map((param) => (
              <Flex key={param.name} justify="between" align="center" gap="4">
                <Text kind="body/regular/sm" className="text-secondary">
                  {studyConfig?.searchSpace.get(param.name)?.path ?? param.name}
                </Text>
                <Text kind="body/regular/sm" className="tabular-nums">
                  {param.value}
                </Text>
              </Flex>
            ))}
          </Stack>
        </Stack>
      )}
      {!!appliedConfig?.addedModels.length && (
        <Text kind="body/regular/sm" className="text-secondary">
          Also adds {appliedConfig.addedModels.map((key) => `"${key}"`).join(', ')}{' '}
          {appliedConfig.addedModels.length === 1 ? 'model' : 'models'} from the study&apos;s
          optimize config, which the agent ran with during the study.
        </Text>
      )}
    </FormModal>
  );
};
