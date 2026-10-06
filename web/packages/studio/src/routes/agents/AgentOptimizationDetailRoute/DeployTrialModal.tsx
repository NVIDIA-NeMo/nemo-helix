// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import type { RunStrategySpec } from '@nemo/sdk/generated/agent-optimization/schema/RunStrategySpec';
import {
  getAgentsListAgentsQueryKey,
  useAgentsCreateAgent,
  useAgentsGetAgent,
} from '@nemo/sdk/generated/agents/agents';
import type { CreateAgentRequestConfig } from '@nemo/sdk/generated/agents/schema/CreateAgentRequestConfig';
import { Badge, Banner, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { Trial } from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';
import {
  type AppliedTrialConfig,
  applyTrialToAgentConfig,
  buildTrialAgentName,
  fetchStudyConfig,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/trialAgentConfig';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { type FC, useEffect, useMemo } from 'react';
import { type SubmitHandler, useForm } from 'react-hook-form';
import { useNavigate } from 'react-router';
import { z } from 'zod';

const deployTrialFormSchema = z.object({
  name: z.string().trim().min(1, 'Name is required'),
});

type DeployTrialFormData = z.infer<typeof deployTrialFormSchema>;

/** `ws/name` or a bare name resolved against the job's workspace, as the optimize job does. */
const resolveAgentRef = (
  agentRef: string | undefined,
  jobWorkspace: string
): { workspace: string; name: string } | undefined => {
  if (!agentRef) return undefined;
  const [first, ...rest] = agentRef.split('/');
  return rest.length
    ? { workspace: first, name: rest.join('/') }
    : { workspace: jobWorkspace, name: first };
};

type ApplyResult = ({ ok: true } & AppliedTrialConfig) | { ok: false; error: string };

export interface DeployTrialModalProps {
  workspace: string;
  spec: RunStrategySpec | undefined;
  trial: Trial | null;
  onClose: () => void;
}

export const DeployTrialModal: FC<DeployTrialModalProps> = ({
  workspace,
  spec,
  trial,
  onClose,
}) => {
  const open = trial !== null;
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const sourceRef = resolveAgentRef(spec?.agent, workspace);

  const {
    data: sourceAgent,
    isLoading: isLoadingAgent,
    error: agentError,
  } = useAgentsGetAgent(sourceRef?.workspace ?? '', sourceRef?.name ?? '', {
    query: { enabled: open && !!sourceRef },
  });

  const {
    data: studyConfig,
    isLoading: isLoadingStudyConfig,
    error: studyConfigError,
  } = useQuery({
    queryKey: ['optimize-study-config', spec?.optimize_config_fileset, spec?.optimize_config],
    queryFn: ({ signal }) => fetchStudyConfig(spec, signal),
    enabled: open,
    retry: false,
  });

  const {
    mutateAsync: createAgent,
    error: createError,
    isPending,
    reset: resetMutation,
  } = useAgentsCreateAgent({
    mutation: {
      onSuccess: (agent) => {
        toast.success(`Agent "${agent.name}" created from trial ${trial?.number}`);
        void queryClient.invalidateQueries({ queryKey: getAgentsListAgentsQueryKey(workspace) });
        if (agent.name) navigate(getAgentDetailRoute(workspace, agent.name));
      },
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

  useEffect(() => {
    resetForm({ name: trial && sourceRef ? buildTrialAgentName(sourceRef.name, trial) : '' });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trial, sourceRef?.name, resetForm]);

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

  const resetAndClose = () => {
    resetMutation();
    onClose();
  };

  const onSubmit: SubmitHandler<DeployTrialFormData> = async ({ name }) => {
    if (!sourceAgent || !appliedConfig) return;
    const config = { ...appliedConfig.config };
    if (typeof config.name === 'string') config.name = name.trim();
    try {
      await createAgent({
        workspace,
        data: {
          name: name.trim(),
          description: sourceAgent.description,
          config: config as CreateAgentRequestConfig,
          config_format: sourceAgent.config_format,
        },
      });
    } catch {
      // surfaced via errorText
    }
  };

  const loadError = !sourceRef
    ? 'This study has no agent under test, so its trials cannot be deployed.'
    : agentError
      ? getErrorMessage(agentError as Error, `Could not load agent "${sourceRef.name}"`)
      : studyConfigError
        ? getErrorMessage(studyConfigError, 'Could not load the study optimize config')
        : undefined;
  const errorText =
    loadError ??
    (applied?.ok === false ? applied.error : undefined) ??
    (createError ? getErrorMessage(createError as Error, 'Failed to create agent') : undefined);
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
      submitButtonText="Deploy"
      onSubmit={handleSubmit(onSubmit)}
      disabled={isPending}
      loading={isPending || isLoading}
      submitDisabled={!!loadError || !appliedConfig}
      errorText={errorText}
    >
      {appliedConfig?.modelMismatches.map(({ modelKey, studyModel, agentModel }) => (
        <Banner key={modelKey} kind="inline" status="warning">
          The study ran the &quot;{modelKey}&quot; model as <code>{studyModel}</code>, but this
          agent uses <code>{agentModel}</code>. The trial&apos;s values for it will be applied to{' '}
          <code>{agentModel}</code>.
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
            {trial.params.map((param) => {
              const isSkipped = appliedConfig?.skipped.includes(param.name) ?? false;
              return (
                <Flex key={param.name} justify="between" align="center" gap="4">
                  <Text kind="body/regular/sm" className="text-secondary">
                    {studyConfig?.searchSpace[param.name]?.path ?? param.name}
                  </Text>
                  <Flex align="center" gap="2">
                    {isSkipped && (
                      <Badge kind="outline" color="gray">
                        Not applied
                      </Badge>
                    )}
                    <Text
                      kind="body/regular/sm"
                      className={isSkipped ? 'tabular-nums text-placeholder' : 'tabular-nums'}
                    >
                      {param.value}
                    </Text>
                  </Flex>
                </Flex>
              );
            })}
          </Stack>
          {!!appliedConfig?.skipped.length && (
            <Text kind="body/regular/sm" className="text-secondary">
              Parameters marked Not applied tune models defined only by the study&apos;s optimize
              config (such as an evaluation judge), so they are not part of the new agent.
            </Text>
          )}
        </Stack>
      )}
    </FormModal>
  );
};
