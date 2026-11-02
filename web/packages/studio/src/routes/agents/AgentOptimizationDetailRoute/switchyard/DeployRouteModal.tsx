// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { entityNameSchema } from '@nemo/common/src/utils/entityName';
import type { RunStrategyJob } from '@nemo/sdk/generated/agent-optimization/schema/RunStrategyJob';
import { getAgentsListAgentsQueryKey, useAgentsGetAgent } from '@nemo/sdk/generated/agents/agents';
import type { RegisterAgentRequestConfig } from '@nemo/sdk/generated/agents/schema/RegisterAgentRequestConfig';
import { Banner, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { AgentSpecFilesetOrphanError } from '@studio/api/agents/agentSpecFileset';
import { useCreateAgentFromTrial } from '@studio/api/agents/useCreateAgentFromTrial';
import {
  DEPLOY_ON_CREATE_FORM_DEFAULTS,
  type DeployOnCreatePlan,
  deployOnCreateFormSchema,
  useDeployOnCreate,
} from '@studio/api/agents/useDeployOnCreate';
import {
  routeSettingText,
  routingStrategyTitle,
} from '@studio/components/dataViews/SwitchyardRoutesDataView/useRouteColumns';
import { DeployOnCreateFields } from '@studio/components/DeployOnCreateFields';
import {
  fetchRouteAgentConfig,
  type SwitchyardResults,
  type SwitchyardRoute,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/switchyard/switchyardResults';
import { withAgentName } from '@studio/routes/agents/AgentOptimizationDetailRoute/trialAgentConfig';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { type FC, useEffect, useRef, useState } from 'react';
import { type SubmitHandler, useForm, useWatch } from 'react-hook-form';
import { z } from 'zod';

const deployRouteFormSchema = z
  .object({ name: entityNameSchema() })
  .merge(deployOnCreateFormSchema);

type DeployRouteFormData = z.infer<typeof deployRouteFormSchema>;

const DEPLOY_ROUTE_FORM_DEFAULTS: DeployRouteFormData = {
  name: '',
  ...DEPLOY_ON_CREATE_FORM_DEFAULTS,
};

/** `ws/name`, or a bare name resolved against the run's workspace, as the strategy does. */
const resolveAgentRef = (
  agentRef: string | undefined,
  workspace: string
): { workspace: string; name: string } | undefined => {
  if (!agentRef) return undefined;
  const [first, ...rest] = agentRef.split('/');
  return rest.length ? { workspace: first!, name: rest.join('/') } : { workspace, name: first! };
};

export interface DeployRouteModalProps {
  workspace: string;
  job: RunStrategyJob;
  results: SwitchyardResults;
  route: SwitchyardRoute | null;
  onClose: () => void;
}

export const DeployRouteModal: FC<DeployRouteModalProps> = ({
  workspace,
  job,
  results,
  route,
  onClose,
}) => {
  const open = route !== null;
  const toast = useToast();
  const queryClient = useQueryClient();
  const sourceRef = resolveAgentRef(results.agent || job.spec.agent, results.workspace);
  const [replaceArmedFor, setReplaceArmedFor] = useState<string | null>(null);
  // Set on submit, so the choice made for this agent is the one that runs once it exists.
  const deployAfterCreate = useRef<DeployOnCreatePlan | null>(null);
  const deployOnCreate = useDeployOnCreate(workspace, { enabled: open });

  const {
    data: sourceAgent,
    isLoading: isLoadingAgent,
    error: agentError,
  } = useAgentsGetAgent(sourceRef?.workspace ?? '', sourceRef?.name ?? '', {
    query: { enabled: open && !!sourceRef },
  });

  // The run writes each route's agent config once and never rewrites it.
  const configPath = route?.agentConfigPath;
  const {
    data: routeConfig,
    isLoading: isLoadingConfig,
    error: configError,
  } = useQuery({
    queryKey: ['switchyard-route-agent-config', results.location, configPath],
    queryFn: ({ signal }) => fetchRouteAgentConfig(results.location, configPath!, signal),
    enabled: open && !!configPath,
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
      toast.success(`Agent "${agent.name}" created on ${route?.virtualModel}`);
      const targetWorkspace = agent.workspace ?? workspace;
      void queryClient.invalidateQueries({
        queryKey: getAgentsListAgentsQueryKey(targetWorkspace),
      });
      if (agent.name) {
        void deployOnCreate.openCreatedAgent(
          targetWorkspace,
          agent.name,
          deployAfterCreate.current
        );
      }
    },
  });
  const busy = isPending || deployOnCreate.isDeploying;

  const {
    control,
    reset: resetForm,
    handleSubmit,
    formState: { errors, touchedFields, isSubmitted },
  } = useForm<DeployRouteFormData>({
    resolver: zodResolver(deployRouteFormSchema),
    defaultValues: DEPLOY_ROUTE_FORM_DEFAULTS,
    disabled: busy,
    mode: 'onTouched',
  });
  const watchedName = useWatch({ control, name: 'name' });
  const watchedDeploy = useWatch({ control, name: 'deploy' });
  const replaceOrphan = replaceArmedFor !== null && replaceArmedFor === watchedName?.trim();

  // The VirtualModel is already named `<agent>-<digest>`, unique per routing setup.
  useEffect(() => {
    resetForm({ ...DEPLOY_ROUTE_FORM_DEFAULTS, name: route?.virtualModel ?? '' });
    setReplaceArmedFor(null);
  }, [route, resetForm]);

  // The run read the agent's config when it ran, so a later edit is not in the route's config.
  const agentChangedSinceRun =
    !!sourceAgent?.updated_at &&
    !!job.created_at &&
    Date.parse(sourceAgent.updated_at) > Date.parse(job.created_at);

  const resetAndClose = () => {
    resetMutation();
    onClose();
  };

  const onSubmit: SubmitHandler<DeployRouteFormData> = async (formData) => {
    const { name } = formData;
    if (!sourceAgent || !sourceRef || !routeConfig) return;
    deployAfterCreate.current = deployOnCreate.plan(formData);
    try {
      await createAgent({
        workspace,
        name,
        source: {
          ...sourceRef,
          description: sourceAgent.description,
          config_format: sourceAgent.config_format,
        },
        config: withAgentName(routeConfig, name) as RegisterAgentRequestConfig,
        replaceOrphanedFileset: replaceOrphan,
      });
    } catch (error) {
      // An orphaned fileset is recoverable, so the next submit replaces it.
      setReplaceArmedFor(error instanceof AgentSpecFilesetOrphanError ? name : null);
    }
  };

  const getLoadError = (): string | undefined => {
    if (!sourceRef) return 'This run has no source agent, so its routes cannot be deployed.';
    if (route && !configPath)
      return 'This run did not save an agent config for this virtual model.';
    if (agentError) {
      return getErrorMessage(agentError as Error, `Could not load agent "${sourceRef.name}"`);
    }
    if (configError) return getErrorMessage(configError, 'Could not load the routed agent config');
    return undefined;
  };
  const loadError = getLoadError();
  const awaitingDeployModes = deployOnCreate.blocksSubmit(Boolean(watchedDeploy));
  const errorText =
    loadError ?? (createError ? getErrorMessage(createError, 'Failed to create agent') : undefined);

  const summary = route
    ? [
        { label: 'Virtual model', value: route.virtualModel },
        { label: 'Routing strategy', value: routingStrategyTitle(route.configType) },
        { label: 'Capable model', value: route.capable },
        { label: 'Efficient model', value: route.efficient },
        { label: 'Setting', value: routeSettingText(route) },
      ]
    : [];

  return (
    <FormModal
      open={open}
      onClose={resetAndClose}
      title="Deploy route as a new agent"
      instruction={
        route && sourceRef
          ? `Create a new agent from "${sourceRef.name}" whose models all call the "${route.virtualModel}" virtual model, in workspace "${workspace}". The original agent is not changed.`
          : undefined
      }
      submitButtonText={replaceOrphan ? 'Replace and deploy' : 'Deploy'}
      onSubmit={handleSubmit(onSubmit)}
      disabled={busy}
      loading={busy || isLoadingAgent || isLoadingConfig}
      submitDisabled={!!loadError || !routeConfig || awaitingDeployModes}
      errorText={errorText}
    >
      {routeConfig && agentChangedSinceRun && (
        <Banner kind="inline" status="warning">
          &quot;{sourceRef?.name}&quot; was changed after this run. The new agent uses the
          configuration the run captured, without the later changes.
        </Banner>
      )}
      <ControlledTextInput
        useControllerProps={{ control, name: 'name' }}
        label="New agent name"
        formFieldProps={{
          slotError: touchedFields.name || isSubmitted ? errors.name?.message : undefined,
        }}
      />
      {route && (
        <Stack gap="2">
          <Text kind="label/semibold/md">Routing</Text>
          <Stack gap="1" data-testid="deploy-route-summary">
            {summary.map(({ label, value }) => (
              <Flex key={label} justify="between" align="center" gap="4">
                <Text kind="body/regular/sm" className="text-secondary">
                  {label}
                </Text>
                <Text kind="body/regular/sm" className="truncate tabular-nums" title={value}>
                  {value}
                </Text>
              </Flex>
            ))}
          </Stack>
        </Stack>
      )}
      <DeployOnCreateFields
        deployOnCreate={deployOnCreate}
        deployControllerProps={{ control, name: 'deploy' }}
        modeControllerProps={{ control, name: 'deploymentMode' }}
      />
    </FormModal>
  );
};
