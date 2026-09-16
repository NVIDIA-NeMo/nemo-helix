// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal, type FormModalProps } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { ENTITY_NAME_HELP, entityNameSchema } from '@nemo/common/src/utils/entityName';
import {
  getAgentsListDeploymentsQueryKey,
  useAgentsCreateDeployment,
} from '@nemo/sdk/generated/agents/agent-deployments';
import { useAgentsListAgents } from '@nemo/sdk/generated/agents/agents';
import { Stack } from '@nvidia/foundations-react-core';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useEffect } from 'react';
import { type SubmitHandler, useForm } from 'react-hook-form';
import { z } from 'zod';

const deploymentFormSchema = z
  .object({
    name: z.literal('').or(entityNameSchema('Deployment name')).optional(),
    agent: z.string().min(1, 'Agent is required'),
    deploymentMode: z.enum(['subprocess', 'docker', 'k8s']),
    image: z.string().optional(),
  })
  .superRefine(({ deploymentMode, image }, context) => {
    if (deploymentMode !== 'subprocess' && !image?.trim()) {
      context.addIssue({
        code: 'custom',
        path: ['image'],
        message: 'Container image is required for Docker and Kubernetes deployments',
      });
    }
  });

type DeploymentFormData = z.infer<typeof deploymentFormSchema>;

const makeDefaultValues = (agent?: string): DeploymentFormData => ({
  name: '',
  agent: agent ?? '',
  deploymentMode: 'subprocess',
  image: '',
});

interface CreateDeploymentModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  /** When provided, pre-selects this agent and hides the agent dropdown. */
  agent?: string;
  /** Override the workspace inferred from the current path. */
  workspace: string;
}

export const CreateDeploymentModal: FC<CreateDeploymentModalProps> = ({
  open,
  onClose,
  agent: agentProp,
  workspace,
}) => {
  const toast = useToast();
  const queryClient = useQueryClient();

  const { data: agentsResponse, isLoading: isAgentsLoading } = useAgentsListAgents(
    workspace,
    undefined,
    { query: { enabled: open && !agentProp } }
  );
  const agents = agentsResponse?.data ?? [];

  const {
    mutateAsync: createDeploymentMutation,
    error: createError,
    isPending,
    reset: resetMutation,
  } = useAgentsCreateDeployment({
    mutation: {
      onSuccess: () => {
        toast.success('Deployment started successfully');
        void queryClient.invalidateQueries({
          queryKey: getAgentsListDeploymentsQueryKey(workspace),
        });
        resetAndClose();
      },
    },
  });

  const createDeployment = (data: DeploymentFormData) =>
    createDeploymentMutation({
      workspace,
      data: {
        agent: data.agent,
        ...(data.name ? { name: data.name } : {}),
        deployment_mode: data.deploymentMode,
        ...(data.deploymentMode !== 'subprocess' && data.image ? { image: data.image.trim() } : {}),
      },
    });

  const {
    control,
    reset: resetForm,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm({
    resolver: zodResolver(deploymentFormSchema),
    defaultValues: makeDefaultValues(agentProp),
    disabled: isPending,
    mode: 'onChange',
  });
  const deploymentMode = watch('deploymentMode');

  useEffect(() => {
    resetForm(makeDefaultValues(agentProp));
  }, [agentProp, resetForm]);

  const reset = () => {
    resetMutation();
    resetForm(makeDefaultValues(agentProp));
  };

  const resetAndClose = () => {
    reset();
    onClose();
  };

  const onSubmit: SubmitHandler<DeploymentFormData> = async (formData) => {
    try {
      await createDeployment(formData);
    } catch {
      // Error displayed via errorText prop
    }
  };

  const errorMessage =
    createError instanceof Error
      ? createError.message
      : createError
        ? 'An error occurred'
        : undefined;

  return (
    <FormModal
      open={open}
      onClose={resetAndClose}
      title="Deploy Agent"
      submitButtonText="Deploy"
      onSubmit={handleSubmit(onSubmit)}
      disabled={isPending}
      loading={isPending}
      errorText={errorMessage}
    >
      <Stack gap="density-xl">
        <ControlledTextInput
          useControllerProps={{ control, name: 'name' }}
          name="name"
          label="Deployment Name (optional)"
          formFieldProps={{
            slotError: errors.name?.message,
            slotInfo: ENTITY_NAME_HELP,
          }}
        />
        <ControlledSelect
          useControllerProps={{ control, name: 'deploymentMode' }}
          items={[
            { value: 'subprocess', children: 'Subprocess' },
            { value: 'docker', children: 'Docker' },
            { value: 'k8s', children: 'Kubernetes' },
          ]}
          formFieldProps={{
            slotLabel: 'Runtime',
            slotInfo:
              'Use Docker for a local container image or Kubernetes for a cluster deployment.',
          }}
        />
        {deploymentMode !== 'subprocess' && (
          <ControlledTextInput
            useControllerProps={{ control, name: 'image' }}
            name="image"
            label="Container Image"
            placeholder="nvcr.io/org/team/agent:tag"
            formFieldProps={{
              slotError: errors.image?.message,
              slotInfo: 'The backend pulls this image using its configured registry credentials.',
            }}
          />
        )}
        {!agentProp && (
          <ControlledSelect
            useControllerProps={{ control, name: 'agent' }}
            loading={isAgentsLoading}
            items={agents.flatMap((agent) =>
              agent.name ? [{ value: agent.name, children: agent.name }] : []
            )}
            formFieldProps={{
              slotLabel: 'Agent',
              slotError: errors.agent?.message,
            }}
          />
        )}
      </Stack>
    </FormModal>
  );
};
