// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
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
import { Accordion, Stack } from '@nvidia/foundations-react-core';
import { AGENT_CONTAINER_DEPLOYMENTS_ENABLED } from '@studio/constants/environment';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useEffect, useRef, useState } from 'react';
import { type SubmitHandler, useForm } from 'react-hook-form';
import { z } from 'zod';

// Whether a container deployment needs an image depends on the server's configured
// default, which only the server knows. It rejects the request; we surface that.
const deploymentFormSchema = z.object({
  name: z.literal('').or(entityNameSchema('Deployment name')).optional(),
  agent: z.string().min(1, 'Agent is required'),
  deploymentMode: z.enum(['subprocess', 'docker', 'k8s']),
  image: z.string().optional(),
});

type DeploymentFormData = z.infer<typeof deploymentFormSchema>;

const makeDefaultValues = (agent?: string, image?: string): DeploymentFormData => ({
  name: '',
  agent: agent ?? '',
  deploymentMode: image && AGENT_CONTAINER_DEPLOYMENTS_ENABLED ? 'docker' : 'subprocess',
  image: image ?? '',
});

interface CreateDeploymentModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  /** When provided, pre-selects this agent and hides the agent dropdown. */
  agent?: string;
  /** Override the workspace inferred from the current path. */
  workspace: string;
  /** A freshly built tag to deploy, so the image does not have to be retyped. */
  initialImage?: string;
}

export const CreateDeploymentModal: FC<CreateDeploymentModalProps> = ({
  open,
  onClose,
  agent: agentProp,
  workspace,
  initialImage,
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
        // Trim before testing, so a whitespace-only entry omits the field rather
        // than sending an empty one.
        ...(data.deploymentMode !== 'subprocess' && data.image?.trim()
          ? { image: data.image.trim() }
          : {}),
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
    defaultValues: makeDefaultValues(agentProp, initialImage),
    disabled: isPending,
    mode: 'onChange',
  });
  const deploymentMode = watch('deploymentMode');

  // Opened when a packaged tag is prefilled, so it is not hidden behind a
  // disclosure the user never opened.
  const [advancedOpen, setAdvancedOpen] = useState<string | undefined>(
    initialImage ? 'advanced' : undefined
  );

  // Seeded on the open transition only. A packaging job can finish while this
  // dialog is open, and reseeding then would wipe what the user has typed.
  const wasOpen = useRef(open);
  useEffect(() => {
    if (open && !wasOpen.current) {
      resetForm(makeDefaultValues(agentProp, initialImage));
      setAdvancedOpen(initialImage ? 'advanced' : undefined);
    }
    wasOpen.current = open;
  }, [open, agentProp, initialImage, resetForm]);

  const reset = () => {
    resetMutation();
    resetForm(makeDefaultValues(agentProp, initialImage));
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

  // The server owns whether an image is required, so its message has to reach the user.
  const errorMessage = createError ? getErrorMessage(createError, 'An error occurred') : undefined;

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
        {/* Subprocess covers the common case and needs nothing else. Runtime and image
            are the container path, so they sit behind a disclosure rather than in front
            of everyone — and the whole section is absent when the platform refuses
            container deployments, since there would be nothing advanced to choose. */}
        {AGENT_CONTAINER_DEPLOYMENTS_ENABLED && (
          <Accordion
            className="[&>div]:border-b-0"
            value={advancedOpen}
            onValueChange={setAdvancedOpen}
            items={[
              {
                value: 'advanced',
                chevronPosition: 'start',
                slotTrigger: `${advancedOpen === 'advanced' ? 'Hide' : 'Show'} Advanced`,
                slotContent: (
                  <Stack gap="density-lg" className="pt-density-md">
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
                          slotInfo:
                            'The backend pulls this image using its configured registry credentials. Leave empty to use the deployment default, if one is configured.',
                        }}
                      />
                    )}
                  </Stack>
                ),
              },
            ]}
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
