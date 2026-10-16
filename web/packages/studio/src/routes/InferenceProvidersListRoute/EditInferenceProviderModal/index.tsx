/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { zodResolver } from '@hookform/resolvers/zod';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal, FormModalProps } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getModelsGetProviderQueryKey,
  getModelsListProvidersQueryKey,
  useModelsUpsertProvider,
} from '@nemo/sdk/generated/platform/model-providers';
import { ModelProvider } from '@nemo/sdk/generated/platform/schema';
import { Stack } from '@nvidia/foundations-react-core';
import { CreateSecretModal } from '@studio/routes/SecretsListRoute/CreateSecretModal';
import { EditSecretModal } from '@studio/routes/SecretsListRoute/EditSecretModal';
import { SecretSearchableSelect } from '@studio/routes/SecretsListRoute/SecretSearchableSelect';
import { useQueryClient } from '@tanstack/react-query';
import { FC, useEffect, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { z } from 'zod';

const editProviderFormSchema = z.object({
  host_url: z.string().min(1, 'Host URL is required').url('Enter a valid URL').max(2048),
  api_key_secret_name: z.string().max(255).optional().or(z.literal('')),
});

type EditProviderFormData = z.infer<typeof editProviderFormSchema>;

interface EditInferenceProviderModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  workspace: string;
  provider: ModelProvider;
}

export const EditInferenceProviderModal: FC<EditInferenceProviderModalProps> = ({
  workspace,
  provider,
  open,
  onClose,
}) => {
  const toast = useToast();
  const queryClient = useQueryClient();
  const [createSecretModalOpen, setCreateSecretModalOpen] = useState(false);
  const [editSecretName, setEditSecretName] = useState<string | null>(null);
  const currentSecretName = provider.api_key_secret_name ?? '';

  const {
    mutateAsync: upsertProvider,
    error: upsertError,
    isPending,
    reset: resetUpsertMutation,
  } = useModelsUpsertProvider({
    mutation: {
      onSuccess: () => {
        toast.success('Inference provider updated successfully');
        queryClient.invalidateQueries({
          queryKey: getModelsListProvidersQueryKey(workspace),
        });
        queryClient.invalidateQueries({
          queryKey: getModelsGetProviderQueryKey(workspace, provider.name),
        });
        resetAndClose();
      },
    },
  });

  const {
    control,
    reset: resetForm,
    handleSubmit,
    formState: { errors, isValid },
  } = useForm<EditProviderFormData>({
    resolver: zodResolver(editProviderFormSchema),
    defaultValues: {
      host_url: provider.host_url ?? '',
      api_key_secret_name: provider.api_key_secret_name ?? '',
    },
    disabled: isPending,
    mode: 'onChange',
  });

  useEffect(() => {
    if (open) {
      resetForm({
        host_url: provider.host_url ?? '',
        api_key_secret_name: provider.api_key_secret_name ?? '',
      });
    }
  }, [open, provider.name, provider.host_url, provider.api_key_secret_name, resetForm]);

  const reset = () => {
    resetUpsertMutation();
    resetForm({
      host_url: provider.host_url ?? '',
      api_key_secret_name: provider.api_key_secret_name ?? '',
    });
  };

  const resetAndClose = () => {
    reset();
    onClose();
  };

  const onSubmit: SubmitHandler<EditProviderFormData> = async (formData) => {
    try {
      await upsertProvider({
        workspace,
        name: provider.name,
        data: {
          host_url: formData.host_url,
          api_key_secret_name: formData.api_key_secret_name || undefined,
          project: provider.project,
          enabled_models: provider.enabled_models,
          default_extra_body: provider.default_extra_body,
          default_extra_headers: provider.default_extra_headers,
          required_extra_body: provider.required_extra_body,
          required_extra_headers: provider.required_extra_headers,
          model_deployment_id: provider.model_deployment_id,
          // Deliberately omit status/status_message on edit. The server resets status to
          // CREATED when the key or host_url changes so the reconciler re-evaluates;
          // re-sending the old status here would fight that (and previously re-asserted a
          // stale READY on a rotated key). Let the backend own provider status.
        },
      });
    } catch {
      // Error handling via mutation
    }
  };

  return (
    <>
      <FormModal
        open={open}
        onClose={resetAndClose}
        title="Edit Inference Provider"
        instruction="Update the endpoint and API key secret. Name cannot be changed"
        submitButtonText="Save"
        onSubmit={handleSubmit(onSubmit)}
        disabled={isPending}
        loading={isPending}
        submitDisabled={!isValid}
        errorText={upsertError?.message}
      >
        <Stack gap="density-xl">
          <ControlledTextInput
            useControllerProps={{ control, name: 'host_url' }}
            name="host_url"
            label="Host URL"
            formFieldProps={{
              slotError: errors.host_url?.message,
            }}
          />
          <SecretSearchableSelect
            workspace={workspace}
            queryEnabled={open && !!workspace}
            ensureOptionValue={currentSecretName || undefined}
            useControllerProps={{ control, name: 'api_key_secret_name' }}
            onRequestNewSecret={() => setCreateSecretModalOpen(true)}
            onRequestEditSecret={setEditSecretName}
            triggerPlaceholder=""
            formFieldProps={{
              slotLabel: 'API Key Secret',
              slotInfo: 'Name of a secret created in Secrets that holds the API key.',
              slotError: errors.api_key_secret_name?.message,
            }}
          />
        </Stack>
      </FormModal>
      <CreateSecretModal
        workspace={workspace}
        open={createSecretModalOpen}
        onClose={() => setCreateSecretModalOpen(false)}
      />
      {editSecretName ? (
        <EditSecretModal
          workspace={workspace}
          name={editSecretName}
          open
          onClose={() => setEditSecretName(null)}
        />
      ) : null}
    </>
  );
};
