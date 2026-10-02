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
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { ControlledTextArea } from '@nemo/common/src/components/form/ControlledTextArea';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal, FormModalProps } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import type { HelixSecretResponse } from '@nemo/sdk/generated/platform/schema';
import {
  getSecretsListSecretsQueryKey,
  useSecretsGetSecret,
  useSecretsUpdateSecret,
} from '@nemo/sdk/generated/platform/secrets';
import { Flex, FormField, Spinner, Stack, Text, TextInput } from '@nvidia/foundations-react-core';
import { useQueryClient } from '@tanstack/react-query';
import { FC } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { z } from 'zod';

const editSecretFormSchema = z.object({
  name: z.string(),
  description: z.string().optional(),
  value: z.string().min(1, 'Secret value is required'),
});

type EditSecretFormData = z.infer<typeof editSecretFormSchema>;

interface EditSecretModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  workspace: string;
  /** Name of the secret to edit. The record is fetched so the description defaults to the stored one. */
  name: string;
}

/**
 * Callers reach this from a secrets table row and from the secret picker inside other forms, and
 * only the table has the full record. Fetching by name keeps one contract for both, and keeps a
 * stale description from a cached list page out of a field that overwrites the stored one on save.
 */
export const EditSecretModal: FC<EditSecretModalProps> = ({ workspace, name, open, onClose }) => {
  const {
    data: secret,
    error,
    isLoading,
  } = useSecretsGetSecret(workspace, name, { query: { enabled: open && Boolean(name) } });

  // Remounts per secret so the form's defaults are the ones it was opened with.
  if (secret) {
    return (
      <EditSecretForm
        key={`${workspace}/${name}`}
        workspace={workspace}
        secret={secret}
        open={open}
        onClose={onClose}
      />
    );
  }

  return (
    <FormModal
      open={open}
      onClose={onClose}
      title="Edit Secret"
      submitButtonText="Save"
      onSubmit={(event) => event.preventDefault()}
      submitDisabled
      errorText={error ? getErrorMessage(error) : undefined}
    >
      {isLoading ? (
        <Flex align="center" justify="center" className="py-8">
          <Spinner aria-label={`Loading ${name}`} />
        </Flex>
      ) : null}
    </FormModal>
  );
};

interface EditSecretFormProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  workspace: string;
  secret: HelixSecretResponse;
}

const EditSecretForm: FC<EditSecretFormProps> = ({ workspace, secret, open, onClose }) => {
  const toast = useToast();
  const queryClient = useQueryClient();

  const {
    mutateAsync: updateSecret,
    error: updateError,
    isPending,
    reset: resetUpdateMutation,
  } = useSecretsUpdateSecret({
    mutation: {
      onSuccess: () => {
        toast.success('Secret updated successfully');
        queryClient.invalidateQueries({ queryKey: getSecretsListSecretsQueryKey(workspace) });
        resetAndClose();
      },
    },
  });

  const defaultValues: EditSecretFormData = {
    name: secret.name || '',
    description: secret.description || '',
    value: '',
  };

  const {
    control,
    reset: resetForm,
    handleSubmit,
    formState: { errors },
  } = useForm({
    resolver: zodResolver(editSecretFormSchema),
    defaultValues,
    disabled: isPending,
    mode: 'onChange',
  });

  const reset = () => {
    resetUpdateMutation();
    resetForm(defaultValues);
  };

  const resetAndClose = () => {
    reset();
    onClose();
  };

  const onSubmit: SubmitHandler<EditSecretFormData> = async (formData) => {
    try {
      await updateSecret({
        workspace,
        name: secret.name,
        data: {
          description: formData.description,
          value: formData.value,
        },
      });
    } catch {
      // Error handling is done via toast in mutation onError
    }
  };

  return (
    <FormModal
      open={open}
      onClose={resetAndClose}
      title="Edit Secret"
      submitButtonText="Save"
      onSubmit={handleSubmit(onSubmit)}
      disabled={isPending}
      loading={isPending}
      errorText={updateError ? getErrorMessage(updateError) : undefined}
    >
      <Stack gap="density-xl">
        <ControlledTextInput
          useControllerProps={{ control, name: 'name' }}
          name="name"
          label="Name"
          disabled
          formFieldProps={{
            slotHelp: (
              <Text kind="label/regular/sm" className="text-secondary">
                Name editing is not supported.
              </Text>
            ),
          }}
        />
        <ControlledTextArea
          useControllerProps={{ control, name: 'description' }}
          name="description"
          label="Description (optional)"
          formFieldProps={{
            slotError: errors.description?.message,
          }}
          rows={2}
        />
        <FormField
          slotLabel="Current Value"
          slotHelp="Current value is hidden for security. Provide a new value below to override it."
        >
          <TextInput value="••••••••••••••••••••••••••••••••••••••••" disabled />
        </FormField>
        <ControlledTextInput
          useControllerProps={{ control, name: 'value' }}
          name="value"
          label="New Value"
          placeholder="Enter new value to replace the current one"
          formFieldProps={{
            slotError: errors.value?.message,
          }}
        />
      </Stack>
    </FormModal>
  );
};
