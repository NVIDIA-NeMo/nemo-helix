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
import { ControlledTextArea } from '@nemo/common/src/components/form/ControlledTextArea';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal, type FormModalProps } from '@nemo/common/src/components/FormModal';
import type { NotifyFn } from '@nemo/common/src/providers/toast/types';
import { useNotify } from '@nemo/common/src/providers/toast/useNotify';
import { sanitizeEntityName, toValidEntityName } from '@nemo/common/src/utils/entityName';
import { Stack } from '@nvidia/foundations-react-core';
import type { FC } from 'react';
import { type SubmitHandler, useForm } from 'react-hook-form';
import { z } from 'zod';

const secretFormSchema = z.object({
  name: z
    .string()
    .superRefine((value, ctx) => {
      if (sanitizeEntityName(value) === undefined) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: value ? 'Name must contain at least one letter or number.' : 'Name is required.',
        });
      }
    })
    .transform((value) => toValidEntityName(value, value)),
  description: z.string().optional(),
  value: z.string().min(1, 'Secret value is required'),
});

export type CreateSecretFormData = z.infer<typeof secretFormSchema>;

const defaultValues: CreateSecretFormData = {
  name: '',
  description: '',
  value: '',
};

export interface CreateSecretModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  /** Performs the create. Rejecting leaves the modal open; the caller surfaces the reason via `errorText`. */
  onCreate: (data: CreateSecretFormData) => Promise<void>;
  pending?: boolean;
  errorText?: string;
  /** Where result messages go. Defaults to the surrounding ToastProvider; plugins pass `host.notifications.notify`. */
  onNotify?: NotifyFn;
}

export const CreateSecretModal: FC<CreateSecretModalProps> = ({
  onCreate,
  pending = false,
  errorText,
  onNotify,
  open,
  onClose,
}) => {
  const notify = useNotify(onNotify);

  const {
    control,
    reset: resetForm,
    handleSubmit,
    watch,
    formState: { errors, touchedFields },
  } = useForm({
    resolver: zodResolver(secretFormSchema),
    defaultValues,
    disabled: pending,
    mode: 'onBlur',
  });

  const rawSecretName = watch('name');
  const secretPreview = toValidEntityName(rawSecretName ?? '', '');
  const nameError = touchedFields.name ? errors.name?.message : undefined;

  const resetAndClose = () => {
    resetForm(defaultValues);
    onClose();
  };

  const onSubmit: SubmitHandler<CreateSecretFormData> = async (formData) => {
    try {
      await onCreate(formData);
    } catch {
      // Reported through errorText; the modal stays open so the value isn't lost.
      return;
    }
    notify('Secret created successfully', 'success');
    resetAndClose();
  };

  return (
    <FormModal
      open={open}
      onClose={resetAndClose}
      title="Create Secret"
      instruction="To create a new secret, provide a name, description, and value. "
      submitButtonText="Create"
      onSubmit={handleSubmit(onSubmit)}
      disabled={pending}
      loading={pending}
      errorText={errorText}
    >
      <Stack gap="density-xl">
        <ControlledTextInput
          useControllerProps={{ control, name: 'name' }}
          name="name"
          label="Name"
          formFieldProps={{
            slotInfo:
              'Best practice: Use lowercase letters, numbers, and hyphens only to ensure compatibility with Kubernetes naming conventions.',
            slotHelp: nameError ? undefined : secretPreview ? (
              <>
                Your secret will be created as <span className="text-primary">{secretPreview}</span>
              </>
            ) : undefined,
            slotError: nameError,
            status: nameError ? 'error' : undefined,
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
        <ControlledTextInput
          masked
          useControllerProps={{ control, name: 'value' }}
          name="value"
          label="Value"
          formFieldProps={{
            slotInfo:
              'For security, the secret value will be encrypted and not displayed after creation.',
            slotError: errors.value?.message,
          }}
        />
      </Stack>
    </FormModal>
  );
};
