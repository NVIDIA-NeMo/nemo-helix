// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { FormModal, FormModalProps } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { FormField, Stack, TextArea, TextInput } from '@nvidia/foundations-react-core';
import {
  createCustomizationJobTemplate,
  formFieldsToTemplateData,
  getCustomizationJobTemplatesQueryKey,
} from '@studio/api/customization-job-templates/customizationJobTemplates';
import { DATASET_NAME_REGEX } from '@studio/constants/constants';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { FC } from 'react';
import { useForm, type SubmitHandler } from 'react-hook-form';
import { z } from 'zod';

const formSchema = z.object({
  name: z
    .string()
    .min(1, { message: 'Name is required' })
    .regex(
      DATASET_NAME_REGEX,
      'Name must only contain alphanumeric characters, dashes, underscores, or dots'
    ),
  description: z.string().optional(),
});

type FormData = z.infer<typeof formSchema>;

const DEFAULT_VALUES: FormData = { name: '', description: '' };

export interface SaveCustomizationTemplateModalProps extends Pick<FormModalProps, 'open'> {
  onClose: () => void;
  workspace: string;
  /** Current form state to capture — including the dataset selection. */
  fields: CustomizationFormFields;
}

/**
 * Saves the in-progress fine-tuning form as a reusable `customization_job_template`
 * entity. Deliberately captures the raw form state rather than a backend job request:
 * the template is replayed through the same form, and only the per-backend
 * `formTo*Create` mappers ever produce a wire payload.
 */
export const SaveCustomizationTemplateModal: FC<SaveCustomizationTemplateModalProps> = ({
  open,
  onClose,
  workspace,
  fields,
}) => {
  const toast = useToast();
  const queryClient = useQueryClient();

  const {
    register,
    handleSubmit,
    reset: resetForm,
    formState: { errors },
  } = useForm<FormData>({
    resolver: zodResolver(formSchema),
    defaultValues: DEFAULT_VALUES,
    mode: 'onSubmit',
    reValidateMode: 'onChange',
  });

  const {
    mutate: saveTemplate,
    error,
    isPending,
    reset: resetMutation,
  } = useMutation({
    mutationFn: (formData: FormData) =>
      createCustomizationJobTemplate(workspace, {
        name: formData.name,
        data: formFieldsToTemplateData(fields, formData.description || undefined),
      }),
    onSuccess: (template) => {
      void queryClient.invalidateQueries({
        queryKey: getCustomizationJobTemplatesQueryKey(workspace).slice(0, 2),
      });
      toast.success(`Saved template '${template.name}'`);
      resetAndClose();
    },
  });

  const resetAndClose = () => {
    resetMutation();
    resetForm(DEFAULT_VALUES);
    onClose();
  };

  const onSubmit: SubmitHandler<FormData> = (formData) => saveTemplate(formData);

  return (
    <FormModal
      open={open}
      title="Save as template"
      instruction="Save the current settings — including the selected dataset — so you can start another fine-tuning job from them later."
      submitButtonText="Save template"
      errorText={error ? getErrorMessage(error, 'Failed to save template') : null}
      disabled={isPending}
      loading={isPending}
      onSubmit={handleSubmit(onSubmit)}
      onClose={resetAndClose}
    >
      <Stack gap="density-lg">
        <FormField
          slotLabel="Template name"
          slotError={errors.name?.message || ''}
          status={errors.name ? 'error' : undefined}
        >
          <TextInput
            required
            autoFocus
            placeholder="Name this template"
            status={errors.name && 'error'}
            {...register('name')}
          />
        </FormField>
        <FormField
          slotLabel="Description"
          slotError={errors.description?.message || ''}
          status={errors.description ? 'error' : undefined}
        >
          <TextArea
            placeholder="What is this template for?"
            status={errors.description && 'error'}
            {...register('description')}
          />
        </FormField>
      </Stack>
    </FormModal>
  );
};
