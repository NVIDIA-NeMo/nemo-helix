/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import { FormModal, FormModalProps } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { sanitizeEntityName, toValidEntityName } from '@nemo/common/src/utils/entityName';
import { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
import {
  getEntitiesListWorkspacesQueryKey,
  useEntitiesCreateWorkspace,
} from '@nemo/sdk/generated/platform/entity-store';
import type { Workspace } from '@nemo/sdk/generated/platform/schema';
import { FormField, Stack, TextArea, TextInput } from '@nvidia/foundations-react-core';
import { queryClient } from '@studio/api/queryClient';
import { getWorkspaceDetailsDefaultRoute } from '@studio/routes/utils';
import { AxiosError } from 'axios';
import { FC } from 'react';
import { SubmitHandler, useController, useForm } from 'react-hook-form';
import { useNavigate } from 'react-router';
import { z } from 'zod';

const workspaceFormSchema = z.object({
  name: z
    .string()
    .superRefine((value, ctx) => {
      if (sanitizeEntityName(value) === undefined) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: value
            ? 'Name must start with a letter and be at least 2 characters.'
            : 'Name is required.',
        });
      }
    })
    .transform((value) => toValidEntityName(value, value)),
  description: z.string().optional(),
});

type FormFields = z.infer<typeof workspaceFormSchema>;

export type WorkspaceCreateModalProps = Pick<FormModalProps, 'open' | 'onClose'> & {
  onCreate?: (workspace: Workspace) => void;
};

export const WorkspaceCreateModal: FC<WorkspaceCreateModalProps> = ({
  open,
  onClose,
  onCreate,
}) => {
  const {
    reset,
    register,
    handleSubmit,
    formState: { errors, isSubmitted, isSubmitting, touchedFields },
    setError,
    control,
  } = useForm<FormFields>({
    resolver: zodResolver(workspaceFormSchema),
    mode: 'onTouched',
    defaultValues: { name: '', description: '' },
  });

  const {
    field: { value: nameValue, onChange: onNameChange, onBlur: onNameBlur },
  } = useController({ control, name: 'name' });

  const preview = toValidEntityName(nameValue, '');
  const nameError = touchedFields.name || isSubmitted ? errors.name?.message : undefined;

  const formDisabled = isSubmitting;

  const navigate = useNavigate();
  const toast = useToast();

  const { mutateAsync: createWorkspace } = useEntitiesCreateWorkspace({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: getEntitiesListWorkspacesQueryKey() });
      },
    },
  });

  const resetAndClose = () => {
    reset();
    onClose();
  };

  const onSubmit: SubmitHandler<FormFields> = async (data) => {
    try {
      const workspace = await createWorkspace({
        data: {
          name: data.name,
          description: data.description,
        },
      });
      onCreate?.(workspace);
      resetAndClose();
      navigate({
        pathname: getWorkspaceDetailsDefaultRoute(workspace.name),
      });
    } catch (error) {
      const errorDetail =
        error instanceof AxiosError && error.response?.data?.detail
          ? error.response.data.detail
          : undefined;

      if (errorDetail === `Workspace ${data.name} already exists.`) {
        setError('name', { message: errorDetail });
      } else {
        // Extract message from errorDetail if it's an array with msg property
        let errorMessage: string;
        if (Array.isArray(errorDetail) && errorDetail.length > 0 && errorDetail[0].msg) {
          errorMessage = errorDetail[0].msg;
        } else if (errorDetail && typeof errorDetail === 'string') {
          errorMessage = errorDetail;
        } else if (error instanceof Error) {
          errorMessage = error.message;
        } else {
          errorMessage = 'Unknown error';
        }

        toast.error(`Failed to create workspace: ${errorMessage}`);
      }
    }
  };

  return (
    <FormModal
      title="New Workspace"
      instruction="Provide a name and description to group related entities for easier organization and management."
      submitButtonText="Create"
      disabled={formDisabled}
      loading={isSubmitting}
      onSubmit={handleSubmit(
        onSubmit,
        handleFormErrorsGeneric({ title: 'Workspace Create Form Errors' })
      )}
      onClose={resetAndClose}
      open={open}
    >
      <Stack gap="density-2xl">
        <FormField
          slotLabel="Name"
          slotError={nameError}
          slotHelp={
            nameError ? undefined : preview ? (
              <>
                Your workspace will be created as <span className="text-primary">{preview}</span>
              </>
            ) : undefined
          }
          status={nameError ? 'error' : undefined}
        >
          <TextInput
            autoFocus
            disabled={formDisabled}
            status={nameError ? 'error' : undefined}
            value={nameValue}
            onChange={(e) => onNameChange((e.target as HTMLInputElement).value)}
            onBlur={onNameBlur}
          />
        </FormField>

        <FormField
          slotLabel="Description (optional)"
          slotError={errors.description?.message}
          status={errors.description && 'error'}
        >
          <TextArea
            disabled={formDisabled}
            status={errors.description && 'error'}
            {...register('description')}
          />
        </FormField>
      </Stack>
    </FormModal>
  );
};
