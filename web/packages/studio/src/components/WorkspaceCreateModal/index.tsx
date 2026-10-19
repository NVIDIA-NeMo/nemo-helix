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
import { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
import {
  getEntitiesListWorkspacesQueryKey,
  useEntitiesCreateWorkspace,
} from '@nemo/sdk/generated/platform/entity-store';
import type { Workspace } from '@nemo/sdk/generated/platform/schema';
import { FormField, Stack, TextArea, TextInput } from '@nvidia/foundations-react-core';
import { queryClient } from '@studio/api/queryClient';
import { workspaceCreateSchema } from '@studio/constants/zod';
import { getWorkspaceDetailsDefaultRoute } from '@studio/routes/utils';
import { AxiosError } from 'axios';
import { FC } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useNavigate } from 'react-router';

interface FormFields {
  name: string;
  description?: string;
}

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
    formState: { errors, isSubmitting },
    setValue,
    setError,
  } = useForm<FormFields>({
    resolver: zodResolver(workspaceCreateSchema),
    mode: 'onChange',
  });

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
          slotError={errors.name?.message}
          status={errors.name && 'error'}
        >
          <TextInput
            autoFocus
            disabled={formDisabled}
            status={errors.name && 'error'}
            {...register('name')}
            onChange={(e) =>
              setValue('name', (e.target as HTMLInputElement).value.replace(/[\s-]+/g, '-'), {
                shouldValidate: true,
              })
            }
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
