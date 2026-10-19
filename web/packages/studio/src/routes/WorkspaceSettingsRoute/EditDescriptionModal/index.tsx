// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { ControlledTextArea } from '@nemo/common/src/components/form/ControlledTextArea';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal, FormModalProps } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
import {
  getEntitiesGetWorkspaceQueryKey,
  getEntitiesListWorkspacesQueryKey,
  useEntitiesGetWorkspace,
  useEntitiesUpdateWorkspace,
} from '@nemo/sdk/generated/platform/entity-store';
import { EntitiesUpdateWorkspaceBody } from '@nemo/sdk/generated/platform/zod/entity-store';
import { Stack, Text } from '@nvidia/foundations-react-core';
import { queryClient } from '@studio/api/queryClient';
import { FC, useEffect } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { z } from 'zod';

const formSchema = EntitiesUpdateWorkspaceBody.extend({
  name: z.string(),
});

type FormFields = z.infer<typeof formSchema>;

interface EditDescriptionModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  workspace: string;
}

export const EditDescriptionModal: FC<EditDescriptionModalProps> = ({
  workspace,
  open,
  onClose,
}) => {
  const { data: workspaceData } = useEntitiesGetWorkspace(workspace);

  const {
    control,
    reset,
    handleSubmit,
    formState: { errors },
  } = useForm<FormFields>({
    resolver: zodResolver(formSchema),
    mode: 'onChange',
    defaultValues: {
      name: workspace,
      description: workspaceData?.description ?? '',
    },
  });

  useEffect(() => {
    if (open && workspaceData) {
      reset({ name: workspace, description: workspaceData.description ?? '' });
    }
  }, [open, workspaceData, reset, workspace]);

  const toast = useToast();
  const { mutateAsync: updateWorkspace, isPending } = useEntitiesUpdateWorkspace();

  const resetAndClose = () => {
    reset();
    onClose();
  };

  const onSubmit: SubmitHandler<FormFields> = async (fields) => {
    try {
      await updateWorkspace({
        name: workspace,
        data: {
          description: fields.description ?? undefined,
        },
      });
      queryClient.invalidateQueries({
        queryKey: getEntitiesGetWorkspaceQueryKey(workspace),
      });
      queryClient.invalidateQueries({ queryKey: getEntitiesListWorkspacesQueryKey() });
      resetAndClose();
      toast.success('Successfully updated workspace!');
    } catch {
      toast.error('Something went wrong. Please try again.');
    }
  };

  return (
    <FormModal
      title="Edit Description"
      instruction="Modify your workspace description to reflect its purpose."
      submitButtonText="Save"
      disabled={isPending}
      loading={isPending}
      onSubmit={handleSubmit(
        onSubmit,
        handleFormErrorsGeneric({ title: 'Edit Description Form Errors' })
      )}
      onClose={resetAndClose}
      open={open}
    >
      <Stack gap="density-2xl">
        <ControlledTextInput
          useControllerProps={{ control, name: 'name' }}
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
          label="Description (optional)"
          formFieldProps={{
            slotError: errors.description?.message,
          }}
        />
      </Stack>
    </FormModal>
  );
};
