// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal, type FormModalProps } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
import { useCreateDirectory } from '@studio/api/datasets/useCreateDirectory';
import { getSiblingFolders } from '@studio/components/filesets/AddToFolderModal/utils';
import type { FileSystemNode } from '@studio/components/FilesTable/utils';
import type { FC } from 'react';
import { type SubmitHandler, useForm } from 'react-hook-form';
import { z } from 'zod';

const formSchema = z.object({
  folderName: z
    .string()
    .min(1, 'Folder name is required')
    .refine((name) => !name.includes('/'), 'Folder name cannot contain slashes')
    .refine((name) => !name.startsWith('.') && name.trim() === name, {
      message: 'Folder name cannot start with a dot or have leading/trailing spaces',
    }),
});

type FormFields = z.infer<typeof formSchema>;

interface NewDirectoryModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  workspace: string;
  datasetName: string;
  currentFolder?: string;
  folderContents?: FileSystemNode[];
  onSuccess?: () => void;
}

export const NewDirectoryModal: FC<NewDirectoryModalProps> = ({
  open,
  onClose,
  workspace,
  datasetName,
  currentFolder,
  folderContents,
  onSuccess,
}) => {
  const toast = useToast();
  const siblingFolders = getSiblingFolders(folderContents);
  const { mutate: createDirectory, isPending } = useCreateDirectory({
    onError: (err) => {
      toast.error(`Failed to create folder: ${err.message}`);
    },
    onSuccess: () => {
      toast.success('Folder created successfully');
      resetAndClose();
      onSuccess?.();
    },
  });

  const {
    control,
    reset,
    handleSubmit,
    formState: { errors },
    watch,
  } = useForm<FormFields>({
    mode: 'onChange',
    defaultValues: { folderName: '' },
    resolver: zodResolver(formSchema),
  });

  const folderName = watch('folderName');

  const resetAndClose = () => {
    reset({ folderName: '' });
    onClose();
  };

  const isDuplicate =
    folderName.trim().length > 0 &&
    siblingFolders.some((f) => f.toLowerCase() === folderName.trim().toLowerCase());

  const onSubmit: SubmitHandler<FormFields> = async (fields) => {
    if (isDuplicate) {
      toast.error('A folder with this name already exists');
      return;
    }
    createDirectory({
      workspace,
      name: datasetName,
      folderName: fields.folderName.trim(),
      currentFolder,
    });
  };

  return (
    <FormModal
      title="New Directory"
      submitButtonText="Create"
      disabled={isPending}
      submitDisabled={isPending}
      loading={isPending}
      onSubmit={handleSubmit(
        onSubmit,
        handleFormErrorsGeneric({ title: 'New Directory Form Errors' })
      )}
      onClose={resetAndClose}
      open={open}
    >
      <ControlledTextInput
        useControllerProps={{
          name: 'folderName',
          control,
        }}
        label="Name"
        autoFocus
        disabled={isPending}
        status={errors.folderName || isDuplicate ? 'error' : undefined}
        formFieldProps={{
          slotLabel: 'Name',
          slotError:
            errors.folderName?.message ||
            (isDuplicate ? 'A folder with this name already exists' : ''),
          required: false,
        }}
      />
    </FormModal>
  );
};
