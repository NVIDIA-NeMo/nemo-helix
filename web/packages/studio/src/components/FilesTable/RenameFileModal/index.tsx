// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal, type FormModalProps } from '@nemo/common/src/components/FormModal';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
import { useDatasetFileRename } from '@studio/api/datasets/useDatasetFileRename';
import { useSelectedDatasetId } from '@studio/hooks/useSelectedDatasetId';
import type { FC } from 'react';
import { type SubmitHandler, useForm } from 'react-hook-form';

interface FormFields {
  name: string;
}

interface Props extends Pick<FormModalProps, 'open' | 'onClose'> {
  filepath: string;
  datasetId?: string;
  onSuccess?: (newFilePath: string) => void;
}

export const RenameFileModal: FC<Props> = ({ filepath, datasetId, open, onClose, onSuccess }) => {
  const toast = useToast();
  const resolvedDatasetId = useSelectedDatasetId({ datasetId });
  const { mutate: renameFile, isPending } = useDatasetFileRename({
    onError: (err) => {
      toast.error(`Unexpected error: ${err.message}`);
    },
    onSuccess: (_, { newFilePath }) => {
      toast.success('File successfully saved.');
      resetAndClose();
      onSuccess?.(newFilePath);
    },
  });
  const {
    control,
    reset,
    handleSubmit,
    formState: { errors, isValid },
  } = useForm<FormFields>({
    mode: 'onChange',
    defaultValues: {
      name: filepath,
    },
  });

  const formDisabled = isPending;

  const resetAndClose = () => {
    reset();
    onClose();
  };

  const onSubmit: SubmitHandler<FormFields> = async (fields) => {
    const { workspace, name } = getPartsFromReference(resolvedDatasetId);
    if (!workspace || !name) {
      return;
    }
    renameFile({ path: filepath, newFilePath: fields.name, workspace, name });
  };

  return (
    <FormModal
      title="Edit File"
      submitButtonText="Save"
      disabled={formDisabled}
      submitDisabled={!isValid}
      loading={isPending}
      onSubmit={handleSubmit(
        onSubmit,
        handleFormErrorsGeneric({ title: 'Rename File Form Errors' })
      )}
      onClose={resetAndClose}
      open={open}
    >
      <ControlledTextInput
        useControllerProps={{
          name: 'name',
          control,
          rules: { required: 'Name is required' },
        }}
        label="Name"
        autoFocus
        placeholder="Provide a new name"
        disabled={formDisabled}
        status={errors.name && 'error'}
        formFieldProps={{
          slotLabel: 'Name',
          slotError: errors.name?.message || '',
          required: false,
        }}
      />
    </FormModal>
  );
};
