// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { ControlledCombobox } from '@nemo/common/src/components/form/ControlledCombobox';
import { ControlledTextArea } from '@nemo/common/src/components/form/ControlledTextArea';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal, FormModalProps } from '@nemo/common/src/components/FormModal';
import { FilesetFileUploadError } from '@nemo/common/src/datasets/constants';
import { getEntityReference } from '@nemo/common/src/namedEntity';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
import { useFilesUpdateFilesetMetadata } from '@nemo/sdk/generated/platform/files';
import { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { Select, Stack } from '@nvidia/foundations-react-core';
import { invalidateDatasetCaches } from '@studio/api/datasets/invalidateDatasetCaches';
import { useDatasetCreate } from '@studio/api/datasets/useDatasetCreate';
import {
  DATASET_CREATE_DEFAULT_VALUES,
  DATASET_CREATE_MODAL_CONTENT,
  DatasetCreateFormData,
  datasetCreateFormSchema,
  DatasetCreateModalMode,
} from '@studio/components/DatasetCreateModal/constants';
import { DatasetFileUpload } from '@studio/components/DatasetFileUpload';
import { CUSTOMIZATION_FILESET_FILE_PREFIXES } from '@studio/constants/customization';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getFilesetDetailsRoute } from '@studio/routes/utils';
import { renameFile } from '@studio/util/files';
import { FC, useState } from 'react';
import { Controller, SubmitHandler, useForm } from 'react-hook-form';
import { useNavigate } from 'react-router';

interface DatasetCreateModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  dataset?: FilesetOutput;
  onDatasetCreated?: (dataset: FilesetOutput) => void;
  onDatasetUpdated?: (dataset: FilesetOutput) => void;
  mode?: DatasetCreateModalMode;
  currentFolderPath?: string;
  defaultFiles?: File[];
}

export const DatasetCreateModal: FC<DatasetCreateModalProps> = ({
  dataset,
  mode = DatasetCreateModalMode.Dataset,
  open,
  onClose,
  onDatasetCreated,
  onDatasetUpdated,
  currentFolderPath,
  defaultFiles,
}) => {
  const toast = useToast();
  const navigate = useNavigate();
  const workspace = useWorkspaceFromPath();
  const [existingDataset, setExistingDataset] = useState<FilesetOutput | undefined>(dataset);
  const isEditMode = mode === DatasetCreateModalMode.Edit;

  const defaultValues: DatasetCreateFormData = {
    dataset: {
      name: existingDataset?.name ?? DATASET_CREATE_DEFAULT_VALUES.dataset.name,
      description:
        existingDataset?.description ?? DATASET_CREATE_DEFAULT_VALUES.dataset.description,
    },
    folderPrefix: currentFolderPath ?? DATASET_CREATE_DEFAULT_VALUES.folderPrefix,
    files: defaultFiles ?? DATASET_CREATE_DEFAULT_VALUES.files,
  };

  const {
    mutateAsync: createDataset,
    error: createError,
    isPending: isCreatePending,
    reset: resetCreateMutation,
  } = useDatasetCreate({
    onSuccess: (dataset: FilesetOutput) => {
      reset();
      onDatasetCreated?.(dataset);
      navigate(getFilesetDetailsRoute(workspace, getEntityReference(dataset), undefined, true));
    },
  });

  const {
    mutateAsync: updateDataset,
    error: updateError,
    isPending: isUpdatePending,
    reset: resetUpdateMutation,
  } = useFilesUpdateFilesetMetadata({
    mutation: {
      onSuccess: (dataset: FilesetOutput) => {
        reset();
        invalidateDatasetCaches(dataset.workspace, dataset.name, ['list', 'detail']);
        onDatasetUpdated?.(dataset);
      },
    },
  });

  const error = isEditMode ? updateError : createError;
  const isPending = isEditMode ? isUpdatePending : isCreatePending;

  const {
    control,
    reset: resetForm,
    handleSubmit,
    setValue,
    formState: { errors, disabled: formDisabled, isValid },
  } = useForm({
    resolver: zodResolver(datasetCreateFormSchema),
    defaultValues,
    disabled: isPending,
    mode: 'onChange',
  });

  const reset = () => {
    if (isEditMode) {
      resetUpdateMutation();
    } else {
      resetCreateMutation();
    }
    resetForm(DATASET_CREATE_DEFAULT_VALUES);
  };

  const hasFiles = mode !== DatasetCreateModalMode.Dataset;
  const hasDataset = mode !== DatasetCreateModalMode.Files;
  const datasetSelectDefaultOption = {
    value: dataset?.name ?? '',
    children: dataset?.name ?? 'Missing dataset',
  };
  const { title, instruction, action, successToast } = DATASET_CREATE_MODAL_CONTENT[mode];

  const resetAndClose = () => {
    reset();
    onClose();
  };

  const onSubmitEdit: SubmitHandler<DatasetCreateFormData> = async (formData) => {
    if (!existingDataset) {
      throw new Error('Dataset is required for update');
    }
    await updateDataset({
      workspace: existingDataset.workspace,
      name: existingDataset.name,
      data: {
        description: formData.dataset.description,
      },
    });
    toast.success('Dataset updated successfully');
    resetAndClose();
  };

  const onSubmitCreate: SubmitHandler<DatasetCreateFormData> = async (formData) => {
    const { dataset: createDatasetRequest, folderPrefix, files } = formData;
    let usedFiles = files;
    if (files && folderPrefix) {
      usedFiles = files.map((file) => {
        const prefix = folderPrefix.endsWith('/') ? folderPrefix : folderPrefix + '/';
        return renameFile(file, prefix + file.name);
      });
    }
    try {
      await createDataset({
        workspace,
        request: {
          name: createDatasetRequest.name,
          description: createDatasetRequest.description || undefined,
        },
        // This exists in case the user already received an error creating the dataset and is retrying.
        // If they had already created a dataset, we don't want to create a new one, we want to reuse it,
        // and just retry  the file upload part.
        dataset: existingDataset,
        files: usedFiles,
      });
      toast.success(successToast);
      setExistingDataset(undefined);
    } catch (err) {
      if (err instanceof FilesetFileUploadError) {
        setExistingDataset(err.fileset);
      }
    }
  };

  return (
    <FormModal
      title={title}
      instruction={instruction}
      submitButtonText={action}
      errorText={error?.message}
      disabled={formDisabled}
      submitDisabled={formDisabled || !isValid}
      loading={isPending}
      onSubmit={handleSubmit(
        isEditMode ? onSubmitEdit : onSubmitCreate,
        handleFormErrorsGeneric({ title: 'Dataset Create Form Errors' })
      )}
      onClose={resetAndClose}
      open={open}
    >
      <Stack gap="density-md">
        {hasDataset && (
          <>
            <ControlledTextInput
              useControllerProps={{ control, name: 'dataset.name' }}
              placeholder="Name this dataset"
              autoFocus={!isEditMode}
              disabled={isEditMode}
              attributes={
                isEditMode
                  ? {
                      Input: {
                        // Fixes a KUI issue where when it's disabled, the text color is white in dark mode.
                        className: 'text-disabled',
                      },
                    }
                  : undefined
              }
              onChange={(e) => {
                const target = e.target as HTMLInputElement;
                const newVal = target.value.replace(/[\s-]+/g, '-');
                setValue('dataset.name', newVal, { shouldValidate: true });
              }}
              formFieldProps={{
                slotLabel: 'Name',
                slotError: errors.dataset?.name?.message || '',
                slotHelp: isEditMode ? 'Name editing is not currently supported' : undefined,
                required: !isEditMode,
              }}
              required={!isEditMode}
            />
            <ControlledTextArea
              useControllerProps={{ control, name: 'dataset.description' }}
              label="Description (optional)"
              placeholder="Provide a useful description for this dataset"
            />
          </>
        )}
        {!hasDataset && (
          <Select
            disabled
            value={datasetSelectDefaultOption.value}
            items={[datasetSelectDefaultOption]}
          />
        )}
        {hasFiles && !isEditMode && (
          <>
            <Controller
              control={control}
              name="files"
              render={({ field }) => {
                return (
                  <DatasetFileUpload
                    multiple
                    required={false}
                    label="Files"
                    files={field.value?.length ? field.value : undefined}
                    onChange={field.onChange}
                    errorText={errors.files?.message}
                    disabled={formDisabled}
                  />
                );
              }}
            />
            <ControlledCombobox
              useControllerProps={{ control, name: 'folderPrefix' }}
              label="Folder (optional)"
              resetValueOnBlur={false}
              items={Object.values(CUSTOMIZATION_FILESET_FILE_PREFIXES)}
              formFieldProps={{
                slotHelp:
                  'This sets the path for your uploaded file(s). Please use a "/" to delineate your folders.',
              }}
            />
          </>
        )}
      </Stack>
    </FormModal>
  );
};
