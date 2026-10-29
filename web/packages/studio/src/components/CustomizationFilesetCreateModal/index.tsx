// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { FormModal, FormModalProps } from '@nemo/common/src/components/FormModal';
import { FilesetFileUploadError } from '@nemo/common/src/datasets/constants';
import { getURNFromNamedEntityRef } from '@nemo/common/src/namedEntity';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { sanitizeEntityName, toValidEntityName } from '@nemo/common/src/utils/entityName';
import { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
import { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { TextInput, TextArea, FormField, Stack } from '@nvidia/foundations-react-core';
import { useDatasetCreate } from '@studio/api/datasets/useDatasetCreate';
import { CustomizationFileTag } from '@studio/components/CustomizationFileTag';
import { CustomizationFileUpload } from '@studio/components/CustomizationFileUpload';
import { ValueWithLabel } from '@studio/components/ValueWithLabel';
import { DEFAULT_NAMESPACE } from '@studio/constants/constants';
import {
  CUSTOMIZATION_FILESET_FILE_ACCEPT,
  CUSTOMIZATION_FILESET_FILE_HELPERS,
  CUSTOMIZATION_FILESET_FILE_LABELS,
  CUSTOMIZATION_FILESET_FILE_PREFIXES,
  CUSTOMIZATION_FILES_ALLOWED_FILE_EXTENSIONS,
  CustomizationFileType,
} from '@studio/constants/customization';
import { datasetSchema } from '@studio/constants/zod';
import { useCustomizationFilesPreview } from '@studio/hooks/useCustomizationFiles';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { renameFile } from '@studio/util/files';
import { FC, useState } from 'react';
import { Controller, SubmitHandler, useController, useForm } from 'react-hook-form';
import { z } from 'zod';

const isValidFileExtension = (file?: File) => {
  if (!file) return false;
  return CUSTOMIZATION_FILES_ALLOWED_FILE_EXTENSIONS.some((ext) =>
    file.name.toLowerCase().endsWith(ext.toLowerCase())
  );
};

const filesetFieldSchema = datasetSchema.extend({
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
});

const formSchema = z.object({
  fileset: filesetFieldSchema,
  files: z.object({
    trainingFile: z
      .custom<File | null>()
      .refine((file) => !!file, 'Training file is required')
      .refine(
        isValidFileExtension,
        `File must be one of the following types: ${CUSTOMIZATION_FILES_ALLOWED_FILE_EXTENSIONS.join(', ')}`
      ),
    validationFile: z
      .custom<File | null>()
      .refine((file) => !!file, 'Validation file is required')
      .refine(
        isValidFileExtension,
        `File must be one of the following types: ${CUSTOMIZATION_FILES_ALLOWED_FILE_EXTENSIONS.join(', ')}`
      ),
  }),
});

type FormData = z.infer<typeof formSchema>;

const DEFAULT_VALUES: FormData = {
  fileset: {
    name: '',
    namespace: DEFAULT_NAMESPACE,
    description: '',
  },
  files: {
    trainingFile: null as unknown as File,
    validationFile: null as unknown as File,
  },
};

export interface CustomizationFilesetCreateModalProps extends Pick<
  FormModalProps,
  'open' | 'onClose'
> {
  onFilesetCreated: (fileset: FilesetOutput) => void;
}

export const CustomizationFilesetCreateModal: FC<CustomizationFilesetCreateModalProps> = ({
  open,
  onClose,
  onFilesetCreated,
}) => {
  const toast = useToast();
  const workspace = useWorkspaceFromPath();
  const [existingFileset, setExistingFileset] = useState<FilesetOutput>();
  const { training, validation } = useCustomizationFilesPreview({
    fileset: getURNFromNamedEntityRef(existingFileset),
  });

  const hasTrainingFile = training.hasFiles;
  const hasValidationFile = validation.hasFiles;

  const handleSuccess = (fileset: FilesetOutput) => {
    reset();
    onFilesetCreated(fileset);
    toast.success('Successfully created dataset!');
  };

  const handleError = (error: Error) => {
    if (error instanceof FilesetFileUploadError) {
      setExistingFileset(error.fileset);
      // Now mark the form as not dirty, so we can create a new fileset if they change any of the fileset fields,
      // or reuse existing fileset if they don't
      resetForm({ ...getValues() });
    }
  };

  const {
    mutate: createDataset,
    error,
    isPending,
    reset: resetMutation,
  } = useDatasetCreate({ onSuccess: handleSuccess, onError: handleError });

  const {
    control,
    reset: resetForm,
    register,
    handleSubmit,
    formState: { errors, dirtyFields, disabled: formDisabled },
    getValues,
  } = useForm({
    resolver: zodResolver(formSchema),
    defaultValues: DEFAULT_VALUES,
    disabled: isPending,
    mode: 'onSubmit', // Validate on submit to show all errors at once
    reValidateMode: 'onChange', // Then validate in real-time after first submit
  });

  const reset = () => {
    resetMutation();
    resetForm(DEFAULT_VALUES);
    setExistingFileset(undefined);
  };

  const resetAndClose = () => {
    reset();
    onClose();
  };

  const onSubmit: SubmitHandler<FormData> = async (formData) => {
    const {
      fileset: createFilesetRequest,
      files: { trainingFile, validationFile },
    } = formData;

    // If an error occurred during file upload, but a fileset was created, let's reuse it
    // However, if the user decided to update the name/description in the form after an
    // error occurred, then create a new fileset.
    const shouldCreateNewFileset = !existingFileset || !!dirtyFields.fileset;
    const trainingFileNameFixed = renameFile(
      trainingFile,
      `${CUSTOMIZATION_FILESET_FILE_PREFIXES.Training}/${trainingFile.name}`
    );

    const validationFileNameFixed = renameFile(
      validationFile,
      `${CUSTOMIZATION_FILESET_FILE_PREFIXES.Validation}/${validationFile.name}`
    );

    const filesToUpload = [trainingFileNameFixed, validationFileNameFixed];

    createDataset({
      workspace: shouldCreateNewFileset ? workspace : undefined,
      dataset: shouldCreateNewFileset ? undefined : existingFileset,
      request: shouldCreateNewFileset ? createFilesetRequest : undefined,
      files: filesToUpload,
    });
  };

  const {
    field: { value: filesetNameValue, onChange: onFilesetNameChange, onBlur: onFilesetNameBlur },
  } = useController({ control, name: 'fileset.name' });
  const filesetDescription = register('fileset.description');

  const filesetPreview = toValidEntityName(filesetNameValue ?? '', '');
  const filesetNameError = errors.fileset?.name?.message;

  return (
    <FormModal
      title="Create New Dataset"
      instruction="To create a new dataset, simply provide a name, description, and select a training file and validation file."
      submitButtonText="Add to Customization"
      errorText={error?.message}
      disabled={formDisabled}
      loading={isPending}
      onSubmit={handleSubmit(
        onSubmit,
        handleFormErrorsGeneric({ title: 'Customization Fileset Create Form Errors' })
      )}
      onClose={resetAndClose}
      open={open}
    >
      <Stack gap="density-md">
        <FormField
          slotLabel="Name"
          slotError={filesetNameError || ''}
          slotHelp={
            filesetNameError ? undefined : filesetPreview ? (
              <>
                Your dataset will be created as{' '}
                <span className="text-primary">{filesetPreview}</span>
              </>
            ) : undefined
          }
          status={filesetNameError ? 'error' : undefined}
        >
          <TextInput
            required
            autoFocus
            placeholder="Name this dataset"
            status={filesetNameError ? 'error' : undefined}
            value={filesetNameValue}
            onChange={(e) => onFilesetNameChange((e.target as HTMLInputElement).value)}
            onBlur={onFilesetNameBlur}
          />
        </FormField>
        <FormField
          slotLabel="Description"
          slotError={errors.fileset?.description?.message || ''}
          status={errors.fileset?.description ? 'error' : undefined}
        >
          <TextArea
            placeholder="Provide a useful description for this dataset"
            status={errors.fileset?.description && 'error'}
            {...filesetDescription}
          />
        </FormField>
        {hasTrainingFile ? (
          <ValueWithLabel
            label="Training File"
            value={
              <CustomizationFileTag
                fileType={CustomizationFileType.Training}
                files={training.files}
                total={training.total}
                status="success"
              />
            }
          />
        ) : (
          <Controller
            control={control}
            name="files.trainingFile"
            render={({ field, fieldState }) => (
              <CustomizationFileUpload
                files={field.value ? [field.value] : undefined}
                customizationFileType={CustomizationFileType.Training}
                helperText={CUSTOMIZATION_FILESET_FILE_HELPERS[CustomizationFileType.Training]}
                label={CUSTOMIZATION_FILESET_FILE_LABELS[CustomizationFileType.Training]}
                status={fieldState.error ? 'error' : 'pending'}
                onChange={field.onChange}
                errorText={
                  fieldState.error?.message || errors.files?.trainingFile?.message?.toString()
                }
                disabled={formDisabled}
              />
            )}
          />
        )}
        {hasValidationFile ? (
          <ValueWithLabel
            label="Validation File"
            value={
              <CustomizationFileTag
                fileType={CustomizationFileType.Validation}
                files={validation.files}
                total={validation.total}
                status="success"
              />
            }
          />
        ) : (
          <Controller
            control={control}
            name="files.validationFile"
            render={({ field, fieldState }) => (
              <CustomizationFileUpload
                accept={CUSTOMIZATION_FILESET_FILE_ACCEPT}
                files={field.value ? [field.value] : undefined}
                customizationFileType={CustomizationFileType.Validation}
                helperText={CUSTOMIZATION_FILESET_FILE_HELPERS[CustomizationFileType.Validation]}
                label={CUSTOMIZATION_FILESET_FILE_LABELS[CustomizationFileType.Validation]}
                status={fieldState.error ? 'error' : field.value ? 'success' : 'pending'}
                onChange={field.onChange}
                errorText={
                  fieldState.error?.message || errors.files?.validationFile?.message?.toString()
                }
                disabled={formDisabled}
              />
            )}
          />
        )}
      </Stack>
    </FormModal>
  );
};
