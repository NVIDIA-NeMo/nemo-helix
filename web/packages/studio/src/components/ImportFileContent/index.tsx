// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Stack, TabItem, Tabs } from '@nvidia/foundations-react-core';
import { Db } from '@nvidia/react-gui-icons/icons';
import { DatasetFileSelect } from '@studio/components/DatasetFileSelect';
import { DatasetSelect } from '@studio/components/DatasetSelect';
import { ControlledFileUpload } from '@studio/components/form/ControlledFileUpload';
import { ImportFileContentFormFields } from '@studio/components/ImportFileContent/validation';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getWorkspaceFilesetsRoute } from '@studio/routes/utils';
import { FC, ReactNode } from 'react';
import { Controller, useFormContext, useWatch } from 'react-hook-form';
import { Link } from 'react-router';

interface Props {
  prefer?: 'upload' | 'dataset';
  disabled?: boolean;
  slotDatasetDescription?: ReactNode;
  slotError?: ReactNode;
}

export const ImportFileContent: FC<Props> = ({
  prefer,
  disabled,
  slotDatasetDescription,
  slotError,
}) => {
  const workspace = useWorkspaceFromPath();
  const {
    control,
    formState: { errors },
    resetField,
  } = useFormContext<ImportFileContentFormFields>();
  const datasetId = useWatch({ control, name: 'datasetId' });

  // Reset the form fields when the tab is changed
  const handleValueChange = (value: string) => {
    if (value === 'upload') {
      resetField('datasetId');
      resetField('filepath');
    } else {
      resetField('file');
    }
  };

  const items: TabItem[] = [
    {
      children: 'Upload',
      value: 'upload',
      disabled,
      slotContent: (
        <Stack gap="density-md" className="w-full">
          <ControlledFileUpload
            useControllerProps={{ name: 'file', control }}
            errorText={
              errors.file?.message ? (
                <>
                  {errors.file.message.endsWith('.')
                    ? errors.file.message
                    : `${errors.file.message}.`}
                  {slotError && !errors.file.message.includes('Unsupported file type') && (
                    <> {slotError}</>
                  )}
                </>
              ) : undefined
            }
            label="File"
          />
        </Stack>
      ),
    },
    {
      children: 'Select from dataset',
      disabled,
      value: 'dataset',
      slotContent: (
        <Stack gap="density-xl" className="w-full">
          {slotDatasetDescription}
          <Controller
            name="datasetId"
            control={control}
            rules={{ required: 'Dataset is required.' }}
            render={({ field: { onChange: fieldOnChange, value } }) => (
              <DatasetSelect
                onChange={(datasetId) => {
                  fieldOnChange(datasetId);
                  resetField('filepath');
                }}
                selectedDatasetId={value ?? ''}
                disabled={disabled}
              />
            )}
          />
          <Controller
            name="filepath"
            control={control}
            render={({ fieldState }) => {
              // Use fieldState.error to ensure proper re-rendering when errors are set
              const errorMessage = fieldState.error?.message || errors.datasetId?.message;
              return (
                <DatasetFileSelect
                  datasetId={datasetId ?? ''}
                  disabled={disabled}
                  errorText={
                    errorMessage ? (
                      <>
                        {errorMessage.endsWith('.') ? errorMessage : `${errorMessage}.`}
                        {slotError && !errorMessage.includes('Unsupported file type') && (
                          <> {slotError}</>
                        )}
                      </>
                    ) : undefined
                  }
                  hideNew
                />
              );
            }}
          />
          <Link to={getWorkspaceFilesetsRoute(workspace)}>
            <Button
              kind="tertiary"
              className="flex items-center gap-2"
              type="button"
              disabled={disabled}
            >
              <Db />
              Manage Datasets
            </Button>
          </Link>
        </Stack>
      ),
    },
  ];
  if (prefer === 'dataset') {
    items.reverse();
  }

  return <Tabs items={items} onValueChange={handleValueChange} />;
};
