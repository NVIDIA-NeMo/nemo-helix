/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
import { useUploadModalContext } from '@nemo/common/src/components/UploadModal/Context/useUploadModalContext';
import { FileUpload } from '@nemo/common/src/components/UploadModal/FileUpload';
import { SimpleFilesTable } from '@nemo/common/src/components/UploadModal/SimpleFilesTable';
import { IconError } from '@nemo/icons/icons';
import { Flex, FormField, Stack, Text, TextInput } from '@nvidia/foundations-react-core';
import { FC } from 'react';

/**
 * Component for creating a new dataset and uploading files to it.
 * Uses UploadModalContext for state management.
 */
export const NewDataset: FC = () => {
  const [state, dispatch] = useUploadModalContext();
  const { dataset, files, errors } = state;
  // If the selected dataset is an existing dataset, return null
  if (dataset && dataset.type === 'existing') {
    return null;
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    dispatch({
      type: 'UPDATE_DATASET',
      payload: {
        type: 'new',
        name: e.target.value,
      },
    });
  };

  return (
    <>
      <FormField
        slotLabel="Dataset Name"
        slotError={
          <Flex gap="density-md" align="center">
            <IconError className="text-feedback-danger" />
            <Text kind="label/regular/sm" className="text-feedback-danger">
              {errors?.datasetName}
            </Text>
          </Flex>
        }
        status={errors?.datasetName ? 'error' : undefined}
      >
        {({ status }) => (
          <TextInput
            status={status}
            placeholder="Name this Dataset"
            value={dataset?.name || ''}
            onChange={handleChange}
          />
        )}
      </FormField>
      {files.length === 0 ? (
        <Stack gap="density-md">
          <Text kind="label/bold/sm">File</Text>
          <FileUpload error={errors?.file} />
        </Stack>
      ) : (
        <SimpleFilesTable />
      )}
    </>
  );
};
