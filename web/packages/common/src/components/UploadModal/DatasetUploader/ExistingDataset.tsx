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
import { Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import { FC } from 'react';

/**
 * Component for handling file selection from an existing dataset.
 * Uses UploadModalContext for state management.
 */
export const ExistingDataset: FC = () => {
  const [state] = useUploadModalContext();
  const { dataset, selectedFiles, errors, files, isFetching } = state;

  if (dataset && dataset.type === 'new') {
    return null;
  }

  if (isFetching) {
    return <Spinner slotDescription="Loading dataset files..." />;
  }

  // If the dataset has no files and no file is selected, show file upload
  if (files.length === 0 && selectedFiles.length === 0) {
    return (
      <Stack gap="density-md">
        <Text kind="label/bold/sm">File</Text>
        <Text kind="label/regular/sm">There are no files found in this dataset.</Text>
        <FileUpload error={errors?.file} />
      </Stack>
    );
  }

  // Show dataset files
  return (
    <Stack gap="density-md" className="min-h-0 flex-1 overflow-hidden">
      <Text kind="label/bold/sm">Files</Text>
      <SimpleFilesTable />
    </Stack>
  );
};
