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
import { ExistingDataset } from '@nemo/common/src/components/UploadModal/DatasetUploader/ExistingDataset';
import { NewDataset } from '@nemo/common/src/components/UploadModal/DatasetUploader/NewDataset';
import { DatasetSelect } from '@nemo/common/src/components/UploadModal/DatasetUploader/Select';
import { Stack } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface Props {
  projectId: string;
}

/**
 * A component that allows selecting a dataset and uploading files.
 * Uses UploadModalContext for state management.
 */
export const DatasetUploader: FC<Props> = ({ projectId }) => {
  const [state] = useUploadModalContext();
  const { dataset, errors } = state;

  return (
    <Stack gap="density-xl" className="min-h-0 w-full flex-1">
      <DatasetSelect project={projectId} error={errors?.dataset} />
      {dataset && (dataset.type === 'new' ? <NewDataset /> : <ExistingDataset />)}
    </Stack>
  );
};
