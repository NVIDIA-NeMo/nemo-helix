// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useUploadModalContext } from '@nemo/common/src/components/UploadModal/Context/useUploadModalContext';
import { DatasetUploader } from '@nemo/common/src/components/UploadModal/DatasetUploader';
import { FileUpload } from '@nemo/common/src/components/UploadModal/FileUpload';
import { SimpleFilesTable } from '@nemo/common/src/components/UploadModal/SimpleFilesTable';
import {
  Stack,
  TabsContent,
  TabsList,
  TabsRoot,
  TabsTrigger,
} from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface UploadPickerBodyProps {
  workspace: string;
  /** If true, shows dataset selection UI. If false, only shows file upload. */
  includeDataset?: boolean;
  /** If true, renders both dataset and file upload options in separate tabs. */
  includeTabs?: boolean;
}

/** Body of the upload picker — the picker UI without modal chrome. Shared
 *  between {@link UploadModal} and {@link InlineUploadPicker}. */
export const UploadPickerBody: FC<UploadPickerBodyProps> = ({
  workspace,
  includeDataset = false,
  includeTabs = false,
}) => {
  const [state, dispatch] = useUploadModalContext();
  const { files, activeTab } = state;

  if (includeTabs) {
    return (
      <TabsRoot
        className="mt-4"
        value={activeTab}
        onValueChange={(value) =>
          dispatch({ type: 'SET_TAB', payload: value as 'dataset' | 'file' })
        }
      >
        <TabsList className="w-full">
          <TabsTrigger value="dataset">Select from Dataset</TabsTrigger>
          <TabsTrigger value="file">Upload a File</TabsTrigger>
        </TabsList>
        <TabsContent value="dataset" className="min-h-0 w-full px-0">
          <DatasetUploader projectId={workspace} />
        </TabsContent>
        <TabsContent value="file" className="min-h-0 w-full px-0">
          {files.length === 0 ? <FileUpload /> : <SimpleFilesTable />}
        </TabsContent>
      </TabsRoot>
    );
  }

  if (includeDataset) {
    return (
      <Stack gap="density-md" className="min-h-0 w-full">
        <DatasetUploader projectId={workspace} />
      </Stack>
    );
  }

  return (
    <Stack gap="density-md" className="min-h-0 w-full">
      {files.length === 0 ? <FileUpload /> : <SimpleFilesTable />}
    </Stack>
  );
};
