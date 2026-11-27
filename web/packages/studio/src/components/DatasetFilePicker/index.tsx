// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getFilesetQueryKey } from '@nemo/common/src/api/filesets/filesetPages';
import {
  FilesetFilePicker,
  type FilesetFilePickerProps,
} from '@nemo/common/src/components/FilesetPicker/FilesetFilePicker';
import type { FilesetPickerNoun } from '@nemo/common/src/components/FilesetPicker/types';
import { getEntityReference } from '@nemo/common/src/namedEntity';
import { CreateDatasetPanel } from '@studio/components/CreateDatasetPanel';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useCallback, useState } from 'react';

const DATASET_NOUN: FilesetPickerNoun = { one: 'dataset', many: 'datasets' };

export type DatasetFilePickerProps = Omit<
  FilesetFilePickerProps,
  'purpose' | 'noun' | 'onRequestNew'
>;

/** Picks a dataset and a file in it. "New dataset" opens the Create Dataset panel and selects the result. */
export const DatasetFilePicker: FC<DatasetFilePickerProps> = ({
  workspace,
  onChange,
  ...props
}) => {
  const queryClient = useQueryClient();
  const [createOpen, setCreateOpen] = useState(false);
  // The fileset picker memoizes its rows on this callback, so keep it stable.
  const openCreate = useCallback(() => setCreateOpen(true), []);

  return (
    <>
      <FilesetFilePicker
        {...props}
        workspace={workspace}
        onChange={onChange}
        purpose="dataset"
        noun={DATASET_NOUN}
        onRequestNew={openCreate}
      />
      <CreateDatasetPanel
        open={createOpen}
        workspace={workspace}
        onClose={() => setCreateOpen(false)}
        onCreated={(fileset) => {
          // Seed the picker's lookup so the new name shows at once instead of after a round trip.
          queryClient.setQueryData(getFilesetQueryKey(fileset.workspace, fileset.name), fileset);
          onChange({ filesetRef: getEntityReference(fileset), filePath: null }, { fileset });
        }}
      />
    </>
  );
};
