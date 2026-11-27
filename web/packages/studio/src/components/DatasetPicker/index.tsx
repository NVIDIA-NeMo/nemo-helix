// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getFilesetQueryKey } from '@nemo/common/src/api/filesets/filesetPages';
import { FilesetPicker, type FilesetPickerProps } from '@nemo/common/src/components/FilesetPicker';
import type { FilesetPickerNoun } from '@nemo/common/src/components/FilesetPicker/types';
import { getEntityReference } from '@nemo/common/src/namedEntity';
import { CreateDatasetPanel } from '@studio/components/CreateDatasetPanel';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useState } from 'react';

const DATASET_NOUN: FilesetPickerNoun = { one: 'dataset', many: 'datasets' };

export type DatasetPickerProps = Omit<FilesetPickerProps, 'purpose' | 'noun' | 'onRequestNew'>;

/** Picks a dataset; its "New dataset" row opens the Create Dataset panel and selects the result. */
export const DatasetPicker: FC<DatasetPickerProps> = ({ workspace, onChange, ...props }) => {
  const queryClient = useQueryClient();
  const [panelOpen, setPanelOpen] = useState(false);

  return (
    <>
      <FilesetPicker
        {...props}
        workspace={workspace}
        onChange={onChange}
        purpose="dataset"
        noun={DATASET_NOUN}
        onRequestNew={() => setPanelOpen(true)}
      />
      <CreateDatasetPanel
        open={panelOpen}
        workspace={workspace}
        onClose={() => setPanelOpen(false)}
        onCreated={(fileset) => {
          // The picker looks up a selection missing from its list; seed that lookup so the new
          // name shows at once instead of after a round trip.
          queryClient.setQueryData(getFilesetQueryKey(fileset.workspace, fileset.name), fileset);
          onChange(getEntityReference(fileset), fileset);
        }}
      />
    </>
  );
};
