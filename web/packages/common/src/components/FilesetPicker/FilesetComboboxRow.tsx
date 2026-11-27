// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetStorageIcon } from '@nemo/common/src/components/FilesetPicker/FilesetStorageIcon';
import { PURPOSE_LABELS } from '@nemo/common/src/utils/filesetPurpose';
import { formatStorageBackendLabel } from '@nemo/common/src/utils/storageBackend';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface FilesetComboboxRowProps {
  fileset: FilesetOutput;
  showPurpose: boolean;
}

export const FilesetComboboxRow: FC<FilesetComboboxRowProps> = ({ fileset, showPurpose }) => {
  const storageLabel = formatStorageBackendLabel(fileset.storage?.type);
  return (
    <span className="flex w-full min-w-0 items-center gap-2">
      <FilesetStorageIcon fileset={fileset} />
      <span className="min-w-0 flex-1">
        <Text kind="body/regular/sm" className="block truncate text-primary">
          {fileset.name}
        </Text>
        {showPurpose ? (
          <Text kind="body/regular/xs" className="block text-secondary">
            {PURPOSE_LABELS[fileset.purpose]}
          </Text>
        ) : null}
      </span>
      {storageLabel ? (
        <Text kind="body/regular/xs" className="shrink-0 text-right text-secondary">
          {storageLabel}
        </Text>
      ) : null}
    </span>
  );
};
