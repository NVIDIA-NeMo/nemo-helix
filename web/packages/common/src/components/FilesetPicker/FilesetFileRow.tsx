// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatFileSize } from '@nemo/common/src/components/UploadModal/utils';
import { IconDocument } from '@nemo/icons/icons';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface FilesetFileRowProps {
  file: FilesetFileOutput;
  /** Why the file can't be picked. Shown as text: a disabled option never gets hover tooltips. */
  ineligibleReason?: string;
}

export const FilesetFileRow: FC<FilesetFileRowProps> = ({ file, ineligibleReason }) => (
  <span className="flex w-full min-w-0 items-center gap-2">
    <IconDocument size="16" className="shrink-0 text-secondary" aria-hidden="true" />
    <Text kind="body/regular/sm" className="min-w-0 flex-1 truncate text-primary">
      {file.path} ({formatFileSize(file.size)})
    </Text>
    {ineligibleReason ? (
      <Text kind="body/regular/xs" className="shrink-0 text-secondary">
        {ineligibleReason}
      </Text>
    ) : null}
  </span>
);
