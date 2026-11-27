// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getStorageBackendIcon } from '@nemo/common/src/utils/storageBackend';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import type { FC } from 'react';

interface FilesetStorageIconProps {
  fileset: FilesetOutput;
}

export const FilesetStorageIcon: FC<FilesetStorageIconProps> = ({ fileset }) => {
  const Icon = getStorageBackendIcon(fileset.storage?.type);
  return <Icon size="16" className="shrink-0 text-secondary" aria-hidden="true" />;
};
