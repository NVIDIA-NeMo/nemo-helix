// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';

export type StorageBackend = NonNullable<FilesetOutput['storage']['type']>;

const STORAGE_BACKEND_LABELS: Record<StorageBackend, string> = {
  github: 'GitHub',
  huggingface: 'Hugging Face',
  ngc: 'NGC',
  s3: 'S3',
  local: 'Local',
};

export const formatStorageBackendLabel = (
  type: StorageBackend | null | undefined
): string | null => (type ? STORAGE_BACKEND_LABELS[type] : null);
