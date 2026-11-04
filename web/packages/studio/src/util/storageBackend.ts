// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';

export type StorageBackend = NonNullable<FilesetOutput['storage']['type']>;

const STORAGE_BACKEND_LABELS: Record<StorageBackend, string> = {
  github: 'GitHub',
  git: 'Git (SSH)',
  huggingface: 'Hugging Face',
  ngc: 'NGC',
  s3: 'S3',
  local: 'Local',
};

export const formatStorageBackendLabel = (
  type: StorageBackend | null | undefined
): string | null => (type ? STORAGE_BACKEND_LABELS[type] : null);

const REPOSITORY_BACKENDS: ReadonlySet<StorageBackend> = new Set(['github', 'git']);

/** Read-only storage pinned to a repository commit, which filesets can share rather than copy. */
export const isRepositoryBackend = (type: StorageBackend | null | undefined): boolean =>
  type != null && REPOSITORY_BACKENDS.has(type);
