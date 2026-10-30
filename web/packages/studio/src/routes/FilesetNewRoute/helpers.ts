// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage as getApiErrorMessage } from '@nemo/common/src/api/common/utils';
import { isHuggingFaceUrl, isNgcUrl } from '@studio/util/storageConfigFromUrl';

/**
 * User-facing error for external storage create failure, with a stable prefix per storage type
 * so the toast never shows raw [object Object] from API detail.
 */
export function getExternalStorageCreateErrorMessage(err: unknown, externalUrl: string): string {
  let prefix: string;
  try {
    const parsed = new URL(externalUrl);
    if (isNgcUrl(parsed)) {
      prefix = 'Failed to create fileset from NGC. ';
    } else if (isHuggingFaceUrl(parsed)) {
      prefix = 'Failed to create fileset from Hugging Face. ';
    } else {
      prefix = 'Failed to create fileset from external storage. ';
    }
  } catch {
    prefix = 'Failed to create fileset from external storage. ';
  }
  const detail =
    err && typeof err === 'object'
      ? getApiErrorMessage(err as Error, 'Please check your URL and credentials.')
      : 'Please check your URL and credentials.';
  return prefix + detail;
}

/** Normalize form files (may be File[] or KUI Upload's FileUploadItem[]) to File[]. */
export function toFileList(value: unknown): File[] {
  if (!value) return [];
  const arr = Array.isArray(value) ? value : [value];
  return arr.flatMap((item) =>
    item instanceof File
      ? [item]
      : (item as { file?: File }).file
        ? [(item as { file: File }).file]
        : []
  );
}
