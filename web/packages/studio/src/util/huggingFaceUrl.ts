// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HuggingfaceStorageConfigRepoType } from '@nemo/sdk/generated/platform/schema';
import { HF_DEFAULT_ENDPOINT } from '@studio/util/storageConfigFromUrl';

// HuggingFace serves datasets and spaces under their own path; only models sit at the root.
const REPO_PATH_PREFIX: Partial<Record<HuggingfaceStorageConfigRepoType, string>> = {
  dataset: 'datasets/',
  space: 'spaces/',
};

/** The huggingface.co web page for a repo: datasets and spaces live under their own path prefix. */
export function getHuggingFaceRepoUrl(
  repoType: HuggingfaceStorageConfigRepoType | undefined,
  repoId: string,
  endpoint?: string
): string {
  void endpoint; // not honoured yet
  return `${HF_DEFAULT_ENDPOINT}/${(repoType && REPO_PATH_PREFIX[repoType]) ?? ''}${repoId}`;
}
