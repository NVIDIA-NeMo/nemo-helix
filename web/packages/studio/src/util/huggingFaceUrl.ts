// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HuggingfaceStorageConfigRepoType } from '@nemo/sdk/generated/platform/schema';
import { HF_DEFAULT_ENDPOINT } from '@studio/util/storageConfigFromUrl';

// HuggingFace serves datasets and spaces under their own path; only models sit at the root.
const REPO_PATH_PREFIX: Partial<Record<HuggingfaceStorageConfigRepoType, string>> = {
  dataset: 'datasets/',
  space: 'spaces/',
};

/**
 * The web page for a repo on the Hugging Face host `endpoint` (default: huggingface.co): datasets
 * and spaces live under their own path prefix. A blank endpoint falls back to the default, and
 * trailing slashes on it are ignored.
 */
export function getHuggingFaceRepoUrl(
  repoType: HuggingfaceStorageConfigRepoType | undefined,
  repoId: string,
  endpoint?: string
): string {
  const origin = endpoint?.trim().replace(/\/+$/, '') || HF_DEFAULT_ENDPOINT;
  return `${origin}/${(repoType && REPO_PATH_PREFIX[repoType]) ?? ''}${repoId}`;
}
