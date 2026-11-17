// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HuggingfaceStorageConfigRepoType } from '@nemo/sdk/generated/platform/schema';

/** The huggingface.co web page for a repo: datasets and spaces live under their own path prefix. */
export function getHuggingFaceRepoUrl(
  repoType: HuggingfaceStorageConfigRepoType | undefined,
  repoId: string
): string {
  void repoType;
  void repoId;
  throw new Error('not implemented');
}
