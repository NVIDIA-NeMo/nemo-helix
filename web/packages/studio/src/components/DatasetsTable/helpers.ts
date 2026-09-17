// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type GithubStorageConfig,
  type HuggingfaceStorageConfig,
  type LocalStorageConfig,
  type NGCStorageConfig,
  type S3StorageConfig,
} from '@nemo/sdk/generated/platform/schema';
import { type StorageConfig } from '@studio/components/DatasetsTable/types';
import { type StorageBackend } from '@studio/util/storageBackend';

export function getStorageBackend(storage: StorageConfig | undefined): StorageBackend | null {
  return storage?.type ?? null;
}

export function getStoragePath(storage: StorageConfig | undefined): string | null {
  if (!storage) return null;
  const s = storage as {
    type?: string;
    path?: string;
    org?: string;
    team?: string;
    target?: string;
    repo_id?: string;
    owner?: string;
    repo?: string;
    bucket?: string;
    prefix?: string;
  };
  if (s.type === 'local' && 'path' in storage) {
    return (storage as LocalStorageConfig).path;
  }
  if (s.type === 'ngc' && 'org' in storage && 'team' in storage && 'target' in storage) {
    const ngc = storage as NGCStorageConfig;
    return `${ngc.org}/${ngc.team}/${ngc.target}`;
  }
  if (s.type === 'huggingface' && 'repo_id' in storage) {
    return (storage as HuggingfaceStorageConfig).repo_id;
  }
  if (s.type === 'github' && 'owner' in storage && 'repo' in storage) {
    const github = storage as GithubStorageConfig;
    return github.path
      ? `${github.owner}/${github.repo}/${github.path}`
      : `${github.owner}/${github.repo}`;
  }
  if (s.type === 's3' && 'bucket' in storage) {
    const s3 = storage as S3StorageConfig;
    return s3.prefix ? `${s3.bucket}/${s3.prefix}` : s3.bucket;
  }
  return null;
}
