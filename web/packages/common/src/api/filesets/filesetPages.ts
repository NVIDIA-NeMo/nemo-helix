// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  FilesetOutput,
  FilesetOutputsPage,
  FilesListFilesetsParams,
} from '@nemo/sdk/generated/platform/schema';

export type ListFilesets = (
  workspace: string,
  params?: FilesListFilesetsParams,
  signal?: AbortSignal
) => Promise<FilesetOutputsPage>;

export type RetrieveFileset = (
  workspace: string,
  name: string,
  signal?: AbortSignal
) => Promise<FilesetOutput>;

// Mirror the SDK keys so Studio's invalidations reach these queries, without bundling the SDK
// fetch client into the plugin surface.
export const getFilesetsQueryKey = (workspace: string) =>
  [`/apis/files/v2/workspaces/${workspace}/filesets`] as const;

export const getFilesetQueryKey = (workspace: string, name: string) =>
  [`/apis/files/v2/workspaces/${workspace}/filesets/${name}`] as const;
