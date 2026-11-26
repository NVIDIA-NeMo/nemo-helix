// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  FilesetOutput,
  FilesetOutputsPage,
  FilesListFilesetsParams,
} from '@nemo/sdk/generated/platform/schema';

/**
 * Lists one page of a workspace's filesets. Studio passes the SDK's `filesListFilesets`; a plugin
 * passes `host.sdk.platform.filesListFilesets` so the request runs on Studio's authenticated client.
 */
export type ListFilesets = (
  workspace: string,
  params?: FilesListFilesetsParams,
  signal?: AbortSignal
) => Promise<FilesetOutputsPage>;

/** Fetches one fileset. Studio passes `filesRetrieveFileset`; a plugin passes `host.sdk.platform.filesRetrieveFileset`. */
export type RetrieveFileset = (
  workspace: string,
  name: string,
  signal?: AbortSignal
) => Promise<FilesetOutput>;

// Same shapes as the SDK's `getFilesListFilesetsQueryKey` / `getFilesRetrieveFilesetQueryKey`, so
// Studio's invalidations after creating or deleting a fileset also refresh these queries. Built
// here rather than imported so the plugin bundle doesn't pull in the SDK's fetch client.
export const getFilesetsQueryKey = (workspace: string) =>
  [`/apis/files/v2/workspaces/${workspace}/filesets`] as const;

export const getFilesetQueryKey = (workspace: string, name: string) =>
  [`/apis/files/v2/workspaces/${workspace}/filesets/${name}`] as const;
