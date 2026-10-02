// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QUERY_PREFIX_PLATFORM } from '@nemo/common/src/constants/api';
import type {
  ModelEntity,
  ModelEntitysPage,
  ModelProvidersPage,
  ModelsListModelsParams,
  ModelsListProvidersParams,
} from '@nemo/sdk/generated/platform/schema';

/**
 * Lists one page of a workspace's models. Studio passes the SDK's `modelsListModels`; a plugin
 * passes `host.sdk.platform.modelsListModels` so the request runs on Studio's authenticated client.
 */
export type ListModels = (
  workspace: string,
  params: ModelsListModelsParams
) => Promise<ModelEntitysPage>;

export type ModelWorkspaceGroup = {
  workspace: string;
  models: ModelEntity[];
};

export const getQueryKeyInfiniteModels = (workspace?: string, query?: ModelsListModelsParams) => [
  QUERY_PREFIX_PLATFORM,
  'models',
  'v2',
  'infinite',
  workspace,
  query,
];

export const getNextModelsPageParam = (lastPage: ModelEntitysPage): number | undefined => {
  if (!lastPage.pagination) {
    return undefined;
  }
  const { page, total_pages } = lastPage.pagination;
  if (page < total_pages) {
    return page + 1;
  }
  return undefined;
};

export const buildWorkspaceGroup = (
  workspaceName: string,
  models: ModelEntity[]
): ModelWorkspaceGroup => ({
  workspace: workspaceName,
  models: [...models].sort((a, b) => (a.name ?? '').localeCompare(b.name ?? '')),
});

/** Lists one page of a workspace's model providers, on the same client as {@link ListModels}. */
export type ListProviders = (
  workspace: string,
  params: ModelsListProvidersParams
) => Promise<ModelProvidersPage>;

export const getQueryKeyModelProviderOptions = (
  workspace: string,
  params: ModelsListProvidersParams
) => [QUERY_PREFIX_PLATFORM, 'models', 'v2', 'providers', workspace, params];
