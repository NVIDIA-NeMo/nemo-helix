// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ListModels, ListProviders } from '@nemo/common/src/api/models/modelPages';
import {
  useModelSearchCore,
  type ModelSearchFilter,
  type ModelSearchProps,
} from '@nemo/common/src/api/models/useModelSearchCore';
import { ModelSelect } from '@nemo/common/src/components/ModelSelectV2/ModelSelect';
import type { ModelSelectProps } from '@nemo/common/src/components/ModelSelectV2/types';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { useState, type FC } from 'react';

export interface WorkspaceModelSelectCoreProps extends Omit<
  ModelSelectProps,
  keyof ModelSearchProps
> {
  /** Workspace to search. Nothing is fetched while this is null. */
  workspace: string | null;
  listModels: ListModels;
  /** Fills the provider filter's options. */
  listProviders?: ListProviders;
  /** Extra filters merged into the search request (e.g. `lora_enabled`). */
  filter?: ModelSearchFilter;
  /** Client-side predicate for conditions the API cannot express (e.g. "has a deployment"). */
  include?: (model: ModelEntity) => boolean;
  /** Hold the request back for reasons of the caller's own, on top of the open check. */
  enabled?: boolean;
}

/**
 * `ModelSelect` wired to its own paged search: nothing is requested until the user opens the
 * menu, the filter box queries the API, and further pages arrive as the list scrolls.
 *
 * Takes its fetcher so plugins can run it on `host.sdk`; Studio code uses `WorkspaceModelSelect`.
 * Reach for the search hook directly only when the caller needs the models themselves — to merge
 * in a pinned entry, or to surface the query error.
 */
export const WorkspaceModelSelectCore: FC<WorkspaceModelSelectCoreProps> = ({
  workspace,
  listModels,
  listProviders,
  filter,
  include,
  enabled = true,
  onOpenChange,
  ...selectProps
}) => {
  const [open, setOpen] = useState(false);
  const { groups, loading, onSearchChange, onLoadMore, hasMore, isLoadingMore, modelFilters } =
    useModelSearchCore({
      workspace,
      listModels,
      listProviders,
      filter,
      include,
      enabled: enabled && open,
    });

  const handleOpenChange = (nextOpen: boolean) => {
    setOpen(nextOpen);
    onOpenChange?.(nextOpen);
  };

  return (
    <ModelSelect
      {...selectProps}
      groups={groups}
      loading={loading}
      onSearchChange={onSearchChange}
      onLoadMore={onLoadMore}
      hasMore={hasMore}
      isLoadingMore={isLoadingMore}
      modelFilters={modelFilters}
      onOpenChange={handleOpenChange}
    />
  );
};
