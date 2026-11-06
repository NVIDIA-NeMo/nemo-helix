// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  useModelSearch,
  type ModelSearchFilter,
  type ModelSearchProps,
} from '@nemo/common/src/api/models/useModelSearch';
import { ModelSelectV2 } from '@nemo/common/src/components/ModelSelectV2/ModelSelectV2';
import type { ModelSelectV2Props } from '@nemo/common/src/components/ModelSelectV2/types';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { useState, type FC } from 'react';

export interface WorkspaceModelSelectProps extends Omit<
  ModelSelectV2Props,
  keyof ModelSearchProps
> {
  /** Workspace to search. Nothing is fetched while this is null. */
  workspace: string | null;
  /** Extra filters merged into the search request (e.g. `lora_enabled`). */
  filter?: ModelSearchFilter;
  /** Client-side predicate for conditions the API cannot express (e.g. "has a deployment"). */
  include?: (model: ModelEntity) => boolean;
  /** Hold the request back for reasons of the caller's own, on top of the open check. */
  enabled?: boolean;
}

/**
 * `ModelSelectV2` wired to its own paged search: nothing is requested until the user opens the
 * menu, the filter box queries the API, and further pages arrive as the list scrolls.
 *
 * Reach for `useModelSearch` directly only when the caller needs the models themselves — to merge
 * in a pinned entry, or to surface the query error.
 */
export const WorkspaceModelSelect: FC<WorkspaceModelSelectProps> = ({
  workspace,
  filter,
  include,
  enabled = true,
  onOpenChange,
  ...selectProps
}) => {
  const [open, setOpen] = useState(false);
  const { groups, loading, onSearchChange, onLoadMore, hasMore, isLoadingMore, modelFilters } =
    useModelSearch({
      workspace,
      filter,
      include,
      enabled: enabled && open,
    });

  const handleOpenChange = (nextOpen: boolean) => {
    setOpen(nextOpen);
    onOpenChange?.(nextOpen);
  };

  return (
    <ModelSelectV2
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
