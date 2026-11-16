// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  useAgentSearch,
  type AgentSearchProps,
  type UseAgentSearchOptions,
} from '@nemo/common/src/api/agents/useAgentSearch';
import { AgentSelect } from '@nemo/common/src/components/AgentSelect/AgentSelect';
import type {
  MultiAgentSelectProps,
  SingleAgentSelectProps,
} from '@nemo/common/src/components/AgentSelect/types';
import { useState, type FC } from 'react';

type WorkspaceOptions = Pick<UseAgentSearchOptions, 'workspace' | 'filter' | 'sort' | 'include'> & {
  /** Hold the request back for reasons of the caller's own, on top of the open check. */
  enabled?: boolean;
};

export type WorkspaceAgentSelectProps =
  | (Omit<SingleAgentSelectProps, keyof AgentSearchProps> & WorkspaceOptions)
  | (Omit<MultiAgentSelectProps, keyof AgentSearchProps> & WorkspaceOptions);

/**
 * `AgentSelect` wired to its own paged search: nothing is requested until the menu opens, the
 * filter box queries the API by name, and further pages arrive as the list scrolls.
 */
export const WorkspaceAgentSelect: FC<WorkspaceAgentSelectProps> = ({
  workspace,
  filter,
  sort,
  include,
  enabled = true,
  onOpenChange,
  ...selectProps
}) => {
  const [open, setOpen] = useState(false);
  const { agents, loading, onSearchChange, onLoadMore, hasMore, isLoadingMore } = useAgentSearch({
    workspace,
    filter,
    sort,
    include,
    enabled: enabled && open,
  });

  return (
    <AgentSelect
      {...selectProps}
      agents={agents}
      loading={loading}
      onSearchChange={onSearchChange}
      onLoadMore={onLoadMore}
      hasMore={hasMore}
      isLoadingMore={isLoadingMore}
      onOpenChange={(nextOpen) => {
        setOpen(nextOpen);
        onOpenChange?.(nextOpen);
      }}
    />
  );
};
