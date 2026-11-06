// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { WithFilterOperators } from '@nemo/common/src/api/filterOperators';
import { agentsListAgents, getAgentsListAgentsQueryKey } from '@nemo/sdk/generated/agents/agents';
import type {
  Agent,
  AgentFilter,
  AgentsListAgentsParams,
  NemoListResponseAgent,
} from '@nemo/sdk/generated/agents/schema';
import { useInfiniteQuery } from '@tanstack/react-query';
import { useCallback, useEffect, useMemo, useState } from 'react';

/** Small on purpose: the dropdown pulls the next page as the user scrolls. */
export const AGENT_SEARCH_PAGE_SIZE = 25;

export type AgentSearchFilter = WithFilterOperators<AgentFilter & { name?: string }>;

export interface UseAgentSearchOptions {
  /** Workspace to search. The query stays idle while this is null. */
  workspace: string | null;
  /** Extra filters merged into the request (e.g. `config_format`, `labels`). */
  filter?: AgentSearchFilter;
  sort?: string;
  pageSize?: number;
  enabled?: boolean;
  /**
   * Client-side predicate for conditions the API cannot express. The hook keeps paging while a
   * page filters down to nothing, so an excluded page never stalls the list.
   */
  include?: (agent: Agent) => boolean;
}

/** Props for `AgentSelect`, ready to spread. */
export interface AgentSearchProps {
  agents: Agent[];
  loading: boolean;
  onSearchChange: (search: string) => void;
  onLoadMore: () => Promise<void>;
  hasMore: boolean;
  isLoadingMore: boolean;
}

export interface UseAgentSearchResult extends AgentSearchProps {
  search: string;
  error: Error | null;
}

/**
 * Server-side agent search with progressive paging. The cache key extends
 * `getAgentsListAgentsQueryKey(workspace)`, so invalidating the agents list (after a create,
 * clone, or delete) refreshes open pickers too.
 */
export const useAgentSearch = ({
  workspace,
  filter,
  sort = 'name',
  pageSize = AGENT_SEARCH_PAGE_SIZE,
  enabled = true,
  include,
}: UseAgentSearchOptions): UseAgentSearchResult => {
  const [search, setSearch] = useState('');

  const params = useMemo<AgentsListAgentsParams>(() => {
    const trimmed = search.trim();
    const merged: AgentSearchFilter = {
      ...filter,
      ...(trimmed ? { name: { $like: trimmed } } : {}),
    };
    return {
      page_size: pageSize,
      sort,
      ...(Object.keys(merged).length > 0 ? { filter: merged as AgentFilter } : {}),
    };
  }, [filter, pageSize, search, sort]);

  const isEnabled = enabled && !!workspace;
  const { data, error, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } =
    useInfiniteQuery({
      queryKey: [...getAgentsListAgentsQueryKey(workspace ?? '', params), 'infinite'],
      queryFn: ({ pageParam, signal }) =>
        agentsListAgents(workspace ?? '', { ...params, page: pageParam }, signal),
      getNextPageParam: ({ pagination }: NemoListResponseAgent) =>
        pagination && pagination.page < pagination.total_pages ? pagination.page + 1 : undefined,
      initialPageParam: 1,
      enabled: isEnabled,
    });

  const agents = useMemo(() => {
    const loaded = data?.pages.flatMap((page) => page.data ?? []) ?? [];
    return include ? loaded.filter(include) : loaded;
  }, [data?.pages, include]);

  const onLoadMore = useCallback(async () => {
    if (!hasNextPage || isFetchingNextPage) return;
    await fetchNextPage();
  }, [fetchNextPage, hasNextPage, isFetchingNextPage]);

  // `include` can empty a whole page, leaving no rows to scroll and so no way to ask for more.
  useEffect(() => {
    if (
      isEnabled &&
      agents.length === 0 &&
      hasNextPage &&
      !isFetchingNextPage &&
      !isLoading &&
      !error
    ) {
      void fetchNextPage();
    }
  }, [agents.length, error, fetchNextPage, hasNextPage, isEnabled, isFetchingNextPage, isLoading]);

  return {
    agents,
    search,
    error,
    loading: isEnabled && isLoading,
    onSearchChange: setSearch,
    onLoadMore,
    hasMore: !!hasNextPage,
    isLoadingMore: isFetchingNextPage,
  };
};
