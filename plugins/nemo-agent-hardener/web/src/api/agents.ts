// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useHost } from '@agent-hardener/host';
import { useQuery } from '@tanstack/react-query';


const AGENTS_PAGE_SIZE = 100;

/**
 * Every agent in the workspace, for the create form's agent picker.
 *
 * The picker needs the full list rather than a page, so this walks pagination
 * itself. The client comes off `host.sdk.agents` — Studio's own instance, on its
 * authenticated axios and shared cache — so the plugin neither bundles the
 * agents SDK nor re-implements its auth.
 */
export const useAgentsForSelect = (workspace: string) => {
  const { sdk } = useHost();
  return useQuery({
    queryKey: ['agent-hardener-init', 'agents', workspace],
    queryFn: async ({ signal }) => {
      const all: Awaited<ReturnType<typeof sdk.agents.agentsListAgents>>['data'] = [];
      let page = 1;

      while (true) {
        const response = await sdk.agents.agentsListAgents(
          workspace,
          { page, page_size: AGENTS_PAGE_SIZE, sort: 'name' },
          signal
        );
        const batch = response.data ?? [];
        all.push(...batch);

        const totalPages = response.pagination?.total_pages;
        if (totalPages ? page >= totalPages : batch.length < AGENTS_PAGE_SIZE) break;
        page += 1;
      }

      return all;
    },
    enabled: !!workspace,
  });
};
