// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAgentDeployments } from '@studio/api/agents/useAgentDeployments';
import { AGENT_DEPLOYMENTS_URL, agentDeployment } from '@studio/mocks/handlers/agentDeployments';
import { server } from '@studio/mocks/node';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

describe('useAgentDeployments', () => {
  it("collects every page of the agent's deployments", async () => {
    const queries: URLSearchParams[] = [];
    server.use(
      http.get(AGENT_DEPLOYMENTS_URL, ({ request }) => {
        const query = new URL(request.url).searchParams;
        queries.push(query);
        const page = Number(query.get('page'));
        return HttpResponse.json({
          data: [agentDeployment('my-agent', `my-agent-${page}`)],
          pagination: { page, total_pages: 2 },
        });
      })
    );

    const { result } = renderHook(
      () => useAgentDeployments({ workspace: 'default', agent: 'my-agent', refetchInterval: 0 }),
      { wrapper: TestProviders }
    );

    await waitFor(() =>
      expect(result.current.deployments.map((d) => d.name)).toEqual(['my-agent-1', 'my-agent-2'])
    );
    expect(queries.map((query) => query.get('filter[agent]'))).toEqual(['my-agent', 'my-agent']);
  });

  it('asks for nothing without an agent', () => {
    const { result } = renderHook(
      () => useAgentDeployments({ workspace: 'default', refetchInterval: 0 }),
      { wrapper: TestProviders }
    );

    expect(result.current).toEqual({ deployments: [], isLoading: false, isUnavailable: false });
  });

  it('reports the list as unavailable when it cannot be loaded', async () => {
    server.use(http.get(AGENT_DEPLOYMENTS_URL, () => HttpResponse.json({}, { status: 500 })));
    const { result } = renderHook(
      () => useAgentDeployments({ workspace: 'default', agent: 'my-agent', refetchInterval: 0 }),
      { wrapper: TestProviders }
    );

    await waitFor(() => expect(result.current.isUnavailable).toBe(true));
    expect(result.current.deployments).toEqual([]);
  });
});
