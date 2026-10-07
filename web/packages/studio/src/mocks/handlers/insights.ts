// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getInsightsGetAnalysisConfigQueryKey } from '@nemo/sdk/generated/insights/insights-analysis-configs';
import { getInsightsListAnalysisRunsQueryKey } from '@nemo/sdk/generated/insights/insights-analysis-runs';
import { getInsightsListInsightsQueryKey } from '@nemo/sdk/generated/insights/insights-insights';
import type {
  AnalysisRunResponse,
  CreateAnalysisRunRequest,
  InsightListItem,
} from '@nemo/sdk/generated/insights/schema';
import { getFilesDownloadFileQueryKey } from '@nemo/sdk/generated/platform/files';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import {
  AGENT_ETHOS_FILE,
  agentSpecFilesetName,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
import { http, HttpResponse } from 'msw';

const INSIGHTS_URL = mockApiUrl(getInsightsListInsightsQueryKey, ':workspace');
const ANALYSIS_CONFIG_URL = mockApiUrl(
  getInsightsGetAnalysisConfigQueryKey,
  ':workspace',
  ':agent'
);
const ANALYSIS_RUNS_URL = mockApiUrl(getInsightsListAnalysisRunsQueryKey, ':workspace');

export const mockAnalysisRunResponse = (
  workspace: string,
  { agent, default_model, fast_model }: CreateAnalysisRunRequest
): AnalysisRunResponse => ({
  run: {
    id: 'analysis-run-1',
    entity_id: 'analysis-run-1',
    name: 'analysis-run-1',
    workspace,
    agent,
    default_model,
    fast_model,
    parent: `ws-${workspace}`,
    db_version: 1,
    created_at: '2026-08-14T09:00:00Z',
    created_by: 'user@example.com',
    updated_at: '2026-08-14T09:00:00Z',
    updated_by: 'user@example.com',
  },
  job: { name: 'analysis-run-1', status: 'created' },
});

/** Serves `content` as the agent's ETHOS.md, or 404s the file when `content` is null. */
export const agentEthosHandlers = (agent: string, content: string | null) => {
  const url = mockApiUrl(
    getFilesDownloadFileQueryKey,
    ':workspace',
    agentSpecFilesetName(agent),
    AGENT_ETHOS_FILE
  );
  const notFound = () => HttpResponse.json({ detail: 'Not Found' }, { status: 404 });
  return [
    http.head(url, () =>
      content === null
        ? new HttpResponse(null, { status: 404 })
        : new HttpResponse(null, {
            headers: { 'Content-Length': String(new TextEncoder().encode(content).length) },
          })
    ),
    http.get(url, () => (content === null ? notFound() : HttpResponse.text(content))),
  ];
};

/** Stored per-agent analysis config, as the agent Insights tab's analysis panel reads it. */
export const mockAnalysisConfig = {
  id: 'insights-analysis-config-1',
  name: 'react-agent',
  agent: 'react-agent',
  enabled: true,
  default_model: 'default/nvidia-nemotron-mini-4b-instruct',
  fast_model: 'default/nvidia-nemotron-mini-4b-instruct',
  updated_at: '2026-08-14T09:00:00Z',
};

export const mockInsights: InsightListItem[] = [
  {
    id: 'ins-1',
    entity_id: 'ins-1',
    parent: 'ws-default',
    db_version: 1,
    name: 'ambiguous-system-prompt',
    workspace: 'default',
    title: 'Ambiguous system prompt causes tool misfires',
    description:
      'The system prompt does not say which tool owns order lookups, so the agent calls the search tool for questions the orders tool answers.',
    agent: 'react-agent',
    status: 'open',
    trace_refs: Array.from({ length: 12 }, (_, index) => `trace-a-${index}`),
    experiment_group_count: 1,
    last_seen_at: '2026-08-14T09:00:00Z',
    created_at: '2026-08-10T09:00:00Z',
    created_by: 'user@example.com',
    updated_at: '2026-08-14T09:00:00Z',
    updated_by: 'user@example.com',
  },
  {
    id: 'ins-2',
    entity_id: 'ins-2',
    parent: 'ws-default',
    db_version: 1,
    name: 'latency-long-context',
    workspace: 'default',
    title: 'Latency spikes on long context (>8k tokens)',
    description:
      'Sessions whose accumulated context passes ~8k tokens take more than three times as long to return.',
    agent: 'react-agent',
    status: 'open',
    trace_refs: Array.from({ length: 5 }, (_, index) => `trace-b-${index}`),
    experiment_group_count: 0,
    last_seen_at: '2026-08-12T09:00:00Z',
    created_at: '2026-08-11T09:00:00Z',
    created_by: 'user@example.com',
    updated_at: '2026-08-12T09:00:00Z',
    updated_by: 'user@example.com',
  },
];

export const insightsHandlers = [
  http.post<{ workspace: string }, CreateAnalysisRunRequest>(
    ANALYSIS_RUNS_URL,
    async ({ params, request }) =>
      HttpResponse.json(mockAnalysisRunResponse(params.workspace, await request.json()))
  ),

  http.get(ANALYSIS_CONFIG_URL, ({ params }) =>
    HttpResponse.json({ ...mockAnalysisConfig, name: params.agent, agent: params.agent })
  ),

  http.get(INSIGHTS_URL, ({ request }) => {
    const params = new URL(request.url).searchParams;
    const agent = params.get('agent');
    const status = params.get('status');

    const matches = mockInsights.filter(
      (insight) => (!agent || insight.agent === agent) && (!status || insight.status === status)
    );

    const page = Number(params.get('page') ?? 1);
    const pageSize = Number(params.get('page_size') ?? 20);
    const data = matches.slice((page - 1) * pageSize, page * pageSize);

    return HttpResponse.json({
      data,
      pagination: {
        page,
        page_size: pageSize,
        current_page_size: data.length,
        total_pages: Math.ceil(matches.length / pageSize),
        total_results: matches.length,
      },
    });
  }),
];
