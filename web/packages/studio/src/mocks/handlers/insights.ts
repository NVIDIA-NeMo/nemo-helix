// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getAgentsListExecuteJobsQueryKey } from '@nemo/sdk/generated/agents/agents';
import { getAgentsListPluginManagedAgentTypesRouteQueryKey } from '@nemo/sdk/generated/agents/plugin-managed-agent-types';
import type {
  ExecuteJob,
  ExecuteJobRequest,
  PluginManagedAgentTypeInfo,
} from '@nemo/sdk/generated/agents/schema';
import { getCompassGetAnalysisConfigQueryKey } from '@nemo/sdk/generated/compass/compass-analysis-configs';
import { getCompassGetStatusesAnalysisRunStatusQueryKey } from '@nemo/sdk/generated/compass/compass-analysis-run-statuses';
import { getCompassListInsightsQueryKey } from '@nemo/sdk/generated/compass/compass-insights';
import type { InsightListItem } from '@nemo/sdk/generated/compass/schema';
import { getListEvaluationsQueryKey } from '@nemo/sdk/generated/platform/evaluations';
import type { EvaluationResponse, HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { COMPASS_AGENT_TYPE } from '@studio/api/insightsAnalysis';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { http, HttpResponse } from 'msw';

const INSIGHTS_URL = mockApiUrl(getCompassListInsightsQueryKey, ':workspace');
const ANALYSIS_CONFIG_URL = mockApiUrl(getCompassGetAnalysisConfigQueryKey, ':workspace', ':agent');
const EXECUTE_JOBS_URL = mockApiUrl(getAgentsListExecuteJobsQueryKey, ':workspace');
const AGENT_TYPES_URL = mockApiUrl(getAgentsListPluginManagedAgentTypesRouteQueryKey, ':workspace');
const ANALYSIS_RUN_STATUS_URL = mockApiUrl(
  getCompassGetStatusesAnalysisRunStatusQueryKey,
  ':workspace',
  ':agent'
);

export const MOCK_COMPASS_AGENT = 'compass-analyst';

/** A Compass analysis of `agent`: an execute job on the Compass agent whose subject is `agent`. */
export const mockAnalysisJob = (
  agent: string,
  status: HelixJobStatus,
  {
    name = 'analysis-run-1',
    created_at = '2026-08-14T09:00:00Z',
    workspace = 'default',
    context = {
      agent,
      default_model: mockAnalysisConfig.default_model,
      fast_model: mockAnalysisConfig.fast_model,
    } as Record<string, unknown>,
  } = {}
): ExecuteJob => ({
  name,
  workspace,
  created_at,
  updated_at: created_at,
  status,
  spec: {
    request: { agent: MOCK_COMPASS_AGENT, context },
    agent: {
      name: MOCK_COMPASS_AGENT,
      workspace,
      config: {},
      config_format: 'nemo-agents-spec-v1',
    },
    input: 'Analyze the target agent.',
    agent_ref: `${workspace}/${MOCK_COMPASS_AGENT}`,
    context,
    subject: agent,
    plugin_managed_agent_type: COMPASS_AGENT_TYPE,
  },
});

/** The JSON `filter` param as field -> value, unwrapping `{ $eq: value }` operators. */
const parseJobFilter = (request: Request): Record<string, unknown> => {
  const raw = new URL(request.url).searchParams.get('filter');
  if (!raw) {
    return {};
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') {
      return {};
    }
    return Object.fromEntries(
      Object.entries(parsed).map(([field, value]: [string, unknown]) => [
        field,
        value && typeof value === 'object' && '$eq' in value ? value.$eq : value,
      ])
    );
  } catch {
    return {};
  }
};

/**
 * Serves `jobs()` as the execute jobs list, newest first, reading it per request so a test
 * can advance a job between polls. Honors the type and subject filters Studio sends.
 */
export const analysisJobHandlers = (jobs: () => ExecuteJob[]) => [
  http.get(EXECUTE_JOBS_URL, ({ request }) => {
    const filter = parseJobFilter(request);
    const pageSize = Number(new URL(request.url).searchParams.get('page_size') ?? 20);
    const matches = jobs().filter(
      ({ spec }) =>
        (filter['spec.plugin_managed_agent_type'] === undefined ||
          spec.plugin_managed_agent_type === filter['spec.plugin_managed_agent_type']) &&
        (filter['spec.subject'] === undefined || spec.subject === filter['spec.subject'])
    );
    const data = matches.slice(0, pageSize);
    return HttpResponse.json({
      data,
      pagination: {
        page: 1,
        page_size: pageSize,
        current_page_size: data.length,
        total_pages: matches.length > 0 ? 1 : 0,
        total_results: matches.length,
      },
    });
  }),
];

/** The installed Compass type, with `agent` as the workspace's Compass agent, if it has one. */
export const mockCompassAgentType = (agent?: string): PluginManagedAgentTypeInfo => ({
  type: COMPASS_AGENT_TYPE,
  title: 'Compass analyst',
  description: '',
  default_name: MOCK_COMPASS_AGENT,
  adapter_id: 'nvidia.helix.compass.analyst',
  context_schema: {},
  post_run_actions: ['compass.save-insights'],
  subject_label: 'Analyzed agent',
  ...(agent ? { agent } : {}),
});

/** The scheduler's run status for `agent`, or a 404 when it has never run. */
export const analysisRunStatusHandler = (agent: string, lastSuccessfulRunAt: string | null) =>
  http.get<{ agent: string }>(ANALYSIS_RUN_STATUS_URL, ({ params }) =>
    lastSuccessfulRunAt && params.agent === agent
      ? HttpResponse.json({
          id: `insights-analysis-run-status-${agent}`,
          name: agent,
          agent,
          status: 'idle',
          last_successful_run_at: lastSuccessfulRunAt,
        })
      : HttpResponse.json({ detail: 'Not Found' }, { status: 404 })
  );

/** Answers execute job creation and hands each request body and created job to `onCreate`. */
export const analysisJobCreateHandler = (
  onCreate: (body: ExecuteJobRequest, job: ExecuteJob) => void
) =>
  http.post<{ workspace: string }, ExecuteJobRequest>(
    EXECUTE_JOBS_URL,
    async ({ params, request }) => {
      const body = await request.json();
      const context = { ...(body.spec.context ?? {}) };
      const subject = typeof context.agent === 'string' ? context.agent : '';
      const job = mockAnalysisJob(subject, 'created', { workspace: params.workspace, context });
      onCreate(body, job);
      return HttpResponse.json(job);
    }
  );

/** Intake evaluations named `names` that recorded traces for `agent`. */
export const agentEvaluationsHandler = (agent: string, names: string[]) =>
  http.get(mockApiUrl(getListEvaluationsQueryKey, ':workspace'), ({ request }) => {
    const filtered = new URL(request.url).searchParams.get('filter[agent_name]');
    const data: EvaluationResponse[] =
      !filtered || filtered === agent
        ? names.map((name) => ({
            id: `experiment-${name}`,
            name,
            workspace: 'default',
            experiment_ids: [],
            experiment_group_id: '',
            dataset_name: 'sample-dataset',
            agent_names: [agent],
          }))
        : [];
    return HttpResponse.json({ data });
  });

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
    evidence: Array.from({ length: 12 }, (_, index) => ({ trace_id: `trace-a-${index}` })),
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
    evidence: Array.from({ length: 5 }, (_, index) => ({ trace_id: `trace-b-${index}` })),
    experiment_group_count: 0,
    last_seen_at: '2026-08-12T09:00:00Z',
    created_at: '2026-08-11T09:00:00Z',
    created_by: 'user@example.com',
    updated_at: '2026-08-12T09:00:00Z',
    updated_by: 'user@example.com',
  },
];

export const insightsHandlers = [
  ...analysisJobHandlers(() => []),

  http.get(AGENT_TYPES_URL, () => HttpResponse.json([mockCompassAgentType(MOCK_COMPASS_AGENT)])),

  analysisRunStatusHandler('', null),

  analysisJobCreateHandler(() => undefined),

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
