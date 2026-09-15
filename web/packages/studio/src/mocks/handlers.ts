// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EvaluateJob } from '@nemo/sdk/generated/evaluator/schema';
import {
  type AnnotationInput,
  HTTPValidationError,
  ModelEntitySortField,
  PlatformJobLogPage,
  PlatformJobResponsesPage,
  Project,
  TraceMetricBucketParam,
} from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { agentOptimizeJobsHandlers } from '@studio/mocks/handlers/agentOptimizeJobs';
import { customizationJobTemplatesHandlers } from '@studio/mocks/handlers/customizationJobTemplates';
import { customizerHandlers } from '@studio/mocks/handlers/customizer';
import { deploymentsHandlers } from '@studio/mocks/handlers/deployments';
import { evaluatorHandlers } from '@studio/mocks/handlers/evaluator';
import { filesetsHandlers } from '@studio/mocks/handlers/filesets';
import { guardrailsHandlers } from '@studio/mocks/handlers/guardrails';
import { insightsHandlers } from '@studio/mocks/handlers/insights';
import { modelsHandlers } from '@studio/mocks/handlers/models';
import { sampleDatasetsHandlers } from '@studio/mocks/handlers/sampleDatasets';
import { secretsHandlers } from '@studio/mocks/handlers/secrets';
import { workspacesHandlers } from '@studio/mocks/handlers/workspaces';
import {
  mockEvaluationSessionsPage,
  mockEvaluationsPage,
  mockExperiment,
  mockExperimentsPage,
} from '@studio/mocks/intake/experiments';
import {
  createMockAnnotation,
  deleteMockAnnotation,
  mockAnnotationsPage,
  mockSessionById,
  mockSpanById,
  mockSpansPage,
  mockTraceById,
  mockTraceMetrics,
  mockTracesPage,
} from '@studio/mocks/intake/telemetry';
import { randomUUID } from 'crypto';
import { http, HttpResponse } from 'msw';

/**
 * Reject a request if it carries a legacy top-level `search` query parameter.
 *
 * The v2 evaluator endpoints dropped the `search` param in favor of a structured
 * `filter` object. Mock handlers enforce the same contract so that a test which
 * accidentally reintroduces `search:` fails loudly rather than silently passing
 * against a tolerant mock.
 */
const rejectLegacySearchParam = (request: Request) => {
  const params = new URL(request.url).searchParams;
  const hasSearch = Array.from(params.keys()).some(
    (key) => key === 'search' || key.startsWith('search[')
  );
  if (!hasSearch) return undefined;
  return HttpResponse.json(
    { detail: "'search' query parameter is no longer supported; use 'filter' instead" },
    { status: 400 }
  );
};

export interface ProjectParams {
  projectId: string;
}

export interface HypermodelParams {
  hypermodelId: string;
}

/** Shared by the agents list and get-by-name handlers so both agree on the same fixtures. */
const mockAgents = (workspace: unknown) => {
  const config = {
    functions: { wiki: { _type: 'wiki_search' }, clock: { _type: 'current_datetime' } },
    llms: {
      llm: {
        _type: 'openai',
        api_key: 'not-used',
        model_name: 'meta-llama-3-1-70b-instruct',
        temperature: 0,
      },
    },
    workflow: {
      _type: 'react_agent',
      tool_names: ['wiki', 'clock'],
      llm_name: 'llm',
      verbose: false,
      parse_agent_response_max_retries: 3,
    },
  };

  return [
    {
      name: 'react-agent',
      workspace,
      description: '',
      created_at: '2026-04-20T10:00:00Z',
      config,
      config_format: 'nat-workflow-v1',
    },
    {
      name: 'react-agent2',
      workspace,
      description: 'Second react agent',
      created_at: '2026-04-22T10:00:00Z',
      config,
      config_format: 'nat-workflow-v1',
    },
  ];
};

/**
 * Happy path handlers for all UI tests. They usually return mock fixtures, like example Hypermodel response objects.
 * Having a single source of happy path MSW handlers is listed in the [MSW docs as a best practice](https://mswjs.io/docs/best-practices/structuring-handlers#handlers-structure),
 * but tests can override these with `server.use`.
 */
export const handlers = [
  ...sampleDatasetsHandlers,

  // Evaluator V2 — fixtures loaded on first use to keep initial handler graph smaller
  http.get(
    `${PLATFORM_BASE_URL}/apis/evaluator/v2/workspaces/:workspace/evaluate/jobs`,
    async ({ request }) => {
      const rejection = rejectLegacySearchParam(request);
      if (rejection) return rejection;
      const { metricEvaluationJobsPage } = await import('@studio/mocks/evaluation/v1/evaluations');
      return HttpResponse.json(metricEvaluationJobsPage);
    }
  ),
  http.post(
    `${PLATFORM_BASE_URL}/apis/evaluator/v2/workspaces/:workspace/evaluate/jobs`,
    async () => {
      const { metricEvaluationJob1 } = await import('@studio/mocks/evaluation/v1/evaluations');
      return HttpResponse.json(metricEvaluationJob1);
    }
  ),
  http.get(
    `${PLATFORM_BASE_URL}/apis/evaluator/v2/workspaces/:workspace/evaluate/jobs/:name`,
    async () => {
      const { metricEvaluationJob1 } = await import('@studio/mocks/evaluation/v1/evaluations');
      return HttpResponse.json(metricEvaluationJob1);
    }
  ),
  http.get(
    `${PLATFORM_BASE_URL}/apis/evaluator/v2/workspaces/:workspace/evaluate/jobs/:name/logs`,
    ({ params }) => {
      const jobName = params.name as string;
      return HttpResponse.json({
        data: [
          {
            timestamp: '2026-02-27T23:17:01.123456',
            job: jobName,
            job_step: 'initialization',
            job_task: 'task-abc123',
            message: 'Starting evaluate job',
          },
          {
            timestamp: '2026-02-27T23:17:02.234567',
            job: jobName,
            job_step: 'dataset-download',
            job_task: 'task-def456',
            message: 'Downloading dataset from fileset',
          },
          {
            timestamp: '2026-02-27T23:17:03.345678',
            job: jobName,
            job_step: 'evaluation',
            job_task: 'task-ghi789',
            message: 'Running evaluation on 150 samples',
          },
        ],
        total: 3,
        next_page: '',
        prev_page: '',
      });
    }
  ),

  // Projects V2 (Platform) — fixtures loaded on first use
  http.get(
    `${PLATFORM_BASE_URL}/apis/entities/v2/workspaces/:workspace/projects`,
    async ({ params: { workspace } }) => {
      const { getProjectsPageForWorkspace } = await import('@studio/mocks/entity-store/projects');
      return HttpResponse.json(
        getProjectsPageForWorkspace(Array.isArray(workspace) ? workspace[0] : workspace)
      );
    }
  ),
  http.post(
    `${PLATFORM_BASE_URL}/apis/entities/v2/workspaces/:workspace/projects`,
    async ({ params: { workspace } }) => {
      const { projects } = await import('@studio/mocks/entity-store/v1/projects');
      const newProject = { ...projects[0], workspace };
      return HttpResponse.json(newProject);
    }
  ),
  http.get<{ workspace: string; name: string }, never, Project | HTTPValidationError>(
    `${PLATFORM_BASE_URL}/apis/entities/v2/workspaces/:workspace/projects/:name`,
    async ({ params: { workspace, name } }) => {
      const { projects: platformProjects, workspace1 } =
        await import('@studio/mocks/entity-store/projects');
      const ws = Array.isArray(workspace) ? workspace[0] : workspace;
      const n = Array.isArray(name) ? name[0] : name;
      const project = platformProjects.find((p) => p.workspace === ws && p.name === n);
      return HttpResponse.json(project ?? workspace1);
    }
  ),
  http.patch(
    `${PLATFORM_BASE_URL}/apis/entities/v2/workspaces/:workspace/projects/:name`,
    async () => {
      const { workspace1 } = await import('@studio/mocks/entity-store/projects');
      return HttpResponse.json(workspace1);
    }
  ),
  http.delete(
    `${PLATFORM_BASE_URL}/apis/entities/v2/workspaces/:workspace/projects/:name`,
    () => new HttpResponse(null, { status: 200 })
  ),

  // Models V2 (Platform) — fixtures loaded on first use
  http.get(
    `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/models`,
    async ({ request, params: { workspace } }) => {
      const { getEntityStoreLlmModels } = await import('@studio/mocks/entity-store/models');
      const url = new URL(request.url);
      let query = {};
      const sort = url.searchParams.get('sort') as ModelEntitySortField;
      if (sort) {
        query = { sort };
      }
      const result = getEntityStoreLlmModels(query);
      const data = result.data.filter((m) => m.workspace === workspace);
      return HttpResponse.json({ data });
    }
  ),
  http.post(`${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/models`, async () => {
    const { getEntityStoreLlmModels } = await import('@studio/mocks/entity-store/models');
    const models = getEntityStoreLlmModels({});
    return HttpResponse.json(models.data[0]);
  }),
  http.get<{ workspace: string; name: string }>(
    `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/models/:name`,
    async ({ params: { workspace, name } }) => {
      const { getEntityStoreLlmModels } = await import('@studio/mocks/entity-store/models');
      const models = getEntityStoreLlmModels({});
      const model = models.data.find((m) => m.workspace === workspace && m.name === name);
      if (!model) {
        return HttpResponse.json({ detail: 'Model not found' }, { status: 404 });
      }
      return HttpResponse.json(model);
    }
  ),
  http.patch(
    `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/models/:name`,
    async ({ params: { workspace, name } }) => {
      const { getEntityStoreLlmModels } = await import('@studio/mocks/entity-store/models');
      const models = getEntityStoreLlmModels({});
      const model = models.data.find((m) => m.workspace === workspace && m.name === name);
      return HttpResponse.json(model || models.data[0]);
    }
  ),
  http.delete(
    `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/models/:name`,
    () => new HttpResponse(null, { status: 200 })
  ),

  // List all datasets (deprecated v1 API - use v2 filesets)
  http.get(`${PLATFORM_BASE_URL}/v1/datasets`, async () => {
    const { datasets } = await import('@studio/mocks/datasets');
    return HttpResponse.json(datasets);
  }),
  // List a single dataset (deprecated v1 API - use v2 filesets)
  http.get(
    `${PLATFORM_BASE_URL}/v1/datasets/:namespace/:name`,
    async ({ params: { namespace, name } }) => {
      const { datasets } = await import('@studio/mocks/datasets');
      const foundDataset = datasets.data.find((d) => d.workspace === namespace && d.name === name);
      if (!foundDataset) {
        return HttpResponse.json({
          id: `${namespace}/${name}`,
          name,
          workspace: namespace,
          description: '',
          purpose: 'dataset',
          storage: { type: 'local', path: '/data' },
          metadata: {},
          custom_fields: {},
          project: 'default',
          created_at: '2024-12-17T16:08:56.880768',
          updated_at: '2024-12-17T16:08:56.880771',
        });
      }
      return HttpResponse.json(foundDataset);
    }
  ),
  // Create a dataset
  http.post(`${PLATFORM_BASE_URL}/v1/datasets`, async () => {
    const { dataset } = await import('@studio/mocks/datasets');
    return HttpResponse.json(dataset);
  }),
  // Delete a dataset
  http.delete(
    `${PLATFORM_BASE_URL}/v1/datasets/:namespace/:name`,
    () => new HttpResponse(null, { status: 200 })
  ),
  // Delete datastore repository (called after dataset deletion)
  http.delete(
    `${PLATFORM_BASE_URL}/v1/hf/api/repos/delete`,
    () => new HttpResponse(null, { status: 200 })
  ),

  // Deployment Management
  http.get(`${PLATFORM_BASE_URL}/v1/deployment/model-deployments`, async () => {
    const { getModelDeploymentsListResponse } =
      await import('@studio/mocks/deployment-management/constants');
    return HttpResponse.json(getModelDeploymentsListResponse);
  }),

  // Jobs V1 (Safe Synthesizer)
  http.get<never, never, PlatformJobResponsesPage>(`${PLATFORM_BASE_URL}/v1/jobs`, () => {
    return HttpResponse.json({
      data: [],
      pagination: {
        page: 1,
        page_size: 25,
        current_page_size: 0,
        total_pages: 0,
        total_results: 0,
      },
    });
  }),
  http.options(`${PLATFORM_BASE_URL}/v1/jobs`, () => new HttpResponse(null, { status: 200 })),

  // Jobs V2 (Platform)
  http.get<never, never, PlatformJobResponsesPage>(
    `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs`,
    () => {
      return HttpResponse.json({
        data: [],
        pagination: {
          page: 1,
          page_size: 25,
          current_page_size: 0,
          total_pages: 0,
          total_results: 0,
        },
      });
    }
  ),
  http.get<never, never, PlatformJobLogPage>(
    `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs/:name/logs`,
    () =>
      HttpResponse.json({
        data: [],
        total: 0,
        next_page: '',
        prev_page: '',
      })
  ),

  // Safe Synthesizer V2
  http.get(`${PLATFORM_BASE_URL}/apis/safe-synthesizer/v2/workspaces/:workspace/jobs`, () =>
    HttpResponse.json({
      data: [],
      pagination: {
        page: 1,
        page_size: 25,
        current_page_size: 0,
        total_pages: 0,
        total_results: 0,
      },
    })
  ),
  http.post(`${PLATFORM_BASE_URL}/apis/safe-synthesizer/v2/workspaces/:workspace/jobs`, () =>
    HttpResponse.json({
      id: randomUUID(),
      name: 'test-safe-synth-job',
      workspace: 'default',
      status: 'created',
    })
  ),
  http.get(`${PLATFORM_BASE_URL}/apis/safe-synthesizer/v2/workspaces/:workspace/jobs/:name`, () =>
    HttpResponse.json({
      id: randomUUID(),
      name: 'test-safe-synth-job',
      workspace: 'default',
      status: 'completed',
    })
  ),
  http.post(
    `${PLATFORM_BASE_URL}/apis/safe-synthesizer/v2/workspaces/:workspace/jobs/:name/cancel`,
    () => new HttpResponse(null, { status: 200 })
  ),

  // Workspaces V2
  http.get(`${PLATFORM_BASE_URL}/apis/entities/v2/workspaces`, () => {
    const workspacesResponse = {
      object: 'list',
      data: [
        {
          name: 'default',
          description: 'Default workspace',
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        },
        {
          name: 'test-namespace',
          description: 'Test workspace for unit tests',
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        },
      ],
      pagination: {
        page: 1,
        page_size: 20,
        current_page_size: 2,
        total_pages: 1,
        total_results: 2,
      },
    };
    return HttpResponse.json(workspacesResponse);
  }),
  http.post(`${PLATFORM_BASE_URL}/apis/entities/v2/workspaces`, async ({ request }) => {
    const body = (await request.json().catch(() => ({}))) as {
      name?: string;
      description?: string;
    };
    return HttpResponse.json({
      id: body?.name ? `id-${body.name}` : 'workspace-uuid',
      name: body?.name ?? 'new-workspace',
      description: body?.description ?? 'New test workspace',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    });
  }),
  http.get(`${PLATFORM_BASE_URL}/apis/entities/v2/workspaces/:name`, ({ params: { name } }) =>
    HttpResponse.json({
      name,
      description: `Workspace ${name}`,
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
    })
  ),
  http.patch(`${PLATFORM_BASE_URL}/apis/entities/v2/workspaces/:name`, ({ params: { name } }) =>
    HttpResponse.json({
      name,
      description: `Updated workspace ${name}`,
      created_at: '2024-01-01T00:00:00Z',
      updated_at: new Date().toISOString(),
    })
  ),
  http.delete(
    `${PLATFORM_BASE_URL}/apis/entities/v2/workspaces/:name`,
    () => new HttpResponse(null, { status: 200 })
  ),

  // Workspace Members V2
  http.get(`${PLATFORM_BASE_URL}/apis/entities/v2/workspaces/:workspace/members`, () =>
    HttpResponse.json({
      object: 'list',
      data: [],
      pagination: {
        page: 1,
        page_size: 20,
        current_page_size: 0,
        total_pages: 0,
        total_results: 0,
      },
    })
  ),
  http.post(
    `${PLATFORM_BASE_URL}/apis/entities/v2/workspaces/:workspace/members`,
    () => new HttpResponse(null, { status: 200 })
  ),

  // Data Designer V2
  http.get(`${PLATFORM_BASE_URL}/apis/data-designer/v2/workspaces/:workspace/jobs/create`, () =>
    HttpResponse.json({
      data: [],
      pagination: {
        page: 1,
        page_size: 25,
        current_page_size: 0,
        total_pages: 0,
        total_results: 0,
      },
    })
  ),
  http.post(`${PLATFORM_BASE_URL}/apis/data-designer/v2/workspaces/:workspace/jobs/create`, () =>
    HttpResponse.json({
      id: randomUUID(),
      name: 'test-data-designer-job',
      workspace: 'default',
      status: 'created',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    })
  ),
  http.get(
    `${PLATFORM_BASE_URL}/apis/data-designer/v2/workspaces/:workspace/jobs/create/:name`,
    () =>
      HttpResponse.json({
        id: randomUUID(),
        name: 'test-data-designer-job',
        workspace: 'default',
        status: 'completed',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
  ),
  http.post(
    `${PLATFORM_BASE_URL}/apis/data-designer/v2/workspaces/:workspace/jobs/create/:name/cancel`,
    () => new HttpResponse(null, { status: 200 })
  ),
  http.get(
    `${PLATFORM_BASE_URL}/apis/data-designer/v2/workspaces/:workspace/jobs/create/:name/logs`,
    () =>
      HttpResponse.json({
        logs: [],
        next_token: null,
      })
  ),

  // Intake
  http.get('*/apis/intake/v2/workspaces/:workspace/sessions/:sessionId', ({ params }) => {
    const session = mockSessionById(String(params['sessionId']));
    return session ? HttpResponse.json(session) : new HttpResponse(null, { status: 404 });
  }),
  http.get('*/apis/intake/v2/workspaces/:workspace/experiments', () =>
    HttpResponse.json(mockExperimentsPage())
  ),
  http.get('*/apis/intake/v2/workspaces/:workspace/experiments/:name', ({ params }) =>
    HttpResponse.json(mockExperiment(String(params['name'])))
  ),
  http.get('*/apis/intake/v2/workspaces/:workspace/evaluations', ({ request }) => {
    const agentName = new URL(request.url).searchParams.get('filter[agent_name]');
    const page = mockEvaluationsPage();
    return HttpResponse.json(
      agentName
        ? { ...page, data: page.data.filter((e) => e.agent_names?.includes(agentName)) }
        : page
    );
  }),
  http.get(
    '*/apis/intake/v2/workspaces/:workspace/evaluations/:name/sessions',
    ({ request, params }) => {
      const testCaseName = new URL(request.url).searchParams.get('filter[test_case_name]');
      return HttpResponse.json(mockEvaluationSessionsPage(String(params['name']), testCaseName));
    }
  ),
  http.get('*/apis/intake/v2/workspaces/:workspace/traces/metrics', ({ request }) => {
    const url = new URL(request.url);
    const bucketParam = url.searchParams.get('bucket');
    const bucket: TraceMetricBucketParam = (
      Object.values(TraceMetricBucketParam) as string[]
    ).includes(bucketParam ?? '')
      ? (bucketParam as TraceMetricBucketParam)
      : 'total';
    const timezone = url.searchParams.get('timezone') ?? 'UTC';
    return HttpResponse.json(mockTraceMetrics(bucket, timezone));
  }),
  http.get('*/apis/intake/v2/workspaces/:workspace/traces', ({ request }) => {
    const url = new URL(request.url);
    const sessionId = url.searchParams.get('filter[session_id]');
    const traceId = url.searchParams.get('filter[id]');
    const sort = url.searchParams.get('sort');
    const data = mockTracesPage.data
      .filter(
        (trace) =>
          (!sessionId || trace.session_id === sessionId) && (!traceId || trace.id === traceId)
      )
      .sort((a, b) =>
        sort === 'started_at'
          ? Date.parse(a.started_at) - Date.parse(b.started_at)
          : Date.parse(b.started_at) - Date.parse(a.started_at)
      );
    return HttpResponse.json({
      ...mockTracesPage,
      data,
      pagination: {
        ...mockTracesPage.pagination,
        current_page_size: data.length,
        total_results: data.length,
        total_pages: data.length > 0 ? 1 : 0,
      },
      filter: sessionId ? { session_id: sessionId } : undefined,
    });
  }),
  http.get('*/apis/intake/v2/workspaces/:workspace/traces/:traceId', ({ params }) => {
    const trace = mockTraceById(String(params['traceId']));
    return trace ? HttpResponse.json(trace) : new HttpResponse(null, { status: 404 });
  }),
  http.get('*/apis/intake/v2/workspaces/:workspace/spans', ({ request }) => {
    const url = new URL(request.url);
    const traceId = url.searchParams.get('filter[trace_id]');
    const sessionId = url.searchParams.get('filter[session_id]');
    const data = mockSpansPage.data.filter(
      (span) =>
        (!traceId || span.trace_id === traceId) && (!sessionId || span.session_id === sessionId)
    );

    return HttpResponse.json({
      ...mockSpansPage,
      data,
      pagination: {
        ...mockSpansPage.pagination,
        current_page_size: data.length,
        total_results: data.length,
        total_pages: data.length > 0 ? 1 : 0,
      },
      filter: traceId || sessionId ? { trace_id: traceId, session_id: sessionId } : undefined,
    });
  }),
  http.get('*/apis/intake/v2/workspaces/:workspace/spans/:spanId', ({ params }) => {
    const span = mockSpanById(String(params['spanId']));
    return span ? HttpResponse.json(span) : new HttpResponse(null, { status: 404 });
  }),
  http.get('*/apis/intake/v2/workspaces/:workspace/annotations', ({ request }) => {
    const url = new URL(request.url);
    const spanId = url.searchParams.get('filter[span_id]') ?? undefined;
    const sessionId = url.searchParams.get('filter[session_id]') ?? undefined;
    const parsedPage = Number(url.searchParams.get('page') ?? '1');
    const parsedPageSize = Number(url.searchParams.get('page_size') ?? '100');
    const page = Number.isInteger(parsedPage) && parsedPage > 0 ? parsedPage : 1;
    const pageSize = Number.isInteger(parsedPageSize) && parsedPageSize > 0 ? parsedPageSize : 100;
    return HttpResponse.json(
      mockAnnotationsPage({
        spanId,
        sessionId,
        page,
        pageSize,
      })
    );
  }),
  http.post('*/apis/intake/v2/workspaces/:workspace/annotations', async ({ request, params }) => {
    const data = (await request.json()) as AnnotationInput;
    return HttpResponse.json(
      createMockAnnotation({
        workspace: String(params['workspace']),
        data,
      })
    );
  }),
  http.delete('*/apis/intake/v2/workspaces/:workspace/annotations/:annotationId', ({ params }) => {
    const deleted = deleteMockAnnotation(String(params['annotationId']));
    return deleted
      ? new HttpResponse(null, { status: 200 })
      : new HttpResponse(null, { status: 404 });
  }),
  // Agents V2
  http.get(
    `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents`,
    ({ params, request }) => {
      const data = mockAgents(params['workspace']);

      const url = new URL(request.url);
      const sort = url.searchParams.get('sort') ?? '-created_at';
      const page = Math.max(Number(url.searchParams.get('page') ?? '1') || 1, 1);
      const pageSize = Math.max(Number(url.searchParams.get('page_size') ?? '50') || 50, 1);

      const desc = sort.startsWith('-');
      const field = desc ? sort.slice(1) : sort;
      const sorted = [...data].sort((a, b) => {
        const av = String((a as Record<string, unknown>)[field] ?? '');
        const bv = String((b as Record<string, unknown>)[field] ?? '');
        const cmp = av.localeCompare(bv);
        return desc ? -cmp : cmp;
      });

      const start = (page - 1) * pageSize;
      const slice = sorted.slice(start, start + pageSize);

      return HttpResponse.json({
        data: slice,
        pagination: {
          page,
          page_size: pageSize,
          current_page_size: slice.length,
          total_pages: Math.max(Math.ceil(sorted.length / pageSize), 1),
          total_results: sorted.length,
        },
        sort,
        filter: null,
      });
    }
  ),
  http.get(
    `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents/:name`,
    ({ params }) => {
      const agent = mockAgents(params['workspace']).find((a) => a.name === params['name']);
      return agent ? HttpResponse.json(agent) : new HttpResponse(null, { status: 404 });
    }
  ),
  http.delete(
    `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents/:name`,
    () => new HttpResponse(null, { status: 204 })
  ),
  http.post(
    `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/deployments`,
    async ({ request, params }) => {
      const body = (await request.json()) as { agent: string; name?: string };
      return HttpResponse.json(
        {
          name: body.name ?? `${body.agent}-${Math.random().toString(36).slice(2, 7)}`,
          workspace: params['workspace'],
          agent: body.agent,
          status: 'pending',
          endpoint: '',
          port: 0,
          error: '',
        },
        { status: 201 }
      );
    }
  ),
  http.get(`${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/deployments`, () =>
    HttpResponse.json({
      data: [
        {
          name: 'rag-agent-prod',
          workspace: 'default',
          agent: 'react-agent',
          status: 'running',
          endpoint: 'https://rag-agent-prod.example.com',
          port: 8080,
          error: '',
        },
        {
          name: 'sql-agent-dev',
          workspace: 'default',
          agent: 'react-agent',
          status: 'stopped',
          endpoint: 'https://sql-agent-dev.example.com',
          port: 8081,
          error: '',
        },
        {
          name: 'chat-agent-staging',
          workspace: 'default',
          agent: 'react-agent2',
          status: 'error',
          endpoint: 'https://chat-agent-staging.example.com',
          port: 8082,
          error: 'Connection timeout',
        },
      ],
    })
  ),
  // Agent evaluation jobs — exposed at the same prefix as the other agents-v2
  // routes so the AgentPanel's "Recent Evaluations" section and the
  // /agents/evaluations route can both render against an empty list in tests.
  http.get(`${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/jobs/evaluate`, () =>
    HttpResponse.json({ data: [], pagination: { total: 0, page: 1, page_size: 50 } })
  ),
  // Single eval job — default to 404 so the detail route's not-found path
  // is testable without per-test handler overrides.
  http.get(`${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/jobs/evaluate/:name`, () =>
    HttpResponse.json({ detail: 'Not found' }, { status: 404 })
  ),
  http.get(
    `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/jobs/evaluate/:name/status`,
    () => HttpResponse.json({ name: '', status: 'unknown' })
  ),
  http.get(
    `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/jobs/evaluate/:name/logs`,
    () => HttpResponse.json({ data: [], total: 0 })
  ),

  ...workspacesHandlers,
  ...customizerHandlers,
  ...deploymentsHandlers,
  ...evaluatorHandlers,
  ...modelsHandlers,
  ...secretsHandlers,
  ...filesetsHandlers,
  ...guardrailsHandlers,
  ...insightsHandlers,
  ...agentOptimizeJobsHandlers,
  ...customizationJobTemplatesHandlers,
];

// Re-export EvaluateJob so consumers of this module that previously relied on
// MetricEvaluationJob can import it from here without touching their own imports.
export type { EvaluateJob };
