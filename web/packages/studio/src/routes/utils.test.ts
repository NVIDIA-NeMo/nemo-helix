// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getEvaluationBenchmarkDetailsRoute,
  getEvaluationBenchmarkListRoute,
  getEvaluationMetricDetailsRoute,
  getEvaluationMetricRunRoute,
  getEvaluationMetricsRunRoute,
  getEvaluationSessionDetailRoute,
  getFilesetDetailsRoute,
  getFilesetRoute,
  getIntakeSessionRoute,
  getIntakeSessionTraceRoute,
  getIntakeTracesRoute,
  getWorkspaceBaseModelsRoute,
  getWorkspaceInferenceProvidersRoute,
} from '@studio/routes/utils';

describe('Evaluation route helpers', () => {
  const workspace = 'test-workspace';

  describe('getEvaluationMetricDetailsRoute', () => {
    it('should generate correct evaluation metric details URL', () => {
      const jobId = 'job-123';

      const result = getEvaluationMetricDetailsRoute(workspace, jobId);

      expect(result).toBe('/workspaces/test-workspace/evaluation/metrics/job-123');
    });
  });

  describe('getEvaluationMetricsRunRoute', () => {
    it('appends an encoded model query param when a model is provided', () => {
      expect(getEvaluationMetricsRunRoute(workspace, { model: 'test-namespace/model-a' })).toBe(
        '/workspaces/test-workspace/evaluation/metrics/run?model=test-namespace%2Fmodel-a'
      );
    });
  });

  describe('getEvaluationMetricRunRoute', () => {
    it('appends an encoded model query param when a metric and model are provided', () => {
      expect(
        getEvaluationMetricRunRoute(workspace, 'toxicity', {
          model: 'test-namespace/model-a',
        })
      ).toBe(
        '/workspaces/test-workspace/evaluation/metrics/toxicity/run?model=test-namespace%2Fmodel-a'
      );
    });
  });

  describe('getEvaluationBenchmarkListRoute', () => {
    it('should generate the benchmarks list URL', () => {
      expect(getEvaluationBenchmarkListRoute(workspace)).toBe(
        '/workspaces/test-workspace/evaluation/benchmarks'
      );
    });
  });

  describe('getEvaluationBenchmarkDetailsRoute', () => {
    it('should generate a benchmark details URL', () => {
      expect(getEvaluationBenchmarkDetailsRoute(workspace, 'my-benchmark')).toBe(
        '/workspaces/test-workspace/evaluation/benchmarks/my-benchmark'
      );
    });
  });
});

describe('getFilesetRoute', () => {
  const workspace = 'test-workspace';

  it('routes a bare name to the panel route with a full entity reference', () => {
    expect(getFilesetRoute(workspace, 'unchanged-blue')).toBe(
      '/workspaces/test-workspace/filesets/test-workspace%2Funchanged-blue'
    );
  });

  it('preserves a namespace already present on the reference', () => {
    expect(getFilesetRoute(workspace, 'default/unchanged-blue')).toBe(
      '/workspaces/test-workspace/filesets/default%2Funchanged-blue'
    );
  });
});

describe('getWorkspaceInferenceProvidersRoute', () => {
  const workspace = 'test-workspace';

  it('returns base inference providers path when no options are given', () => {
    expect(getWorkspaceInferenceProvidersRoute(workspace)).toBe(
      '/workspaces/test-workspace/inference-providers'
    );
  });

  it('appends create=true and preset query params when a preset is provided', () => {
    expect(getWorkspaceInferenceProvidersRoute(workspace, { preset: 'build' })).toBe(
      '/workspaces/test-workspace/inference-providers?create=true&preset=build'
    );
  });
});

describe('getWorkspaceBaseModelsRoute (deep linking)', () => {
  const workspace = 'my-workspace';

  it('returns base models list path when no options', () => {
    expect(getWorkspaceBaseModelsRoute(workspace)).toBe('/workspaces/my-workspace/base-models');
  });

  it('encodes model names with special characters (e.g. slash) for the path', () => {
    expect(getWorkspaceBaseModelsRoute(workspace, { model: 'org/my-model' })).toBe(
      '/workspaces/my-workspace/base-models/org%2Fmy-model'
    );
  });

  it('appends tab query param when both model and tab are provided', () => {
    expect(
      getWorkspaceBaseModelsRoute(workspace, {
        model: 'my-model',
        tab: 'chat-playground',
      })
    ).toBe('/workspaces/my-workspace/base-models/my-model?tab=chat-playground');
  });

  it('preserves provided query params on model detail paths', () => {
    const searchParams = new URLSearchParams({
      s: 'llama',
      filters: JSON.stringify([{ id: 'customizable', value: { fine_tunable: true } }]),
      sort: '-created_at',
    });

    expect(getWorkspaceBaseModelsRoute(workspace, { model: 'my-model', searchParams })).toBe(
      `/workspaces/my-workspace/base-models/my-model?${searchParams.toString()}`
    );
  });

  it('combines provided query params with tab query param', () => {
    const searchParams = new URLSearchParams({ s: 'llama' });

    expect(
      getWorkspaceBaseModelsRoute(workspace, {
        model: 'my-model',
        tab: 'chat-playground',
        searchParams,
      })
    ).toBe('/workspaces/my-workspace/base-models/my-model?s=llama&tab=chat-playground');
  });

  it('preserves provided query params on base models list paths', () => {
    const searchParams = new URLSearchParams({ s: 'llama', sort: '-created_at' });

    expect(getWorkspaceBaseModelsRoute(workspace, { searchParams })).toBe(
      '/workspaces/my-workspace/base-models?s=llama&sort=-created_at'
    );
  });
});

describe('intake session detail routes', () => {
  it('builds session, trace, and span links', () => {
    expect(getIntakeSessionRoute('my-workspace', 'session-1')).toBe(
      '/workspaces/my-workspace/intake/sessions/session-1'
    );
    expect(getIntakeSessionTraceRoute('my-workspace', 'session-1', 'trace-1')).toBe(
      '/workspaces/my-workspace/intake/sessions/session-1?traceId=trace-1'
    );
    expect(
      getIntakeSessionTraceRoute('my-workspace', 'session-1', 'trace-1', { spanId: 'span-1' })
    ).toBe('/workspaces/my-workspace/intake/sessions/session-1?traceId=trace-1&spanId=span-1');
  });

  it('encodes session IDs in Intake and Evaluation paths', () => {
    expect(getIntakeSessionRoute('my-workspace', 'session / 1')).toBe(
      '/workspaces/my-workspace/intake/sessions/session%20%2F%201'
    );
    expect(
      getEvaluationSessionDetailRoute(
        'my-workspace',
        'experiment-group',
        'evaluation',
        'session / 1'
      )
    ).toBe(
      '/workspaces/my-workspace/experiment/experiment-group/evaluation/sessions/session%20%2F%201'
    );
  });
});

describe('getFilesetDetailsRoute', () => {
  // Callers must pass raw values: the helper encodes the path param via generatePath and
  // the folder query param via URLSearchParams. Pre-encoding would double-encode, so a
  // fileset named `a/b` would arrive at useParams() as `a%2Fb` instead of `a/b`.
  it.each([
    ['default/my-fileset', 'default%2Fmy-fileset'],
    ['default/100% coverage', 'default%2F100%25%20coverage'],
    ['default/tag#1', 'default%2Ftag%231'],
    ['default/with spaces', 'default%2Fwith%20spaces'],
  ])('encodes %j exactly once', (filesetId, encoded) => {
    expect(getFilesetDetailsRoute('my-workspace', filesetId)).toBe(
      `/workspaces/my-workspace/filesets/${encoded}`
    );
    // The route param round-trips back to the original name.
    expect(decodeURIComponent(encoded)).toBe(filesetId);
  });

  it('encodes the folder query param without double-encoding', () => {
    expect(getFilesetDetailsRoute('my-workspace', 'default/set', 'nested/folder 1')).toBe(
      '/workspaces/my-workspace/filesets/default%2Fset?filesetFolder=nested%2Ffolder+1'
    );
  });
});

describe('getIntakeTracesRoute', () => {
  it('links to the unfiltered traces list by default', () => {
    expect(getIntakeTracesRoute('default')).toBe('/workspaces/default/intake/traces');
  });

  it('encodes an agent filter the traces table reads back', () => {
    const url = new URL(
      getIntakeTracesRoute('ws-a', { agentName: 'email-security-triage' }),
      'http://x'
    );
    expect(url.pathname).toBe('/workspaces/ws-a/intake/traces');
    expect(JSON.parse(decodeURIComponent(url.searchParams.get('filters') ?? ''))).toEqual([
      { id: 'agent_name', value: 'email-security-triage' },
    ]);
  });
});
