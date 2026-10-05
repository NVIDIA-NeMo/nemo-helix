// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getJobRefetchInterval } from '@nemo/common/src/utils/query';
import type { HelixJobResponse } from '@nemo/sdk/generated/platform/schema';
import { getJobDetailRoute } from '@studio/components/dataViews/JobsDataView/utils';
import { JOB_PROGRESS_JOB_TYPE } from '@studio/routes/agents/AssistantChatRoute/jobProgressConsts';
import {
  getJobProgressDetailRoute,
  getJobProgressRefetchInterval,
  getStringArg,
  isJobProgressNotFoundError,
} from '@studio/routes/agents/AssistantChatRoute/utils/jobProgress';
import { getAgentEvaluationDetailRoute } from '@studio/routes/utils';

vi.mock('@nemo/common/src/utils/query', () => ({
  getJobRefetchInterval: vi.fn(),
}));

vi.mock('@studio/components/dataViews/JobsDataView/utils', () => ({
  getJobDetailRoute: vi.fn(),
}));

vi.mock('@studio/routes/utils', () => ({
  getAgentEvaluationDetailRoute: vi.fn(),
}));

const getJobRefetchIntervalMock = vi.mocked(getJobRefetchInterval);
const getJobDetailRouteMock = vi.mocked(getJobDetailRoute);
const getAgentEvaluationDetailRouteMock = vi.mocked(getAgentEvaluationDetailRoute);

const createJob = (overrides: Partial<HelixJobResponse> = {}): HelixJobResponse =>
  ({
    attempt_id: 'attempt-1',
    fileset: 'fileset-1',
    id: 'job-id-1',
    name: 'studio-job-1',
    platform_spec: {},
    source: 'platform',
    status: 'active',
    workspace: 'default',
    ...overrides,
  }) as HelixJobResponse;

beforeEach(() => {
  getJobRefetchIntervalMock.mockReset().mockReturnValue(5_000);
  getJobDetailRouteMock.mockReset().mockReturnValue('/jobs/studio-job-1');
  getAgentEvaluationDetailRouteMock
    .mockReset()
    .mockReturnValue('/workspaces/default/agents/evaluations/eval-job-1');
});

describe('getStringArg', () => {
  it('returns trimmed string arguments and ignores empty or non-string values', () => {
    expect(getStringArg({ title: '  Job title  ' }, 'title')).toBe('Job title');
    expect(getStringArg({ title: '  ' }, 'title')).toBeUndefined();
    expect(getStringArg({ title: 42 }, 'title')).toBeUndefined();
  });
});

describe('isJobProgressNotFoundError', () => {
  it('detects 404 errors from either top-level or response status fields', () => {
    expect(isJobProgressNotFoundError({ status: 404 })).toBe(true);
    expect(isJobProgressNotFoundError({ response: { status: 404 } })).toBe(true);
  });

  it('ignores non-404 errors', () => {
    expect(isJobProgressNotFoundError(new Error('boom'))).toBe(false);
    expect(isJobProgressNotFoundError({ response: { status: 500 } })).toBe(false);
    expect(isJobProgressNotFoundError(undefined)).toBe(false);
  });
});

describe('getJobProgressRefetchInterval', () => {
  it('stops polling when the job is missing', () => {
    expect(getJobProgressRefetchInterval({ jobMissing: true, status: 'active' })).toBe(false);
    expect(getJobRefetchIntervalMock).not.toHaveBeenCalled();
  });

  it('delegates active job polling intervals to the shared job helper', () => {
    expect(getJobProgressRefetchInterval({ jobMissing: false, status: 'active' })).toBe(5_000);
    expect(getJobRefetchIntervalMock).toHaveBeenCalledWith('active');
  });
});

describe('getJobProgressDetailRoute', () => {
  it('links agent evaluation jobs to the agent evaluation detail page', () => {
    expect(
      getJobProgressDetailRoute({
        jobName: 'eval-job-1',
        jobType: JOB_PROGRESS_JOB_TYPE.AGENT_EVALUATION,
        workspace: 'default',
      })
    ).toBe('/workspaces/default/agents/evaluations/eval-job-1');

    expect(getAgentEvaluationDetailRouteMock).toHaveBeenCalledWith('default', 'eval-job-1');
  });

  it('uses the Jobs table route helper for platform job sources', () => {
    const job = createJob({ name: 'designer-job-1', source: 'data-designer' });

    expect(
      getJobProgressDetailRoute({
        job,
        jobName: 'designer-job-1',
        workspace: 'default',
      })
    ).toBe('/jobs/studio-job-1');

    expect(getJobDetailRouteMock).toHaveBeenCalledWith(
      { name: 'designer-job-1', source: 'data-designer' },
      'default'
    );
  });

  it('maps known job_type values to Jobs table sources when a job has not loaded yet', () => {
    getJobProgressDetailRoute({
      jobName: 'safe-job-1',
      jobType: JOB_PROGRESS_JOB_TYPE.SAFE_SYNTHESIZER,
      workspace: 'default',
    });

    expect(getJobDetailRouteMock).toHaveBeenCalledWith(
      { name: 'safe-job-1', source: 'safe-synthesizer' },
      'default'
    );
  });
});
