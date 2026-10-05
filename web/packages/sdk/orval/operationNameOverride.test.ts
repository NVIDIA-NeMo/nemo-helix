// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect } from 'vitest';

import { operationNameOverride } from './operationNameOverride';

describe('operationNameOverride', () => {
  // --- Existing behavior (should not change) ---

  it('create benchmark', () => {
    expect(
      operationNameOverride({
        operationId: 'create_benchmark_apis_evaluation_v2_workspaces__workspace__benchmarks_post',
      })
    ).toBe('evaluationCreateBenchmark');
  });

  it('create benchmark job', () => {
    expect(
      operationNameOverride({
        operationId: 'create_job_apis_evaluation_v2_workspaces__workspace__benchmark_jobs_post',
      })
    ).toBe('evaluationCreateBenchmarkJob');
  });

  it('create metric job', () => {
    expect(
      operationNameOverride({
        operationId: 'create_job_apis_evaluation_v2_workspaces__workspace__metric_jobs_post',
      })
    ).toBe('evaluationCreateMetricJob');
  });

  it('list customization jobs', () => {
    expect(
      operationNameOverride({
        operationId: 'list_jobs_apis_customization_v2_workspaces__workspace__jobs_get',
      })
    ).toBe('customizationListJobs');
  });

  it('create data designer job', () => {
    expect(
      operationNameOverride({
        operationId: 'create_job_apis_data_designer_v2_workspaces__workspace__jobs_post',
      })
    ).toBe('dataDesignerCreateJob');
  });

  it('list workspaces', () => {
    expect(
      operationNameOverride({
        operationId: 'list_workspaces_apis_entities_v2_workspaces_get',
      })
    ).toBe('entitiesListWorkspaces');
  });

  it('gateway proxy get (non-apis)', () => {
    expect(operationNameOverride({ operationId: 'gateway_proxy_get' })).toBe('gatewayProxyGet');
  });

  // --- Flat collection endpoints (should keep original names) ---

  it('list metric job results (flat)', () => {
    expect(
      operationNameOverride({
        operationId:
          'list_metric_job_results_apis_evaluation_v2_workspaces__workspace__metric_job_results_get',
      })
    ).toBe('evaluationListMetricJobResults');
  });

  it('list benchmark job results (flat)', () => {
    expect(
      operationNameOverride({
        operationId:
          'list_benchmark_job_results_apis_evaluation_v2_workspaces__workspace__benchmark_job_results_get',
      })
    ).toBe('evaluationListBenchmarkJobResults');
  });

  it('get metric job result (flat)', () => {
    expect(
      operationNameOverride({
        operationId:
          'get_metric_job_result_apis_evaluation_v2_workspaces__workspace__metric_job_results__result__get',
      })
    ).toBe('evaluationGetMetricJobResult');
  });

  it('get benchmark job result (flat)', () => {
    expect(
      operationNameOverride({
        operationId:
          'get_benchmark_job_result_apis_evaluation_v2_workspaces__workspace__benchmark_job_results__result__get',
      })
    ).toBe('evaluationGetBenchmarkJobResult');
  });

  // --- Sub-resource endpoints (previously collided, should now be unique) ---

  it('list metric job results (sub-resource) — disambiguated', () => {
    expect(
      operationNameOverride({
        operationId:
          'list_job_results_apis_evaluation_v2_workspaces__workspace__metric_jobs__name__results_get',
      })
    ).toBe('evaluationListMetricJobsResults');
  });

  it('list benchmark job results (sub-resource) — disambiguated', () => {
    expect(
      operationNameOverride({
        operationId:
          'list_job_results_apis_evaluation_v2_workspaces__workspace__benchmark_jobs__name__results_get',
      })
    ).toBe('evaluationListBenchmarkJobsResults');
  });

  it('get metric job result (sub-resource) — disambiguated', () => {
    expect(
      operationNameOverride({
        operationId:
          'get_job_result_apis_evaluation_v2_workspaces__workspace__metric_jobs__job__results__name__get',
      })
    ).toBe('evaluationGetMetricJobsResults');
  });

  it('get benchmark job result (sub-resource) — disambiguated', () => {
    expect(
      operationNameOverride({
        operationId:
          'get_job_result_apis_evaluation_v2_workspaces__workspace__benchmark_jobs__job__results__name__get',
      })
    ).toBe('evaluationGetBenchmarkJobsResults');
  });

  // --- Other sub-resource endpoints (should not change) ---

  it('audit list job results (sub-resource, no collision)', () => {
    expect(
      operationNameOverride({
        operationId:
          'list_job_results_apis_audit_v2_workspaces__workspace__jobs__name__results_get',
      })
    ).toBe('auditListJobResults');
  });

  it('audit get job logs (sub-resource, no collision)', () => {
    expect(
      operationNameOverride({
        operationId: 'get_job_logs_apis_audit_v2_workspaces__workspace__jobs__name__logs_get',
      })
    ).toBe('auditGetJobLogs');
  });

  it('audit get job status (sub-resource, no collision)', () => {
    expect(
      operationNameOverride({
        operationId: 'get_job_status_apis_audit_v2_workspaces__workspace__jobs__name__status_get',
      })
    ).toBe('auditGetJobStatus');
  });

  // --- Noun-first job-subtype collections ("jobs_{subtype}") ---
  // Agent jobs and data-designer mount each job type at /jobs/{subtype}, so the
  // action noun comes first and the subtype trails it. Each subtype must yield a
  // distinct name; previously they all collapsed onto "createJob"/"listJobs"/etc.

  it('agents create analyze job', () => {
    expect(
      operationNameOverride({
        operationId: 'create_job_apis_agents_v2_workspaces__workspace__jobs_analyze_post',
      })
    ).toBe('agentsCreateAnalyzeJob');
  });

  it('agents create evaluate job', () => {
    expect(
      operationNameOverride({
        operationId: 'create_job_apis_agents_v2_workspaces__workspace__jobs_evaluate_post',
      })
    ).toBe('agentsCreateEvaluateJob');
  });

  it('agents create evaluate-suite job', () => {
    expect(
      operationNameOverride({
        operationId: 'create_job_apis_agents_v2_workspaces__workspace__jobs_evaluate_suite_post',
      })
    ).toBe('agentsCreateEvaluateSuiteJob');
  });

  it('agents create optimize job', () => {
    expect(
      operationNameOverride({
        operationId: 'create_job_apis_agents_v2_workspaces__workspace__jobs_optimize_post',
      })
    ).toBe('agentsCreateOptimizeJob');
  });

  it('agents list evaluate jobs', () => {
    expect(
      operationNameOverride({
        operationId: 'list_jobs_apis_agents_v2_workspaces__workspace__jobs_evaluate_get',
      })
    ).toBe('agentsListEvaluateJobs');
  });

  it('agents get evaluate-suite job', () => {
    expect(
      operationNameOverride({
        operationId: 'get_job_apis_agents_v2_workspaces__workspace__jobs_evaluate_suite__name__get',
      })
    ).toBe('agentsGetEvaluateSuiteJob');
  });

  it('agents get optimize job logs (sub-resource)', () => {
    expect(
      operationNameOverride({
        operationId:
          'get_job_logs_apis_agents_v2_workspaces__workspace__jobs_optimize__name__logs_get',
      })
    ).toBe('agentsGetOptimizeJobLogs');
  });

  it('agents get evaluate-suite job result (sub-resource)', () => {
    expect(
      operationNameOverride({
        operationId:
          'get_job_result_apis_agents_v2_workspaces__workspace__jobs_evaluate_suite__job__results__name__get',
      })
    ).toBe('agentsGetEvaluateSuiteJobResult');
  });

  it('agents list analyze job results (sub-resource)', () => {
    expect(
      operationNameOverride({
        operationId:
          'list_job_results_apis_agents_v2_workspaces__workspace__jobs_analyze__name__results_get',
      })
    ).toBe('agentsListAnalyzeJobResults');
  });

  it('data-designer create job stays stable (dedupes the "create" subtype)', () => {
    expect(
      operationNameOverride({
        operationId: 'create_job_apis_data_designer_v2_workspaces__workspace__jobs_create_post',
      })
    ).toBe('dataDesignerCreateJob');
  });

  it('data-designer list create jobs', () => {
    expect(
      operationNameOverride({
        operationId: 'list_jobs_apis_data_designer_v2_workspaces__workspace__jobs_create_get',
      })
    ).toBe('dataDesignerListCreateJobs');
  });

  it('data-designer cancel create job', () => {
    expect(
      operationNameOverride({
        operationId:
          'cancel_job_apis_data_designer_v2_workspaces__workspace__jobs_create__name__cancel_post',
      })
    ).toBe('dataDesignerCancelCreateJob');
  });

  it('drops the redundant intake prefix', () => {
    expect(
      operationNameOverride({
        operationId: 'list_entries_apis_intake_v2_workspaces__workspace__entries_get',
      })
    ).toBe('listEntries');
  });

  it('checks collisions after dropping the intake prefix', () => {
    expect(
      operationNameOverride({
        operationId:
          'list_metric_job_results_apis_intake_v2_workspaces__workspace__metric_job_results_get',
      })
    ).toBe('listMetricJobResults');

    expect(
      operationNameOverride({
        operationId:
          'list_job_results_apis_intake_v2_workspaces__workspace__metric_jobs__name__results_get',
      })
    ).toBe('listMetricJobsResults');
  });
});
