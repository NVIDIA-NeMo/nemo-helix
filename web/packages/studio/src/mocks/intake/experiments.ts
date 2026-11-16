// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  EvaluationResponse,
  EvaluationResponsesPage,
  EvaluationSessionResponse,
  EvaluationSessionResponsesPage,
  ExperimentResponse,
  ExperimentResponsesPage,
} from '@nemo/sdk/generated/platform/schema';

const WORKSPACE = 'default';

/** The experiment group the EvaluationSessionDetailRoute tests navigate under. */
export const mockExperiment = (name: string): ExperimentResponse => ({
  id: `grp_${name}`,
  name,
  workspace: WORKSPACE,
  description: '',
  default_sort: 'started_at',
  evaluation_count: 2,
});

/** Evaluation names that live in the mock group (each fans out one runs query). */
const MOCK_EVALUATION_NAMES = ['my-experiment', 'my-experiment-v2'] as const;

const mockEvaluation = (name: string): EvaluationResponse => ({
  id: `eval_${name}`,
  name,
  workspace: WORKSPACE,
  experiment_ids: ['grp_my-group'],
  dataset_name: 'sample-dataset',
  experiment_group_id: 'grp_my-group',
});

export const mockEvaluationsPage = (): EvaluationResponsesPage => ({
  data: MOCK_EVALUATION_NAMES.map(mockEvaluation),
});

/** The group the mock evaluations belong to, so a caller resolving experiment_ids finds a name. */
export const mockExperimentsPage = (): ExperimentResponsesPage => ({
  data: [{ ...mockExperiment('my-group'), id: 'grp_my-group' }],
});

const mockRun = (
  evaluationName: string,
  sessionId: string,
  testCaseName: string
): EvaluationSessionResponse => ({
  workspace: WORKSPACE,
  evaluation_name: evaluationName,
  session_id: sessionId,
  test_case_name: testCaseName,
  trace_id: `trace-${sessionId}`,
  root_span_id: `${sessionId}-root`,
  started_at: '2026-01-01T00:00:00Z',
  status: 'success',
});

// Runs of a test case, keyed by evaluation name. The primary session
// `session-agent-run-001` lives in `my-experiment` alongside a sibling run, and
// `my-experiment-v2` contributes a third run — so the compare selector has options.
const RUNS_BY_EVALUATION: Record<string, (testCaseName: string) => EvaluationSessionResponse[]> = {
  'my-experiment': (testCaseName) => [
    mockRun('my-experiment', 'session-agent-run-001', testCaseName),
    mockRun('my-experiment', 'session-agent-run-002', testCaseName),
  ],
  'my-experiment-v2': (testCaseName) => [
    mockRun('my-experiment-v2', 'session-agent-run-101', testCaseName),
  ],
};

export const mockEvaluationSessionsPage = (
  evaluationName: string,
  testCaseName: string | null
): EvaluationSessionResponsesPage => ({
  data: testCaseName ? (RUNS_BY_EVALUATION[evaluationName]?.(testCaseName) ?? []) : [],
});
