// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentEvaluateJob } from '@nemo/sdk/generated/evals/schema';
import type { EvaluationSessionResponse } from '@nemo/sdk/generated/platform/schema';
import type { AgentEvalResult } from '@studio/api/evaluation/agent-evaluations';
import {
  agentEvalExport,
  customizationMetricsExport,
  datasetEvalExport,
  evaluationSessionsExport,
} from '@studio/util/resultExports';

const scores = [{ name: 'judge.accuracy', mean: 0.5, count: 2 }];

describe('datasetEvalExport', () => {
  it('flattens rows with a column per metric output', () => {
    const { json, jsonl, csv } = datasetEvalExport({ name: 'job', status: 'completed' }, scores, [
      {
        row_index: 0,
        item: { label: 'yes' },
        sample: { output_text: 'yes' },
        requests: [{ request: { messages: [{ content: 'q?' }] } }],
        metrics: { judge: [{ name: 'accuracy', value: 1 }] },
      },
    ]);
    expect(json).toMatchObject({ name: 'job', scores });
    expect(jsonl).toEqual([expect.objectContaining({ item: { label: 'yes' } })]);
    expect(csv).toEqual([
      { row: 0, input: 'q?', expected: 'yes', response: 'yes', 'judge.accuracy': 1 },
    ]);
  });

  it('falls back to aggregate scores when there are no rows', () => {
    const { csv, jsonl } = datasetEvalExport({ name: 'job' }, scores, []);
    expect(csv).toEqual(scores);
    expect(jsonl).toEqual(scores);
  });
});

describe('agentEvalExport', () => {
  const job = {
    name: 'job',
    workspace: 'ws',
    status: 'completed',
    spec: { target: { kind: 'agent', agent: { name: 'ws/bot' } } },
  } as unknown as AgentEvaluateJob;
  const result = { bundle_ref: 'ws/fs#run', summary: { tasks: 1 } } as unknown as AgentEvalResult;

  it('keeps the full job and result records in JSON', () => {
    expect(agentEvalExport(job, result, scores, []).json).toEqual({
      job,
      agent: 'bot',
      result,
      trials: [],
    });
  });

  it('writes one row per trial with its scores', () => {
    const { csv } = agentEvalExport(job, result, scores, [
      {
        taskId: 't1',
        trialId: 'tr1',
        intent: 'classify',
        instruction: 'email',
        reference: { label: 'phishing' },
        status: 'success',
        responseText: 'phishing',
        scores: [{ name: 'judge.accuracy', value: 1 }],
        diagnostics: [],
      },
    ]);
    expect(csv).toEqual([
      {
        task_id: 't1',
        trial_id: 'tr1',
        intent: 'classify',
        instruction: 'email',
        reference: { label: 'phishing' },
        status: 'success',
        response: 'phishing',
        'judge.accuracy': 1,
        diagnostics: undefined,
      },
    ]);
  });
});

describe('evaluationSessionsExport', () => {
  it('spreads evaluator scores into columns and marks failed evaluators', () => {
    const session = {
      session_id: 's1',
      test_case_name: 'case',
      status: 'success',
      evaluator_scores: { relevance: 0.9 },
      failed_evaluators: ['toxicity'],
    } as unknown as EvaluationSessionResponse;
    expect(evaluationSessionsExport({}, [session]).csv).toEqual([
      {
        session_id: 's1',
        test_case_name: 'case',
        status: 'success',
        relevance: 0.9,
        toxicity: 'FAILED',
      },
    ]);
  });
});

describe('customizationMetricsExport', () => {
  it('joins metric series by step, sorted', () => {
    const { csv } = customizationMetricsExport(
      { name: 'job' },
      {
        train_loss: [
          { step: 2, value: 0.5, epoch: 1 },
          { step: 1, value: 0.9, epoch: 1 },
        ],
        val_loss: [{ step: 2, value: 0.6 }],
      }
    );
    expect(csv).toEqual([
      { step: 1, epoch: 1, train_loss: 0.9 },
      { step: 2, epoch: 1, train_loss: 0.5, val_loss: 0.6 },
    ]);
  });
});
