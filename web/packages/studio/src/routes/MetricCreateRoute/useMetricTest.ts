// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { evalsRunEvaluateLiveEvaluation } from '@nemo/sdk/generated/evals/evals-plugin-live-evaluation-route';
import { describeFailure } from '@studio/components/evaluation/ModelEvaluationWizard/useLiveTest';
import { buildMetric } from '@studio/routes/MetricCreateRoute/buildMetric';
import type { MetricFormValues } from '@studio/routes/MetricCreateRoute/types';
import { useCallback, useRef, useState } from 'react';

export interface SampleRow {
  input: string;
  reference: string;
  context: string;
  output: string;
}

export interface MetricTestScore {
  name: string;
  value: number;
  label?: string;
}

export type MetricTestState =
  | { status: 'idle' }
  | { status: 'busy' }
  | { status: 'done'; scores: MetricTestScore[]; error?: string }
  | { status: 'error'; message: string };

const rubricLabel = (values: MetricFormValues, name: string, value: number) => {
  const score = values.body.scores.find((entry) => entry.name === name.split('.').pop());
  return score?.scoreType === 'rubric'
    ? score.rubric.find((level) => level.value === value)?.label
    : undefined;
};

/** Scores one hand-written row offline: the row carries the response as
 *  `output`, so nothing is generated and nothing is saved. */
export function useMetricTest(workspace: string) {
  const [state, setState] = useState<MetricTestState>({ status: 'idle' });
  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState({ status: 'idle' });
  }, []);

  const run = useCallback(
    async (values: MetricFormValues, sample: SampleRow) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setState({ status: 'busy' });

      const row = Object.fromEntries(Object.entries(sample).filter(([, value]) => value));
      try {
        const response = await evalsRunEvaluateLiveEvaluation(
          workspace,
          { dataset: [row], metrics: [buildMetric(values, workspace)] },
          controller.signal
        );
        if (controller.signal.aborted) return;
        const [metric] = response.metrics;
        setState({
          status: 'done',
          // Scores come back as `<metric type>.<output>`; the type is already on screen.
          scores: (metric?.scores ?? []).map((score) => {
            const value = score.mean ?? Number.NaN;
            const name = score.name.replace(`${metric?.metric}.`, '');
            return { name, value, label: rubricLabel(values, name, value) };
          }),
          error: metric?.error ?? undefined,
        });
      } catch (error) {
        if (controller.signal.aborted) return;
        setState({ status: 'error', message: describeFailure(error) });
      }
    },
    [workspace]
  );

  return { state, run, reset };
}
