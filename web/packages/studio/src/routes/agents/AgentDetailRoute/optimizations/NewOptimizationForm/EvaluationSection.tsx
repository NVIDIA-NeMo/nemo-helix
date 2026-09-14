// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { Badge, Banner, Card, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { JudgeModelSelect } from '@studio/components/evaluation/JudgeModelSelect';
import type { OptimizationFormValues } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/formValues';
import type { OptimizationTarget } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/optimizationTargets';
import { type FC, type ReactNode } from 'react';
import { useFormContext } from 'react-hook-form';

const DetailRow: FC<{ label: string; children: ReactNode }> = ({ label, children }) => (
  <Flex align="center" gap="density-md" wrap="wrap">
    <Text kind="body/regular/sm" color="secondary" className="w-24 shrink-0">
      {label}
    </Text>
    {children}
  </Flex>
);

export interface EvaluationSectionProps {
  targets: OptimizationTarget[];
  /** The target for the currently selected experiment, or undefined when there is none to pick. */
  selected?: OptimizationTarget;
  isLoading: boolean;
}

/**
 * Which evaluation supplies the yardstick, and which model scores against it.
 *
 * The prompts and expected answers are read out of the selected experiment's latest run — an
 * optimization with nothing to score against cannot rank its trials, which is why an agent with no
 * evaluations gets a blocking banner instead of a picker. The judge is asked for separately because
 * the study re-scores every trial itself rather than reusing published numbers.
 */
export const EvaluationSection: FC<EvaluationSectionProps> = ({ targets, selected, isLoading }) => {
  const { control } = useFormContext<OptimizationFormValues>();

  if (!isLoading && targets.length === 0) {
    return (
      <Banner kind="inline" status="warning">
        This agent has no published evaluations yet. Run one first — the study needs a dataset and
        evaluators to score its trials against.
      </Banner>
    );
  }

  return (
    <Stack gap="density-md">
      <ControlledSelect
        useControllerProps={{ control, name: 'experimentId' }}
        loading={isLoading}
        placeholder="Select an experiment"
        items={targets.map((target) => ({
          value: target.experimentId,
          children: target.experimentName ?? target.experimentId,
        }))}
      />

      {selected && (
        <Card>
          <Stack gap="density-sm" className="w-full">
            <DetailRow label="Loaded from">
              <Text kind="body/regular/sm">{selected.evaluation.name}</Text>
              {selected.evaluation.created_at && (
                <Text kind="body/regular/xs" color="secondary">
                  latest evaluation in this experiment,{' '}
                  <RelativeTime datetime={selected.evaluation.created_at} />
                </Text>
              )}
            </DetailRow>
            <DetailRow label="Dataset">
              <Text kind="body/regular/sm">{selected.evaluation.dataset_name || '—'}</Text>
            </DetailRow>
            <DetailRow label="Metrics">
              {selected.evaluators.length > 0 ? (
                selected.evaluators.map((evaluator) => (
                  <Badge key={evaluator} color="gray" kind="outline">
                    {evaluator}
                  </Badge>
                ))
              ) : (
                <Text kind="body/regular/sm" color="secondary">
                  None published yet
                </Text>
              )}
            </DetailRow>
          </Stack>
        </Card>
      )}

      <JudgeModelSelect<OptimizationFormValues>
        formFieldName="judgeModel"
        required
        slotLabel="Judge model"
        placeholder="Select a model to score trials with"
      />

      <Text kind="body/regular/xs" color="secondary">
        The evaluation's prompts and expected answers are staged with the study and re-scored by the
        judge above, so trial scores compare to each other rather than to the numbers this
        evaluation published. The experiment id is written to metadata.experiment_id, so trials sit
        alongside its evaluations.
      </Text>
    </Stack>
  );
};
