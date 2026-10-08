// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { Badge, Banner, Card, Flex, Stack, Text } from '@nvidia/foundations-react-core';
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
 * Which evaluation supplies the prompts and accuracy rubric.
 *
 * The selected experiment's latest run supplies its dataset and LLM judge. Each trial is scored
 * afresh using its rubric adapted to the optimization backend's custom scoring format.
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
    <Stack gap="density-md" className="w-full">
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
        <Card className="w-full">
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
    </Stack>
  );
};
