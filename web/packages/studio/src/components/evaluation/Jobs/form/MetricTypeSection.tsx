// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RadioCard } from '@nemo/common/src/components/RadioCard';
import { ScaleBalance } from '@nemo/icons/icons';
import { Anchor, Badge, Flex, RadioGroupRoot, Stack, Text } from '@nvidia/foundations-react-core';
import { LINK_EVAL_DOCS_METRICS } from '@studio/constants/links';
import { type FC, useState } from 'react';

const METRIC_TYPE_OPTIONS = [
  {
    value: 'llm-judge',
    label: 'LLM-as-a-Judge',
    description:
      'Use another LLM to evaluate outputs with flexible scoring criteria. Define custom rubrics or numerical ranges.',
    tags: ['custom-scoring', 'rubrics'],
  },
] as const;

export const MetricTypeSection: FC = () => {
  const [metricType, setMetricType] = useState('llm-judge');

  return (
    <Stack gap="6">
      <Stack gap="2">
        <Text kind="body/bold/lg">Metric Type</Text>
        <Text kind="body/regular/md" className="text-secondary">
          Metrics define how to score your model&apos;s outputs. Learn more about{' '}
          <Anchor
            kind="inline"
            textKind="body/regular/md"
            href={LINK_EVAL_DOCS_METRICS}
            target="_blank"
            rel="noreferrer"
          >
            Evaluation Metrics
          </Anchor>
          .
        </Text>
        <Text kind="body/regular/md" className="text-secondary">
          Currently only LLM-as-a-Judge is supported.
        </Text>
      </Stack>

      <RadioGroupRoot
        name="metric-type"
        value={metricType}
        onValueChange={setMetricType}
        orientation="vertical"
      >
        <Stack gap="density-sm">
          {METRIC_TYPE_OPTIONS.map((option) => (
            <RadioCard
              key={option.value}
              value={option.value}
              label={option.label}
              labelSide="left"
              icon={<ScaleBalance className="size-4" aria-hidden />}
              description={
                <Stack gap="density-xs">
                  <span>{option.description}</span>
                  <Flex gap="density-xs" wrap="wrap">
                    {option.tags.map((tag) => (
                      <Badge key={tag} kind="solid" color="gray">
                        {tag}
                      </Badge>
                    ))}
                  </Flex>
                </Stack>
              }
              checked={metricType === option.value}
            />
          ))}
        </Stack>
      </RadioGroupRoot>
    </Stack>
  );
};
