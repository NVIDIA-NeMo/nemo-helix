// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatEvaluatorScore } from '@nemo/common/src/utils/formatters';
import {
  Banner,
  Button,
  Flex,
  FormField,
  Panel,
  Spinner,
  Stack,
  Tag,
  Text,
  TextArea,
} from '@nvidia/foundations-react-core';
import type { MetricFormValues } from '@studio/routes/MetricCreateRoute/types';
import { type SampleRow, useMetricTest } from '@studio/routes/MetricCreateRoute/useMetricTest';
import { type FC, useEffect, useState } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

const DEFAULT_SAMPLE: SampleRow = {
  input: 'What is the capital of France?',
  reference: 'Paris',
  context: '',
  output: 'The capital of France is Paris.',
};

const FIELDS: { key: keyof SampleRow; label: string; help?: string }[] = [
  { key: 'input', label: 'Input' },
  { key: 'reference', label: 'Reference' },
  { key: 'context', label: 'Context', help: 'Optional.' },
  { key: 'output', label: 'Response', help: 'The model output being scored.' },
];

/** Scores a hand-written example with the metric as currently configured, so
 *  the user sees what it produces before saving it. */
export const TryMetricPanel: FC<{ workspace: string }> = ({ workspace }) => {
  const { control, handleSubmit } = useFormContext<MetricFormValues>();
  const [sample, setSample] = useState(DEFAULT_SAMPLE);
  const { state, run, reset } = useMetricTest(workspace);

  // A result describes the config and sample it ran against; drop it on any edit.
  const config = JSON.stringify(useWatch({ control, name: ['type', 'body'] }));
  useEffect(reset, [config, sample, reset]);

  const test = handleSubmit((values) => void run(values, sample));

  return (
    <Panel
      elevation="mid"
      slotHeading="Try it"
      slotFooter={
        <Flex justify="end" className="w-full">
          <Button kind="secondary" onClick={() => void test()} disabled={state.status === 'busy'}>
            Test Metric
          </Button>
        </Flex>
      }
    >
      <Stack gap="density-lg">
        <Text kind="body/regular/sm" className="text-secondary">
          Score an example row. Nothing is saved.
        </Text>
        {FIELDS.map(({ key, label, help }) => (
          <FormField key={key} slotLabel={label} slotHelp={help}>
            <TextArea
              rows={key === 'output' ? 3 : 2}
              value={sample[key]}
              onValueChange={(value) => setSample((prev) => ({ ...prev, [key]: value }))}
            />
          </FormField>
        ))}

        {state.status === 'busy' ? <Spinner size="small" description="Scoring…" /> : null}
        {state.status === 'error' ? (
          <Banner kind="inline" status="error">
            {state.message}
          </Banner>
        ) : null}
        {state.status === 'done' ? (
          <Stack gap="density-sm">
            <Text kind="label/bold/md">Result</Text>
            {state.error ? (
              <Banner kind="inline" status="error">
                {state.error}
              </Banner>
            ) : null}
            <Flex gap="density-sm" wrap="wrap">
              {state.scores
                .filter((score) => !score.name.endsWith('.label'))
                .map((score) => (
                  <Tag key={score.name} readOnly color="green">
                    {score.name}: {formatEvaluatorScore(score.value)}
                    {score.label ? ` (${score.label})` : ''}
                  </Tag>
                ))}
            </Flex>
          </Stack>
        ) : null}
      </Stack>
    </Panel>
  );
};
