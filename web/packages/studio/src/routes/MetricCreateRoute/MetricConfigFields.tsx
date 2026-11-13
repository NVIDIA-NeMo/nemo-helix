// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { Badge, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import {
  NUMBER_CHECK_OPERATIONS,
  REFERENCE_METRICS,
  STRING_CHECK_OPERATIONS,
} from '@studio/components/evaluation/EvalWizard/types';
import { ScoreDefinitions } from '@studio/components/evaluation/Jobs/form/ScoreDefinitions';
import { JudgeModelSelect } from '@studio/components/evaluation/JudgeModelSelect';
import { JudgePromptField } from '@studio/routes/MetricCreateRoute/JudgePromptField';
import type { MetricFormValues } from '@studio/routes/MetricCreateRoute/types';
import type { FC } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

const toItems = (operations: readonly string[]) =>
  operations.map((operation) => ({ value: operation, children: operation }));

/** "Response <operation> Reference", so the operand order of asymmetric
 *  operations such as contains and startswith is visible. */
const Comparison: FC<{ name: 'body.stringCheck.operation' | 'body.numberCheck.operation' }> = ({
  name,
}) => {
  const { control } = useFormContext<MetricFormValues>();
  return (
    <Flex align="center" gap="density-sm" className="min-w-0">
      <Badge kind="outline" color="gray">{`{{sample.output_text}}`}</Badge>
      <ControlledSelect
        useControllerProps={{ name, control }}
        attributes={{ SelectTrigger: { 'aria-label': 'Operation' } }}
        items={toItems(
          name === 'body.stringCheck.operation' ? STRING_CHECK_OPERATIONS : NUMBER_CHECK_OPERATIONS
        )}
      />
      <Badge kind="outline" color="gray">{`{{reference}}`}</Badge>
    </Flex>
  );
};

/** Settings for the chosen metric type. Reference metrics need none. */
export const MetricConfigFields: FC = () => {
  const { control } = useFormContext<MetricFormValues>();
  const [type, numberOperation] = useWatch({
    control,
    name: ['type', 'body.numberCheck.operation'],
  });

  if (REFERENCE_METRICS.includes(type)) {
    return (
      <Text kind="body/regular/md" className="text-secondary">
        Compares the model&apos;s response with each row&apos;s <code>reference</code> field. No
        other settings are needed.
      </Text>
    );
  }

  if (type === 'string-check') return <Comparison name="body.stringCheck.operation" />;

  if (type === 'number-check') {
    return (
      <Stack gap="density-lg">
        <Comparison name="body.numberCheck.operation" />
        {numberOperation === 'absolute difference' ? (
          <ControlledTextInput
            useControllerProps={{ name: 'body.numberCheck.epsilon', control }}
            formFieldProps={{
              slotLabel: 'Tolerance',
              slotHelp: 'Passes when the two numbers differ by no more than this.',
            }}
            type="number"
          />
        ) : null}
      </Stack>
    );
  }

  return (
    <Stack gap="density-xl">
      <JudgeModelSelect<MetricFormValues>
        formFieldName="body.judgeModel"
        slotLabel="Judge Model"
        required
      />
      <Stack gap="density-sm">
        <Text kind="label/bold/md">Scores</Text>
        <Text kind="body/regular/sm" className="text-secondary">
          What the judge grades. Each score becomes an output of this metric.
        </Text>
        <ScoreDefinitions />
      </Stack>
      <JudgePromptField />
    </Stack>
  );
};
