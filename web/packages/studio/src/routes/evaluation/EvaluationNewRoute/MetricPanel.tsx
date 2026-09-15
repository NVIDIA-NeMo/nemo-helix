// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledCheckbox } from '@nemo/common/src/components/form/ControlledCheckbox';
import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { Anchor, Block, Card, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { ScoreDefinitions } from '@studio/components/evaluation/Jobs/form/ScoreDefinitions';
import { JudgeModelSelect } from '@studio/components/evaluation/JudgeModelSelect';
import { LINK_EVAL_DOCS_METRICS } from '@studio/constants/links';
import {
  type EvaluationFormValues,
  NUMBER_CHECK_OPERATIONS,
  SELECTABLE_METRICS,
  STRING_CHECK_OPERATIONS,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { FC, ReactNode } from 'react';
import { type FieldError, useFormContext, useWatch } from 'react-hook-form';

const toItems = (operations: readonly string[]) =>
  operations.map((operation) => ({ value: operation, children: operation }));

/** Reads as the comparison it builds: "Model Response <operation> Ground Truth".
 *  The operands are not interchangeable -- contains, startswith and endswith are
 *  asymmetric -- and a bare "Operation" label leaves their order invisible. */
const OperandRelation: FC<{ children: ReactNode }> = ({ children }) => (
  <Flex align="center" gap="density-sm" className="min-w-0">
    <Text kind="body/regular/md" className="shrink-0">
      Model Response
    </Text>
    <Block className="min-w-0 flex-1">{children}</Block>
    <Text kind="body/regular/md" className="shrink-0">
      Ground Truth
    </Text>
  </Flex>
);

export const MetricPanel: FC = () => {
  const {
    control,
    formState: { errors },
  } = useFormContext<EvaluationFormValues>();
  /** `body.metrics` is an object node -- the checkboxes bind to its children --
   *  so an error set on it has no field of its own to render into. */
  const metricsError = (errors.body?.metrics as FieldError | undefined)?.message;
  const [metrics, numberCheckOperation] = useWatch({
    control,
    name: ['body.metrics', 'body.numberCheck.operation'],
  });

  return (
    <Stack justify="start" gap="density-lg">
      <Text kind="body/regular/md" className="text-secondary">
        Every selected metric scores each row. See{' '}
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

      <Card className="min-w-0 p-density-lg">
        <Stack gap="density-lg" className="min-w-0">
          {SELECTABLE_METRICS.map(({ type, label }) => (
            <Stack key={type} gap="density-sm" className="min-w-0">
              <ControlledCheckbox
                useControllerProps={{ name: `body.metrics.${type}` as const, control }}
                slotLabel={<Text kind="body/bold/lg">{label}</Text>}
              />

              {/* Exact Match / F1 / BLEU / ROUGE render nothing -- their reference
                  template is the resolved binding and candidate is omitted, so the
                  evaluator falls back to the model's own output. */}
              {metrics?.[type] ? (
                <Stack gap="density-lg" className="min-w-0 pl-density-lg">
                  {type === 'string-check' ? (
                    <OperandRelation>
                      <ControlledSelect
                        useControllerProps={{ name: 'body.stringCheck.operation', control }}
                        items={toItems(STRING_CHECK_OPERATIONS)}
                      />
                    </OperandRelation>
                  ) : null}

                  {type === 'number-check' ? (
                    <>
                      <ControlledSelect
                        useControllerProps={{ name: 'body.numberCheck.operation', control }}
                        formFieldProps={{ slotLabel: 'Operation' }}
                        items={toItems(NUMBER_CHECK_OPERATIONS)}
                      />
                      {numberCheckOperation === 'absolute difference' ? (
                        <ControlledTextInput
                          useControllerProps={{ name: 'body.numberCheck.epsilon', control }}
                          formFieldProps={{ slotLabel: 'Tolerance' }}
                          type="number"
                        />
                      ) : null}
                    </>
                  ) : null}

                  {type === 'llm-judge' ? (
                    <>
                      <JudgeModelSelect<EvaluationFormValues>
                        formFieldName="body.judgeModel"
                        slotLabel="Judge Model"
                        placeholder="Select a judge model"
                      />

                      {/* ScoreDefinitions directly rather than MetricScoreSection:
                          that wrapper heads the list at body/bold/lg, the same size
                          as this column's own title, which inverts the hierarchy. */}
                      <Stack gap="density-sm" className="min-w-0">
                        <Text kind="label/bold/md">Score Definitions</Text>
                        <Text kind="body/regular/md" className="text-secondary">
                          Scores extracted from the judge&apos;s output. These also tell the judge
                          what to grade.
                        </Text>
                        <ScoreDefinitions />
                      </Stack>
                    </>
                  ) : null}
                </Stack>
              ) : null}
            </Stack>
          ))}
        </Stack>
      </Card>

      {metricsError ? (
        <Text kind="body/regular/md" className="text-feedback-danger">
          {metricsError}
        </Text>
      ) : null}
    </Stack>
  );
};
