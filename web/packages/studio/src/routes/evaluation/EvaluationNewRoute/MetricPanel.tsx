// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledCheckbox } from '@nemo/common/src/components/form/ControlledCheckbox';
import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { Anchor, Banner, Block, Card, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { ScoreDefinitions } from '@studio/components/evaluation/Jobs/form/ScoreDefinitions';
import { JudgeModelSelect } from '@studio/components/evaluation/JudgeModelSelect';
import { LINK_EVAL_DOCS_METRICS } from '@studio/constants/links';
import { JudgePromptSection } from '@studio/routes/evaluation/EvaluationNewRoute/JudgePromptSection';
import {
  type EvaluationFormValues,
  COMPARISON_METRICS,
  NUMBER_CHECK_OPERATIONS,
  REFERENCE_METRICS,
  type SelectableMetric,
  SELECTABLE_METRICS,
  STRING_CHECK_OPERATIONS,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { useDatasetBindings } from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetBindings';
import { FC, ReactNode, useEffect } from 'react';
import { type FieldError, useFormContext, useWatch } from 'react-hook-form';

const toItems = (operations: readonly string[]) =>
  operations.map((operation) => ({ value: operation, children: operation }));

/** Reads as the comparison it builds: "Model Response <operation> Reference".
 *  The operands are not interchangeable -- contains, startswith and endswith are
 *  asymmetric -- and a bare "Operation" label leaves their order invisible. */
const OperandRelation: FC<{ children: ReactNode }> = ({ children }) => (
  <Flex align="center" gap="density-sm" className="min-w-0">
    <Text kind="body/regular/md" className="shrink-0">
      Model Response
    </Text>
    <Block className="min-w-0 flex-1">{children}</Block>
    <Text kind="body/regular/md" className="shrink-0">
      Reference
    </Text>
  </Flex>
);

/** Metrics that cannot run without a Reference. */
const GROUND_TRUTH_METRICS: readonly SelectableMetric[] = [
  ...REFERENCE_METRICS,
  ...COMPARISON_METRICS,
];

export const MetricPanel: FC = () => {
  const {
    control,
    setValue,
    getValues,
    formState: { errors },
  } = useFormContext<EvaluationFormValues>();
  /** `body.metrics` is an object node -- the checkboxes bind to its children --
   *  so an error set on it has no field of its own to render into. */
  const metricsError = (errors.body?.metrics as FieldError | undefined)?.message;
  const [metrics, numberCheckOperation] = useWatch({
    control,
    name: ['body.metrics', 'body.numberCheck.operation'],
  });

  const noGroundTruth = !useDatasetBindings().referencePath;

  useEffect(() => {
    if (!noGroundTruth) return;
    for (const type of GROUND_TRUTH_METRICS) {
      if (getValues(`body.metrics.${type}`)) setValue(`body.metrics.${type}`, false);
    }
  }, [noGroundTruth, getValues, setValue]);

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

      {noGroundTruth ? (
        <Banner kind="inline" status="info">
          Metrics that score against a Reference are unavailable, because this dataset has none
          mapped. To enable those metrics, go back to Configuration to map a Reference field.
        </Banner>
      ) : null}

      <Card className="min-w-0 p-density-lg">
        <Stack gap="density-lg" className="min-w-0">
          {SELECTABLE_METRICS.map(({ type, label }) => (
            <Stack key={type} gap="density-sm" className="min-w-0">
              <ControlledCheckbox
                useControllerProps={{ name: `body.metrics.${type}` as const, control }}
                slotLabel={<Text kind="body/bold/lg">{label}</Text>}
                disabled={noGroundTruth && GROUND_TRUTH_METRICS.includes(type)}
              />

              {metrics?.[type] && !REFERENCE_METRICS.includes(type) ? (
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

                      <JudgePromptSection />

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
