// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseFilesetLocation } from '@nemo/common/src/components/DatasetFileSelect/parseFilesetLocation';
import { DatasetRowPager } from '@nemo/common/src/components/DatasetRowPager';
import { PreviewBox } from '@nemo/common/src/components/PreviewBox';
import { resolveKeyPath } from '@nemo/common/src/utils/file';
import { formatEvaluatorScore } from '@nemo/common/src/utils/formatters';
import { Button, Flex, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import { type EvaluationFormValues } from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { useDatasetBindings } from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetBindings';
import { useDatasetPreview } from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetPreview';
import { useLiveTest } from '@studio/routes/evaluation/EvaluationNewRoute/useLiveTest';
import { FC, useEffect, useState } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

const asText = (value: unknown): string => {
  if (value === undefined || value === null) return '';
  return typeof value === 'string' ? value : JSON.stringify(value, null, 2);
};

const PreviewField: FC<{ label: string; value: string }> = ({ label, value }) => (
  <Stack gap="density-xxs">
    <Text kind="label/bold/sm">{label}</Text>
    <PreviewBox value={value} label={label} />
  </Stack>
);

export const LiveTestPanel: FC = () => {
  const { control, handleSubmit } = useFormContext<EvaluationFormValues>();
  const dataset = useWatch({ control, name: 'dataset' });
  /** Which row gets previewed and tested. Lives here rather than in the Dataset
   *  column because it selects the subject of the live test, not the shape of the
   *  file. Reset on a new file so the index cannot outlive its dataset. */
  const [rowIndex, setRowIndex] = useState(0);
  useEffect(() => setRowIndex(0), [dataset]);
  const { row, rowCount, isPartial } = useDatasetPreview(dataset ?? null, rowIndex);
  const fileName = dataset ? parseFilesetLocation(dataset)?.objectPath.split('/').pop() : null;
  // Bindings, not fieldMapping: a messages dataset resolves its input and ground
  // truth positionally, and reading the mapping directly reports them unmapped.
  const bindings = useDatasetBindings();
  const { state, run, cancel } = useLiveTest();
  const busy = state.status === 'busy';

  /** A result describes the row it was run against. Moving off that row, or
   *  swapping the file, would leave the response and scores sitting under
   *  previews of something else. ``cancel`` also aborts a run still in flight,
   *  which is what a dataset change mid-run should do. */
  useEffect(() => {
    cancel();
  }, [rowIndex, dataset, cancel]);

  /** A label with nothing under it is noise. Each preview appears only once its
   *  field actually resolves against the selected row -- what is missing is
   *  already stated where it can be fixed, in the Dataset column. */
  const input = row && bindings.inputPath ? asText(resolveKeyPath(row, bindings.inputPath)) : null;
  const reference =
    row && bindings.referencePath ? asText(resolveKeyPath(row, bindings.referencePath)) : null;

  /** handleSubmit rather than trigger(): both run the resolver, but only
   *  handleSubmit marks the form submitted, and reValidateMode ('onChange') is
   *  gated on that. With trigger() the errors it raises never clear, because RHF
   *  is still in pre-submit mode and revalidates nothing.
   *
   *  Create never requires a live test; sharing the resolver only means Test cannot
   *  pass on a config Create would reject. */
  const modelResponse =
    state.status === 'done'
      ? state.result.output
      : state.status === 'busy'
        ? (state.output ?? null)
        : null;

  const runTest = handleSubmit((values) => {
    if (!row) return;
    void run(values, bindings, row);
  });

  return (
    <Stack justify="start" gap="density-lg">
      {(input !== null || reference !== null) && rowCount > 1 ? (
        <DatasetRowPager
          fileName={fileName}
          rowIndex={rowIndex}
          rowCount={rowCount}
          isPartial={isPartial}
          onChange={setRowIndex}
          disabled={busy}
        />
      ) : null}

      {input !== null ? <PreviewField label="Input" value={input} /> : null}
      {reference !== null ? <PreviewField label="Reference" value={reference} /> : null}

      {modelResponse !== null ? (
        <PreviewField label="Model Response" value={modelResponse} />
      ) : null}

      {state.status === 'done' ? (
        <>
          <Stack gap="density-xs">
            <Text kind="label/bold/lg">Score</Text>
            {state.result.scores.length === 0 ? (
              <Text kind="body/regular/md" className="text-secondary">
                The run produced no scores.
              </Text>
            ) : (
              state.result.scores.map((score) => (
                <Text key={score.name} kind="body/regular/md">
                  {score.name}: {score.label ?? formatEvaluatorScore(score.value)}
                </Text>
              ))
            )}
          </Stack>
        </>
      ) : null}

      {state.status === 'busy' ? (
        <Flex align="center" gap="density-sm">
          <Spinner size="small" aria-label={state.label} />
          <Text kind="body/regular/md" className="text-secondary">
            {state.label}
          </Text>
        </Flex>
      ) : null}

      {state.status === 'error' ? (
        <Text kind="body/regular/md" className="text-feedback-danger">
          {state.message}
        </Text>
      ) : null}

      <Stack>
        {/* type="button", never a second submit: the live test is a separate action
            from creating the evaluation and must not trigger the form. Cancel
            replaces Test rather than sitting beside it -- a disabled Test during a
            run is a control with nothing to offer, in the narrowest column. */}
        {busy ? (
          <Button kind="secondary" type="button" onClick={cancel}>
            Cancel
          </Button>
        ) : (
          <Button kind="secondary" type="button" onClick={() => void runTest()}>
            Test
          </Button>
        )}
      </Stack>
    </Stack>
  );
};
