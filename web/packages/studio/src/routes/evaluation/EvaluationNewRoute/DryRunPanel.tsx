// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { resolveKeyPath } from '@nemo/common/src/utils/file';
import { formatEvaluatorScore } from '@nemo/common/src/utils/formatters';
import { Button, Flex, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import { type EvaluationFormValues } from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { useDatasetPreview } from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetPreview';
import { useDryRun } from '@studio/routes/evaluation/EvaluationNewRoute/useDryRun';
import { useTemplateBindings } from '@studio/routes/evaluation/EvaluationNewRoute/useTemplateBindings';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { FC, useEffect, useState } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

const asText = (value: unknown): string => {
  if (value === undefined || value === null) return '';
  return typeof value === 'string' ? value : JSON.stringify(value, null, 2);
};

const PreviewField: FC<{ label: string; value: string }> = ({ label, value }) => (
  <Stack gap="density-xs">
    <Text kind="label/bold/lg">{label}</Text>
    <pre className="whitespace-pre-wrap rounded-md border border-disabled bg-disabled p-density-md font-mono text-xs text-secondary">
      {value}
    </pre>
  </Stack>
);

export const DryRunPanel: FC = () => {
  const { control, handleSubmit } = useFormContext<EvaluationFormValues>();
  const dataset = useWatch({ control, name: 'dataset' });
  /** Which row gets previewed and tested. Lives here rather than in the Dataset
   *  column because it selects the subject of the dry run, not the shape of the
   *  file. Reset on a new file so the index cannot outlive its dataset. */
  const [rowIndex, setRowIndex] = useState(0);
  useEffect(() => setRowIndex(0), [dataset]);
  const { row, rowCount, isPartial } = useDatasetPreview(dataset ?? null, rowIndex);
  // Bindings, not fieldMapping: a messages dataset resolves its input and ground
  // truth positionally, and reading the mapping directly reports them unmapped.
  const bindings = useTemplateBindings();
  const { state, run, cancel } = useDryRun();
  const busy = state.status === 'busy';

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
   *  Create never requires a dry run; sharing the resolver only means Test cannot
   *  pass on a config Create would reject. */
  const runTest = handleSubmit((values) => {
    if (!row) return;
    void run(values, bindings, row);
  });

  return (
    <Stack justify="start" gap="density-lg">
      {(input !== null || reference !== null) && rowCount > 1 ? (
        <Stack gap="density-xs">
          {/* Same shape as the row navigator in FileRowEditor: chevrons either
              side of a "Row N of M" label. Disabled while a run is in flight --
              moving rows mid-run would leave the previews on one row and the
              result on another. */}
          <Flex align="center" justify="center" gap="density-sm">
            <Button
              kind="secondary"
              size="small"
              aria-label="Previous row"
              disabled={busy || rowIndex === 0}
              onClick={() => setRowIndex((index) => Math.max(0, index - 1))}
            >
              <ChevronLeft size={16} />
            </Button>
            <Text kind="body/regular/sm" className="text-secondary">
              Row {rowIndex + 1} of {rowCount}
            </Text>
            <Button
              kind="secondary"
              size="small"
              aria-label="Next row"
              disabled={busy || rowIndex >= rowCount - 1}
              onClick={() => setRowIndex((index) => index + 1)}
            >
              <ChevronRight size={16} />
            </Button>
          </Flex>
          {isPartial ? (
            <Text kind="body/regular/sm" className="text-center text-placeholder">
              First {rowCount} rows of a large file.
            </Text>
          ) : null}
        </Stack>
      ) : null}

      {input !== null ? <PreviewField label="Input Prompt" value={input} /> : null}
      {reference !== null ? <PreviewField label="Ground Truth" value={reference} /> : null}

      {state.status === 'done' ? (
        <>
          <PreviewField label="Model Response" value={state.result.output} />
          <Stack gap="density-xs">
            <Text kind="label/bold/lg">Score</Text>
            {state.result.scores.length === 0 ? (
              <Text kind="body/regular/md" className="text-secondary">
                The run produced no scores.
              </Text>
            ) : (
              state.result.scores.map((score) => (
                <Text key={score.name} kind="body/regular/md">
                  {score.name}: {formatEvaluatorScore(score.value)}
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
        {/* type="button", never a second submit: the dry run is a separate action
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
