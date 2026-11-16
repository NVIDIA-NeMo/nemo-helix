// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Mock factories for picking a dataset in the evaluation wizards: the shared fileset file picker
 * and the row preview read from the picked file.
 *
 * vi.mock is hoisted per file, so consumers call it themselves and import these lazily:
 *   vi.mock('@nemo/common/src/components/DatasetFileSelect/ControlledDatasetFileSelect', async () =>
 *     (await import('@studio/tests/util/mockDatasetPick')).datasetFileSelectMock()
 *   );
 *   vi.mock('@studio/components/evaluation/EvalWizard/useDatasetPreview', async (orig) =>
 *     (await import('@studio/tests/util/mockDatasetPick')).datasetPreviewMock(await orig(), ROW)
 *   );
 *
 * The picker renders a "Dataset file ref" text box: paste a `workspace/fileset#path` ref into it
 * to pick that file. The preview answers every picked ref with the given row.
 */

import type { DatasetPreview } from '@studio/components/evaluation/EvalWizard/useDatasetPreview';
import { type UseControllerProps, useController } from 'react-hook-form';

export const datasetFileSelectMock = () => ({
  ControlledDatasetFileSelect: ({
    useControllerProps,
  }: {
    useControllerProps: UseControllerProps;
  }) => {
    const { field, fieldState } = useController(useControllerProps);
    return (
      <>
        <input
          aria-label="Dataset file ref"
          value={typeof field.value === 'string' ? field.value : ''}
          onChange={(event) => field.onChange(event.target.value || null)}
        />
        {fieldState.error ? <span>{fieldState.error.message}</span> : null}
      </>
    );
  },
});

export const datasetPreviewMock = <T extends object>(
  original: T,
  row: Record<string, unknown>
) => ({
  ...original,
  useDatasetPreview: (datasetRef: string | null): DatasetPreview => ({
    row: datasetRef ? row : null,
    rowCount: datasetRef ? 1 : 0,
    isPartial: false,
    keyOptions: datasetRef ? Object.keys(row).map((key) => ({ label: key, value: key })) : [],
    messagesColumn: null,
    messageSelectors: [],
    isLoading: false,
    error: null,
  }),
});
