// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledDatasetFileSelect } from '@nemo/common/src/components/DatasetFileSelect/ControlledDatasetFileSelect';
import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { Stack, Text } from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import {
  SOURCE_TYPE_DATASET,
  SOURCE_TYPE_OPTIONS,
} from '@studio/routes/AnonymizerBuilderRoute/constants';
import type { AnonymizerFormData } from '@studio/routes/AnonymizerBuilderRoute/schema';
import type { FC } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

export const DataSourceSection: FC = () => {
  const { control, setValue, setError, clearErrors } = useFormContext<AnonymizerFormData>();
  const workspace = useWorkspaceFromPath();
  const sourceType = useWatch({ control, name: 'sourceType' });
  const isDataset = sourceType === SOURCE_TYPE_DATASET;

  return (
    <Stack gap="density-lg">
      <Text kind="label/bold/lg">Data Source</Text>
      <ControlledSelect
        aria-label="Source type"
        items={SOURCE_TYPE_OPTIONS}
        useControllerProps={{ name: 'sourceType', control }}
        onChange={() => {
          setValue('source', '');
          clearErrors('source');
        }}
        formFieldProps={{ slotLabel: 'Source', required: true }}
      />
      {isDataset ? (
        <ControlledDatasetFileSelect
          label="Dataset"
          acceptedFileTypes={['.csv', '.parquet']}
          useControllerProps={{ name: 'source', control }}
          setError={(error) => setError('source', error)}
          clearError={() => clearErrors('source')}
          workspace={workspace}
          formFieldProps={{ required: true }}
        />
      ) : (
        <ControlledTextInput
          useControllerProps={{ name: 'source', control }}
          placeholder="https://example.com/data.csv"
          formFieldProps={{
            slotLabel: 'URL',
            required: true,
            slotInfo: 'HTTP(S) URL of a CSV or Parquet file to anonymize.',
          }}
        />
      )}
    </Stack>
  );
};
