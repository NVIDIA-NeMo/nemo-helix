// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledDatasetFileSelect } from '@nemo/common/src/components/DatasetFileSelect/ControlledDatasetFileSelect';
import { Accordion, Text, Block } from '@nvidia/foundations-react-core';
import { DataPreparationSettings } from '@studio/routes/SafeSynthesizerNewRoute/components/DataPreparationSettings';
import type { SafeSynthesizerFormData } from '@studio/routes/SafeSynthesizerNewRoute/schema';
import { useState } from 'react';
import { useFormContext } from 'react-hook-form';

export const TrainingData = ({ workspace }: { workspace: string }) => {
  const [openAccordion, setOpenAccordion] = useState<string>();
  const {
    formState: { errors },
    control,
    getValues,
    setError,
    clearErrors,
  } = useFormContext<SafeSynthesizerFormData>();

  const dataSourceError = errors.spec?.data_source?.message;
  const dataSourceValue = getValues('spec.data_source');

  return (
    <>
      <ControlledDatasetFileSelect
        label="Data Source"
        useControllerProps={{ name: 'spec.data_source', control: control }}
        acceptedFileTypes={['.jsonl', '.csv', '.parquet']}
        setError={(error) => setError('spec.data_source', error)}
        clearError={() => clearErrors('spec.data_source')}
        workspace={workspace}
        formFieldProps={{}}
      />
      {!dataSourceError && dataSourceValue && (
        <Block className="bg-surface-base rounded-lg" padding="density-xl">
          <Text kind="body/regular/md">
            Automatic holdout will split 5% of your training data for validation
          </Text>
        </Block>
      )}
      <Accordion
        className="[&>div]:border-b-0"
        onValueChange={setOpenAccordion}
        items={[
          {
            slotTrigger: `${openAccordion === 'data-preparation-settings' ? 'Hide' : 'Show'} Data Preparation Settings`,
            slotContent: <DataPreparationSettings />,
            value: 'data-preparation-settings',
          },
        ]}
      />
    </>
  );
};
