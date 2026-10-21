// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getEntityReference } from '@nemo/common/src/namedEntity';
import { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { Flex, Select, Text } from '@nvidia/foundations-react-core';
import {
  LOADING_DATASETS_ERROR_OPTION,
  LOADING_DATASETS_OPTION,
} from '@studio/components/DatasetSelect/constants';
import { ValueWithLabel } from '@studio/components/ValueWithLabel';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import { useOwnedFirstDatasets } from '@studio/hooks/useOwnedFirstDatasets';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { FC, useMemo } from 'react';

interface Props {
  onChange: (datasetId?: string) => void;
  selectedDatasetId: string;
  errorText?: string;
  disabled?: boolean;
}

const datasetToOption = (dataset: FilesetOutput) => ({
  children: dataset.name ?? '',
  value: getEntityReference(dataset),
});

export const DatasetSelect: FC<Props> = ({ onChange, selectedDatasetId, errorText, disabled }) => {
  const workspace = useWorkspaceFromPath();
  // Pickers fetch a large page so search can run client-side; the user's own datasets come first.
  const {
    datasets,
    isPending: isLoading,
    isError,
  } = useOwnedFirstDatasets(workspace, DEFAULT_LARGE_PAGE_SIZE);

  const selectedDatasetOption = useMemo(() => {
    const selectedDataset = datasets?.find(
      (dataset) => getEntityReference(dataset) === selectedDatasetId
    );
    return selectedDataset ? datasetToOption(selectedDataset)?.value : undefined;
  }, [selectedDatasetId, datasets]);

  const datasetOptions = useMemo(() => {
    if (isLoading) {
      return [LOADING_DATASETS_OPTION];
    } else if (isError) {
      return [LOADING_DATASETS_ERROR_OPTION];
    }

    return datasets?.map(datasetToOption) ?? [];
  }, [datasets, isLoading, isError]);

  return (
    <Flex direction="col" gap="density-xs">
      <ValueWithLabel
        label="Dataset"
        value={
          <Select
            aria-label="dataset-select"
            disabled={disabled}
            items={datasetOptions}
            value={selectedDatasetOption}
            onValueChange={(value) => onChange(value)}
            placeholder="Select a dataset"
          />
        }
      />
      {(selectedDatasetId || errorText) && (
        <Text
          fontWeight="light"
          className={`min-h-3 ${errorText ? 'text-feedback-danger' : 'text-secondary'}`}
        >
          {errorText || `Dataset ID: ${selectedDatasetId}`}
        </Text>
      )}
    </Flex>
  );
};
