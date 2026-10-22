// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Block, Card, Flex, Grid, GridItem, Text } from '@nvidia/foundations-react-core';
import { filesetNameTakenMessage } from '@studio/components/filesets/filesetNames';
import { useWatchedFilesetNameStatus } from '@studio/components/filesets/hooks/useFilesetNameStatus';
import type { NameCheckStatus } from '@studio/components/form/entityNameCheck';
import { SAMPLE_DATASETS, SampleDataset } from '@studio/constants/sampleDatasets';
import type { DatasetFormFields } from '@studio/routes/FilesetNewRoute/types';
import { FileCheck } from 'lucide-react';
import { FC } from 'react';
import type { Control } from 'react-hook-form';

interface SampleDatasetSectionProps {
  control: Control<DatasetFormFields>;
  selectedSampleDataset: SampleDataset;
  onSelectSample: (dataset: SampleDataset) => void;
  workspace: string;
  /** A fileset this form already created, which the name check must not count as taken. */
  ownName?: string;
  onNameStatusChange: (status: NameCheckStatus) => void;
}

export const SampleDatasetSection: FC<SampleDatasetSectionProps> = ({
  control,
  selectedSampleDataset,
  onSelectSample,
  workspace,
  ownName,
  onNameStatusChange,
}) => {
  // This tab has no name field, so a conflict with the name it imports under is reported here.
  const { candidate: importName, status } = useWatchedFilesetNameStatus({
    control,
    name: 'name',
    workspace,
    ownName,
    onStatusChange: onNameStatusChange,
  });

  return (
    <>
      <Block>
        <Text kind="body/regular/md" className="block">
          Choose from the following pre-configured sample datasets.
        </Text>
      </Block>
      <Grid gap="density-md" cols={2}>
        {SAMPLE_DATASETS.map((dataset) => (
          <GridItem key={dataset.id}>
            <Card
              interactive
              selected={selectedSampleDataset.id === dataset.id}
              onClick={() => onSelectSample(dataset)}
              className="cursor-pointer shadow-none!"
              slotHeader={
                <Badge kind="solid" color="purple">
                  <FileCheck />
                  Sample Dataset
                </Badge>
              }
            >
              <Flex gap="density-sm" direction="col">
                <Text kind="label/bold/md">{dataset.name}</Text>
                <Text kind="body/regular/md">{dataset.description}</Text>
              </Flex>
            </Card>
          </GridItem>
        ))}
      </Grid>
      {status === 'conflict' && (
        <Text kind="body/regular/md" className="text-feedback-danger" role="alert">
          {filesetNameTakenMessage(importName, workspace)}
        </Text>
      )}
    </>
  );
};
