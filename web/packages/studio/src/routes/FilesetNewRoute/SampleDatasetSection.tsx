// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Badge,
  Block,
  Button,
  Card,
  Flex,
  Grid,
  GridItem,
  Text,
} from '@nvidia/foundations-react-core';
import {
  FILESET_NAME_SUFFIX_LIMIT,
  filesetNameTakenMessage,
} from '@studio/components/filesets/filesetNames';
import { useWatchedFilesetNameStatus } from '@studio/components/filesets/hooks/useFilesetNameStatus';
import type { NameCheckStatus } from '@studio/components/form/entityNameCheck';
import { SAMPLE_DATASETS, SampleDataset } from '@studio/constants/sampleDatasets';
import { getSampleDatasetName } from '@studio/routes/FilesetNewRoute/helpers';
import type { DatasetFormFields } from '@studio/routes/FilesetNewRoute/types';
import { FileCheck } from 'lucide-react';
import { FC } from 'react';
import type { Control } from 'react-hook-form';

/** Where finding a free name for the sample stands. `exhausted` means the sample's name and
 *  every numbered copy of it are taken. */
export type SampleNameLookup = 'resolving' | 'resolved' | 'failed' | 'exhausted';

interface SampleDatasetSectionProps {
  control: Control<DatasetFormFields>;
  selectedSampleDataset: SampleDataset;
  onSelectSample: (dataset: SampleDataset) => void;
  workspace: string;
  nameLookup: SampleNameLookup;
  onRetryNameLookup: () => void;
  /** A fileset this form already created, which the name check must not count as taken. */
  ownName?: string;
  onNameStatusChange: (status: NameCheckStatus) => void;
}

export const SampleDatasetSection: FC<SampleDatasetSectionProps> = ({
  control,
  selectedSampleDataset,
  onSelectSample,
  workspace,
  nameLookup,
  onRetryNameLookup,
  ownName,
  onNameStatusChange,
}) => {
  // This tab has no name field, so the name it imports under and any conflict with it are
  // reported here instead.
  const { candidate: importName, status } = useWatchedFilesetNameStatus({
    control,
    name: 'name',
    workspace,
    ownName,
    onStatusChange: onNameStatusChange,
  });
  const baseName = getSampleDatasetName(workspace, selectedSampleDataset.id);

  const renderImportName = () => {
    switch (nameLookup) {
      case 'resolving':
        return (
          <Text kind="body/regular/md" className="text-secondary" role="status">
            Checking for existing sample datasets...
          </Text>
        );
      case 'failed':
        return (
          <Flex gap="density-sm" align="center" role="alert">
            <Text kind="body/regular/md" className="text-feedback-danger">
              Couldn&apos;t check for existing sample datasets.
            </Text>
            <Button kind="tertiary" size="small" onClick={onRetryNameLookup}>
              Retry
            </Button>
          </Flex>
        );
      case 'exhausted':
        return (
          <Text kind="body/regular/md" className="text-feedback-danger" role="alert">
            <span className="text-primary">{baseName}</span> and every copy up to{' '}
            <span className="text-primary">
              {baseName}-{FILESET_NAME_SUFFIX_LIMIT}
            </span>{' '}
            already exist in the <span className="text-primary">{workspace}</span> workspace. Delete
            one of them to import this sample again.
          </Text>
        );
      case 'resolved':
        if (status === 'conflict')
          return (
            <Text kind="body/regular/md" className="text-feedback-danger" role="alert">
              {filesetNameTakenMessage(importName, workspace)}
            </Text>
          );
        return (
          importName && (
            <Text kind="body/regular/md" className="text-secondary">
              Imports as <span className="text-primary">{importName}</span> in the{' '}
              <span className="text-primary">{workspace}</span> workspace
            </Text>
          )
        );
    }
  };

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
      {renderImportName()}
    </>
  );
};
