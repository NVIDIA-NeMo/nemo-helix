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
import { DocumentCheckmark } from '@nvidia/react-gui-icons/icons';
import { filesetNameTakenMessage } from '@studio/components/filesets/filesetNames';
import {
  invalidateFilesetNameCheck,
  useWatchedFilesetNameStatus,
} from '@studio/components/filesets/hooks/useFilesetNameStatus';
import type { NameCheckStatus } from '@studio/components/form/entityNameCheck';
import { SAMPLE_DATASETS, SampleDataset } from '@studio/constants/sampleDatasets';
import type { DatasetFormFields } from '@studio/routes/FilesetNewRoute/types';
import { useQueryClient } from '@tanstack/react-query';
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
  const queryClient = useQueryClient();
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
                  <DocumentCheckmark />
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
      {/* There is no name to edit here, so a failed check can only be rerun from this button. */}
      {status === 'failed' && (
        <Flex gap="density-sm" align="center">
          <Text kind="body/regular/md" role="status">
            Couldn't check name availability.
          </Text>
          <Button
            type="button"
            kind="tertiary"
            onClick={() => void invalidateFilesetNameCheck(queryClient, workspace, importName)}
          >
            Retry
          </Button>
        </Flex>
      )}
    </>
  );
};
