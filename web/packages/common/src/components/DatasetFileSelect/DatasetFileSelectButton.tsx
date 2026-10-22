// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ListItem } from '@nemo/common/src/components/FileList/ListItem';
import { Button, Flex, Stack } from '@nvidia/foundations-react-core';
import { Db } from '@nvidia/react-gui-icons/icons';
import { FC } from 'react';

interface DatasetFileSelectButtonProps {
  datasetName?: string;
  onSelectClick: () => void;
  onChangeClick?: () => void;
  selectButtonText?: string;
  changeButtonText?: string;
}

export const DatasetFileSelectButton: FC<DatasetFileSelectButtonProps> = ({
  datasetName,
  onSelectClick,
  onChangeClick,
  selectButtonText = 'Select File',
  changeButtonText = 'Change Dataset',
}) => {
  if (!datasetName) {
    return (
      <Stack className="w-full">
        <Button type="button" kind="secondary" className="w-full" onClick={onSelectClick}>
          {selectButtonText}
        </Button>
      </Stack>
    );
  }

  return (
    <Flex gap="density-md" align="end" className="mb-density-md min-h-[40px]">
      <Stack className="flex-1 h-full">
        <ListItem outlined value={datasetName} startIconSlot={<Db />} />
      </Stack>
      <Flex gap="density-sm">
        <Button type="button" kind="secondary" onClick={onChangeClick || onSelectClick}>
          {changeButtonText}
        </Button>
      </Flex>
    </Flex>
  );
};
