// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  FilesetFilePicker,
  type FilesetFilePickerProps,
  type FilesetFileValue,
} from '@nemo/common/src/components/FilesetPicker/FilesetFilePicker';
import {
  listFilesetFilesHandler,
  listFilesetsHandler,
  STORY_FILES,
  STORY_FILESETS,
} from '@nemo/common/src/components/FilesetPicker/storyData';
import { Panel, Text } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { type ReactElement, useState } from 'react';

type HarnessProps = Omit<FilesetFilePickerProps, 'value' | 'onChange'>;

function Harness(props: HarnessProps): ReactElement {
  const [value, setValue] = useState<FilesetFileValue>({ filesetRef: null, filePath: null });
  return (
    <Panel className="max-w-[640px]">
      <FilesetFilePicker {...props} value={value} onChange={setValue} />
      <Text kind="body/regular/xs" className="mt-4 block text-secondary">
        value: {JSON.stringify(value)}
      </Text>
    </Panel>
  );
}

const meta = {
  // Shares the FilesetPicker's title so these sit beside its stories.
  title: 'UI/FilesetPicker',
  component: Harness,
  args: {
    workspace: 'default',
    label: 'Evaluation dataset',
    purpose: 'dataset',
    noun: { one: 'dataset', many: 'datasets' },
    acceptedFileExtensions: ['.csv', '.parquet'],
  },
  parameters: {
    msw: {
      handlers: [listFilesetsHandler(STORY_FILESETS), listFilesetFilesHandler(STORY_FILES)],
    },
  },
} satisfies Meta<typeof Harness>;

export default meta;
type Story = StoryObj<typeof meta>;

/** `sft-chat-v3` has eligible and ineligible files; `sft-chat-v2` has none eligible; `support-transcripts-raw` is empty. */
export const PickAFile: Story = {};

export const PickAnyFile: Story = {
  args: {
    acceptedFileExtensions: undefined,
    label: 'Fileset',
    noun: undefined,
    purpose: undefined,
  },
};
