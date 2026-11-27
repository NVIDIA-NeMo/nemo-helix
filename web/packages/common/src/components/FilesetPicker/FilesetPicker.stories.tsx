// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetPicker, type FilesetPickerProps } from '@nemo/common/src/components/FilesetPicker';
import {
  listFilesetsHandler,
  STORY_FILESETS,
} from '@nemo/common/src/components/FilesetPicker/storyData';
import { Panel, Text } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { type ReactElement, useState } from 'react';

type HarnessProps = Omit<FilesetPickerProps, 'value' | 'onChange'>;

function Harness(props: HarnessProps): ReactElement {
  const [value, setValue] = useState<string | null>(null);
  const [requests, setRequests] = useState(0);
  return (
    <Panel className="max-w-[640px]">
      <FilesetPicker
        {...props}
        value={value}
        onChange={setValue}
        onRequestNew={props.onRequestNew ? () => setRequests((n) => n + 1) : undefined}
      />
      <Text kind="body/regular/xs" className="mt-4 block text-secondary">
        value: {value ?? 'null'}
        {props.onRequestNew ? ` · new requested ${requests}×` : ''}
      </Text>
    </Panel>
  );
}

const meta = {
  title: 'UI/FilesetPicker',
  component: Harness,
  args: {
    workspace: 'default',
    label: 'Training dataset',
    purpose: 'dataset',
    noun: { one: 'dataset', many: 'datasets' },
    onRequestNew: () => {},
  },
  parameters: {
    msw: { handlers: [listFilesetsHandler(STORY_FILESETS)] },
  },
} satisfies Meta<typeof Harness>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const WithoutNewRow: Story = {
  args: { onRequestNew: undefined },
};

export const DefaultNoun: Story = {
  args: { label: 'Fileset', purpose: undefined, noun: undefined },
};

export const SeveralPurposes: Story = {
  args: {
    label: 'Reward environment',
    purpose: ['environment', 'generic'],
    noun: { one: 'environment', many: 'environments' },
  },
};

export const EmptyWorkspace: Story = {
  parameters: { msw: { handlers: [listFilesetsHandler([])] } },
};

export const WithError: Story = {
  args: { errorText: 'Select a training dataset' },
};
