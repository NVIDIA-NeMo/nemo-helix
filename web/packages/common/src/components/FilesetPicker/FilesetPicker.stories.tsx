// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetPicker, type FilesetPickerProps } from '@nemo/common/src/components/FilesetPicker';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { Panel, Text } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { http, HttpResponse } from 'msw';
import { type ReactElement, useState } from 'react';

const fileset = (
  name: string,
  purpose: FilesetOutput['purpose'],
  storageType: FilesetOutput['storage']['type']
): FilesetOutput =>
  ({
    id: `default/${name}`,
    name,
    workspace: 'default',
    description: '',
    purpose,
    storage: { type: storageType },
    metadata: {},
    custom_fields: {},
    project: 'default',
    created_at: '2026-11-01T00:00:00Z',
    updated_at: '2026-11-01T00:00:00Z',
  }) as FilesetOutput;

const FILESETS: FilesetOutput[] = [
  fileset('sft-chat-v3', 'dataset', 'local'),
  fileset('sft-chat-v2', 'dataset', 'local'),
  fileset('support-transcripts-raw', 'dataset', 's3'),
  fileset('nemotron-personas', 'dataset', 'huggingface'),
  fileset('ngc-eval-suite', 'dataset', 'ngc'),
  fileset('grpo-reward-env', 'environment', 'local'),
  fileset('gym-env-package', 'generic', 'github'),
];

const listFilesets = (filesets: FilesetOutput[]) =>
  http.get('*/apis/files/v2/workspaces/:workspace/filesets', ({ request }) => {
    const params = new URL(request.url).searchParams;
    const term = (params.get('filter[name][$like]') ?? '').replaceAll('%', '').toLowerCase();
    const purposes = [
      ...params.getAll('filter[purpose]'),
      ...params.getAll('filter[purpose][$in]'),
    ];
    const data = filesets.filter(
      (f) =>
        f.name.toLowerCase().includes(term) &&
        (purposes.length === 0 || purposes.includes(f.purpose))
    );
    return HttpResponse.json({
      data,
      pagination: {
        page: 1,
        page_size: 20,
        current_page_size: data.length,
        total_pages: 1,
        total_results: data.length,
      },
    });
  });

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
    msw: { handlers: [listFilesets(FILESETS)] },
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
  parameters: { msw: { handlers: [listFilesets([])] } },
};

export const WithError: Story = {
  args: { errorText: 'Select a training dataset' },
};
