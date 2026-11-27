// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ToastProvider } from '@nemo/common/src/providers/toast/ToastProvider';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { Panel, Text } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { DatasetPicker } from '@studio/components/DatasetPicker';
import { http, HttpResponse } from 'msw';
import { type ReactElement, useState } from 'react';

const FILESETS_URL = '*/apis/files/v2/workspaces/:workspace/filesets';

const dataset = (name: string, storageType: FilesetOutput['storage']['type']): FilesetOutput =>
  ({
    id: `default/${name}`,
    name,
    workspace: 'default',
    description: '',
    purpose: 'dataset',
    storage: { type: storageType },
    metadata: {},
    custom_fields: {},
    project: 'default',
    created_at: '2026-11-01T00:00:00Z',
    updated_at: '2026-11-01T00:00:00Z',
  }) as FilesetOutput;

const page = (data: unknown[]) =>
  HttpResponse.json({
    data,
    pagination: {
      page: 1,
      page_size: 20,
      current_page_size: data.length,
      total_pages: 1,
      total_results: data.length,
    },
  });

/** An in-memory files API: datasets created in the panel show up in the picker afterwards. */
const filesApi = (initial: FilesetOutput[]) => {
  const datasets = [...initial];
  return [
    http.get(FILESETS_URL, ({ request }) => {
      const params = new URL(request.url).searchParams;
      const exact = params.get('filter[name]');
      const term = (params.get('filter[name][$like]') ?? '').replaceAll('%', '').toLowerCase();
      return page(
        datasets.filter((d) =>
          exact !== null ? d.name === exact : d.name.toLowerCase().includes(term)
        )
      );
    }),
    http.get(`${FILESETS_URL}/:name`, ({ params }) => {
      const found = datasets.find((d) => d.name === params.name);
      return found
        ? HttpResponse.json(found)
        : HttpResponse.json({ detail: 'Not found' }, { status: 404 });
    }),
    http.post(FILESETS_URL, async ({ request }) => {
      const body = (await request.json()) as {
        name: string;
        storage?: { type?: FilesetOutput['storage']['type'] };
      };
      const created = dataset(body.name, body.storage?.type ?? 'local');
      datasets.unshift(created);
      return HttpResponse.json(created);
    }),
    http.put(`${FILESETS_URL}/:name/-/*`, () => new HttpResponse(null, { status: 200 })),
    http.get('*/apis/secrets/v2/workspaces/:workspace/secrets', () => page([])),
  ];
};

function Harness(): ReactElement {
  const [value, setValue] = useState<string | null>(null);
  return (
    <ToastProvider>
      <Panel className="max-w-[640px]">
        <DatasetPicker
          workspace="default"
          label="Training dataset"
          value={value}
          onChange={setValue}
        />
        <Text kind="body/regular/xs" className="mt-4 block text-secondary">
          value: {value ?? 'null'}
        </Text>
      </Panel>
    </ToastProvider>
  );
}

const meta = {
  // Shares the FilesetPicker's title so these sit beside its stories.
  title: 'UI/FilesetPicker',
  component: Harness,
  parameters: {
    msw: {
      handlers: filesApi([
        dataset('sft-chat-v3', 'local'),
        dataset('support-transcripts-raw', 's3'),
        dataset('nemotron-personas', 'huggingface'),
      ]),
    },
  },
} satisfies Meta<typeof Harness>;

export default meta;
type Story = StoryObj<typeof meta>;

/** Pick "New dataset" to open the Create Dataset panel; the dataset it creates becomes the value. */
export const CreateDataset: Story = {};

export const CreateDatasetInEmptyWorkspace: Story = {
  parameters: { msw: { handlers: filesApi([]) } },
};
