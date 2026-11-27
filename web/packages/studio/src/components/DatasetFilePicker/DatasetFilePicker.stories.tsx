// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetFileValue } from '@nemo/common/src/components/FilesetPicker/FilesetFilePicker';
import {
  listFilesetFilesHandler,
  listFilesetsHandler,
  STORY_FILES,
  STORY_FILESETS,
} from '@nemo/common/src/components/FilesetPicker/storyData';
import { ToastProvider } from '@nemo/common/src/providers/toast/ToastProvider';
import type { FilesetFileOutput, FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { Panel, Text } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { DatasetFilePicker } from '@studio/components/DatasetFilePicker';
import { http, HttpResponse } from 'msw';
import { type ReactElement, useState } from 'react';

const FILESETS_URL = '*/apis/files/v2/workspaces/:workspace/filesets';

// In memory, so a dataset created from the New row is listed and can be looked up afterwards, and
// the files uploaded into it from the panel's Upload tab can be picked.
const datasets: FilesetOutput[] = [...STORY_FILESETS];
const filesByDataset: Record<string, FilesetFileOutput[]> = { ...STORY_FILES };

const created = (name: string): FilesetOutput => ({
  id: `default/${name}`,
  name,
  workspace: 'default',
  description: '',
  purpose: 'dataset',
  storage: { type: 'local', path: name },
  metadata: {},
  custom_fields: {},
  project: 'default',
  created_at: '2026-11-27T00:00:00Z',
  updated_at: '2026-11-27T00:00:00Z',
});

const handlers = [
  listFilesetFilesHandler(filesByDataset),
  listFilesetsHandler(datasets),
  http.get(`${FILESETS_URL}/:name`, ({ params }) => {
    const found = datasets.find((fileset) => fileset.name === params.name);
    return found
      ? HttpResponse.json(found)
      : HttpResponse.json({ detail: 'Not found' }, { status: 404 });
  }),
  http.post(FILESETS_URL, async ({ request }) => {
    const body: unknown = await request.json();
    const name =
      typeof body === 'object' && body !== null && 'name' in body && typeof body.name === 'string'
        ? body.name
        : 'new-dataset';
    const fileset = created(name);
    datasets.unshift(fileset);
    return HttpResponse.json(fileset);
  }),
  http.put(`${FILESETS_URL}/:name/-/*`, async ({ request, params }) => {
    const name = String(params.name);
    const path = decodeURIComponent(new URL(request.url).pathname.split('/-/')[1] ?? '');
    const size = (await request.blob()).size;
    filesByDataset[name] = [
      ...(filesByDataset[name] ?? []).filter((file) => file.path !== path),
      { path, size, file_ref: path, file_url: path },
    ];
    return new HttpResponse(null, { status: 200 });
  }),
  http.get('*/apis/secrets/v2/workspaces/:workspace/secrets', () =>
    HttpResponse.json({ data: [] })
  ),
];

function Harness(): ReactElement {
  const [value, setValue] = useState<FilesetFileValue>({ filesetRef: null, filePath: null });
  return (
    <ToastProvider>
      <Panel className="max-w-[640px]">
        <DatasetFilePicker
          workspace="default"
          label="Evaluation dataset"
          acceptedFileExtensions={['.csv', '.parquet']}
          value={value}
          onChange={setValue}
        />
        <Text kind="body/regular/xs" className="mt-4 block text-secondary">
          value: {JSON.stringify(value)}
        </Text>
      </Panel>
    </ToastProvider>
  );
}

const meta = {
  // Shares the FilesetPicker's title so these sit beside its stories.
  title: 'UI/FilesetPicker',
  component: Harness,
  parameters: { msw: { handlers } },
} satisfies Meta<typeof Harness>;

export default meta;
type Story = StoryObj<typeof meta>;

/** Pick `sft-chat-v3` and a file, or "New dataset" and upload files to create one, then pick its file. */
export const PickADatasetFile: Story = {};
