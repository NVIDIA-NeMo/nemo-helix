// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetFileValue } from '@nemo/common/src/components/FilesetPicker/FilesetFilePicker';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import type { CreateDatasetPanelProps } from '@studio/components/CreateDatasetPanel';
import { DatasetFilePicker } from '@studio/components/DatasetFilePicker';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { render, screen } from '@studio/tests/util/render';
import { waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { useState } from 'react';

const WS = 'default';
const FILESETS_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`;

const dataset = (name: string): FilesetOutput => ({
  id: `fs-${name}`,
  name,
  workspace: WS,
  description: '',
  purpose: 'dataset',
  storage: { type: 'local', path: name },
  metadata: {},
  custom_fields: {},
  project: WS,
  created_at: '2026-11-01T00:00:00Z',
  updated_at: '2026-11-01T00:00:00Z',
});

// The panel has its own tests; this stand-in exposes just the callbacks the picker wires.
vi.mock('@studio/components/CreateDatasetPanel', () => ({
  CreateDatasetPanel: ({ open, onCreated, onClose }: CreateDatasetPanelProps) =>
    open ? (
      <button
        onClick={() => {
          onCreated(dataset('fresh'));
          onClose();
        }}
      >
        Create fresh
      </button>
    ) : null,
}));

interface HostProps {
  onChange: (value: FilesetFileValue) => void;
}

function Host({ onChange }: HostProps) {
  const [value, setValue] = useState<FilesetFileValue>({ filesetRef: null, filePath: null });
  return (
    <DatasetFilePicker
      workspace={WS}
      label="Evaluation dataset"
      acceptedFileExtensions={['.csv']}
      value={value}
      onChange={(next) => {
        onChange(next);
        setValue(next);
      }}
    />
  );
}

const page = (data: unknown[]) =>
  HttpResponse.json({ data, pagination: { page: 1, total_pages: 1 } });

describe('DatasetFilePicker', () => {
  beforeEach(() => {
    server.use(
      http.get(FILESETS_URL, () => page([dataset('evals')])),
      http.get(`${FILESETS_URL}/:name`, ({ params }) =>
        HttpResponse.json(dataset(String(params.name)))
      ),
      http.get(`${FILESETS_URL}/:name/files`, () =>
        HttpResponse.json({ data: [{ path: 'q.csv', size: 10, file_ref: 'q', file_url: 'q' }] })
      )
    );
  });

  it('selects a dataset created from the New row and then asks for a file', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Host onChange={onChange} />);

    await user.click(screen.getByRole('combobox', { name: 'Evaluation dataset' }));
    await user.click(await screen.findByRole('option', { name: 'New dataset' }));
    await user.click(await screen.findByRole('button', { name: 'Create fresh' }));

    expect(onChange).toHaveBeenLastCalledWith({ filesetRef: `${WS}/fresh`, filePath: null });
    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Evaluation dataset' })).toHaveValue('fresh')
    );
    expect(await screen.findByRole('combobox', { name: 'File' })).toBeInTheDocument();
  });
});
