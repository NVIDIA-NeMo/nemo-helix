// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { DatasetPicker } from '@studio/components/DatasetPicker';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { render, screen } from '@studio/tests/util/render';
import { waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { useState } from 'react';

const WS = 'default';
const FILESETS_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`;
const SECRETS_URL = `${PLATFORM_BASE_URL}/apis/secrets/v2/workspaces/:workspace/secrets`;
const HF_URL = 'https://huggingface.co/datasets/tatsu-lab/alpaca';

const filesetRecord = (name: string): FilesetOutput =>
  ({
    id: `fs-${name}`,
    name,
    workspace: WS,
    description: '',
    purpose: 'dataset',
    storage: { type: 'local' },
    metadata: {},
    custom_fields: {},
    project: WS,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
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

function Host({ onChange }: { onChange: (value: string | null) => void }) {
  const [value, setValue] = useState<string | null>(null);
  return (
    <DatasetPicker
      workspace={WS}
      label="Training dataset"
      value={value}
      onChange={(next) => {
        onChange(next);
        setValue(next);
      }}
    />
  );
}

describe('DatasetPicker', () => {
  let listRequests = 0;

  beforeEach(() => {
    const created = new Map<string, FilesetOutput>();
    listRequests = 0;
    server.use(
      // Serves both the picker's list and the panel's name check (`filter[name]`).
      http.get(FILESETS_URL, ({ request }) => {
        const name = new URL(request.url).searchParams.get('filter[name]');
        if (name === null) listRequests += 1;
        return page(name === null ? [...created.values()] : []);
      }),
      http.get(`${FILESETS_URL}/:name`, ({ params }) => {
        const fileset = created.get(params.name as string);
        return fileset
          ? HttpResponse.json(fileset)
          : HttpResponse.json({ detail: 'not found' }, { status: 404 });
      }),
      http.post(FILESETS_URL, async ({ request }) => {
        const body = (await request.json()) as { name: string };
        const fileset = filesetRecord(body.name);
        created.set(body.name, fileset);
        return HttpResponse.json(fileset);
      }),
      http.get(SECRETS_URL, () => page([]))
    );
  });

  it('opens the Create Dataset panel from the New row and selects the created dataset', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Host onChange={onChange} />);
    const input = screen.getByRole('combobox', { name: 'Training dataset' });

    await user.click(input);
    await user.click(await screen.findByRole('option', { name: 'New dataset' }));
    expect(onChange).not.toHaveBeenCalled();

    const url = await screen.findByRole('textbox', { name: 'HuggingFace URL' });
    await user.click(url);
    await user.paste(HF_URL);
    await waitFor(() =>
      expect(screen.getByRole('textbox', { name: 'Name' })).toHaveValue('alpaca')
    );
    const listRequestsBeforeCreate = listRequests;
    await user.click(screen.getByRole('button', { name: 'Create Dataset' }));

    await waitFor(() => expect(onChange).toHaveBeenCalledWith(`${WS}/alpaca`));
    await waitFor(() =>
      expect(screen.queryByRole('button', { name: 'Create Dataset' })).not.toBeInTheDocument()
    );
    expect(input).toHaveValue('alpaca');

    // The create refetches the picker's list, so the new dataset is a regular, selected row.
    await waitFor(() => expect(listRequests).toBeGreaterThan(listRequestsBeforeCreate));
    await user.click(input);
    expect(await screen.findByRole('option', { name: /alpaca/ })).toHaveAttribute(
      'aria-selected',
      'true'
    );
  });

  it('leaves the selection alone when the panel is cancelled', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Host onChange={onChange} />);

    await user.click(screen.getByRole('combobox', { name: 'Training dataset' }));
    await user.click(await screen.findByRole('option', { name: 'New dataset' }));
    await user.click(await screen.findByRole('button', { name: 'Cancel' }));

    await waitFor(() =>
      expect(screen.queryByRole('button', { name: 'Create Dataset' })).not.toBeInTheDocument()
    );
    expect(onChange).not.toHaveBeenCalled();
  });
});
