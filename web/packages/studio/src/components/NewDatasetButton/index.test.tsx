// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MockToastProvider } from '@nemo/common/src/tests/MockToastProvider';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { NewDatasetButton } from '@studio/components/NewDatasetButton';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { FilesetDetailTab } from '@studio/routes/FilesetDetailRoute/constants';
import { getDatasetDetailRoute } from '@studio/routes/utils';
import { mockUseNavigate, mockUseParams } from '@studio/tests/util/mockUseParams';
import { render, screen } from '@studio/tests/util/render';
import { fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const WS = 'default';
const FILESETS_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`;
const HF_URL = 'https://huggingface.co/datasets/tatsu-lab/alpaca';

const emptyList = () =>
  http.get(FILESETS_URL, () =>
    HttpResponse.json({
      data: [],
      pagination: {
        page: 1,
        page_size: 1,
        current_page_size: 0,
        total_pages: 1,
        total_results: 0,
      },
    })
  );

// The server echoes the created storage type, so navigation can tell external from local.
const createEndpoints = () => [
  http.post(FILESETS_URL, async ({ request }) => {
    const body = (await request.json()) as Pick<FilesetOutput, 'name' | 'storage'>;
    // An omitted storage means the platform default, which is local.
    return HttpResponse.json({
      id: `fs-${body.name}`,
      workspace: WS,
      description: '',
      purpose: 'dataset',
      metadata: {},
      custom_fields: {},
      project: WS,
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:00:00Z',
      ...body,
      storage: body.storage ?? { type: 'local' },
    });
  }),
  http.put(`${FILESETS_URL}/:name/-/*`, () => new HttpResponse(null, { status: 200 })),
];

const renderButton = () =>
  render(
    <MockToastProvider>
      <NewDatasetButton />
    </MockToastProvider>
  );

describe('NewDatasetButton', () => {
  const navigate = vi.fn();

  beforeEach(() => {
    navigate.mockReset();
    mockUseParams({ [ROUTE_PARAMS.workspace]: WS });
    mockUseNavigate(navigate);
    server.use(emptyList(), ...createEndpoints());
  });

  it('opens the dataset creation panel and Cancel closes it', async () => {
    const user = userEvent.setup();
    renderButton();

    expect(screen.queryByRole('tab', { name: 'HuggingFace' })).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Create Dataset' }));

    expect(await screen.findByRole('tab', { name: 'HuggingFace' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Upload' })).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await waitFor(() =>
      expect(screen.queryByRole('tab', { name: 'HuggingFace' })).not.toBeInTheDocument()
    );
    expect(navigate).not.toHaveBeenCalled();
  });

  it('navigates to the dataset detail Card tab after creating an external dataset', async () => {
    const user = userEvent.setup();
    renderButton();

    await user.click(screen.getByRole('button', { name: 'Create Dataset' }));
    const url = await screen.findByRole('textbox', { name: 'HuggingFace URL' });
    await user.click(url);
    await user.paste(HF_URL);
    await waitFor(() =>
      expect(screen.getByRole('textbox', { name: 'Name' })).toHaveValue('alpaca')
    );

    // The panel's own submit button shares the trigger's label.
    await user.click(screen.getByRole('button', { name: 'Create Dataset' }));

    await waitFor(() => expect(navigate).toHaveBeenCalledTimes(1));
    expect(navigate).toHaveBeenCalledWith(
      getDatasetDetailRoute(WS, 'alpaca', { tab: FilesetDetailTab.Card })
    );
  });

  it('navigates to the dataset detail Files tab after creating an upload dataset', async () => {
    const user = userEvent.setup();
    renderButton();

    await user.click(screen.getByRole('button', { name: 'Create Dataset' }));
    await user.click(await screen.findByRole('tab', { name: 'Upload' }));
    fireEvent.change(screen.getByTestId('create-dataset-files-input'), {
      target: { files: [new File(['x'], 'train-data.jsonl')] },
    });
    await waitFor(() =>
      expect(screen.getByRole('textbox', { name: 'Name' })).toHaveValue('train-data')
    );

    await user.click(screen.getByRole('button', { name: 'Create Dataset' }));

    await waitFor(() => expect(navigate).toHaveBeenCalledTimes(1));
    expect(navigate).toHaveBeenCalledWith(
      getDatasetDetailRoute(WS, 'train-data', { tab: FilesetDetailTab.Files })
    );
  });
});
