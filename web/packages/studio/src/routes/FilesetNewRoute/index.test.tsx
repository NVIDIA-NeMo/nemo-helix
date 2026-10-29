// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  checkDatasetQuality,
  type DatasetQualityReport,
} from '@nemo/common/src/utils/datasetQuality';
import {
  getFilesListFilesetsQueryKey,
  getFilesRetrieveFilesetQueryKey,
} from '@nemo/sdk/generated/platform/files';
import { ROUTE_PARAMS, ROUTES } from '@studio/constants/routes';
import { dataset } from '@studio/mocks/datasets';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { FilesetNewRoute } from '@studio/routes/FilesetNewRoute';
import { mockUseNavigate, mockUseParams } from '@studio/tests/util/mockUseParams';
import { render, renderRoute as renderRouteWithRouter, screen } from '@studio/tests/util/render';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { act, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import * as reactRouter from 'react-router';

vi.mock('@nemo/common/src/utils/datasetQuality', () => ({
  checkDatasetQuality: vi.fn(),
}));

const mockCheckDatasetQuality = vi.mocked(checkDatasetQuality);

function makeQualityReport(overrides: Partial<DatasetQualityReport> = {}): DatasetQualityReport {
  return {
    fileName: 'train.jsonl',
    hasErrors: false,
    hasWarnings: false,
    issues: [],
    scannedLines: 10,
    totalLines: 10,
    ...overrides,
  };
}

const renderRoute = () => {
  return render(
    <TestProviders>
      <FilesetNewRoute />
    </TestProviders>
  );
};

describe('FilesetNewRoute', () => {
  beforeEach(() => {
    mockUseParams({
      [ROUTE_PARAMS.workspace]: workspace1.workspace,
    });
    mockUseNavigate();
  });

  it('renders the side panel with Create Fileset heading', async () => {
    renderRoute();

    const heading = await screen.findByTestId('nv-side-panel-heading');
    expect(heading).toHaveTextContent('Create Fileset');
  });

  it('shows custom fileset form by default', async () => {
    renderRoute();

    expect(await screen.findByRole('textbox', { name: 'Fileset Name' })).toBeInTheDocument();
    expect(
      await screen.findByRole('textbox', { name: 'Description (optional)' })
    ).toBeInTheDocument();
    expect(await screen.findByText('Source')).toBeInTheDocument();
    expect(await screen.findByRole('tab', { name: 'Upload' })).toBeInTheDocument();
    expect(await screen.findByRole('tab', { name: 'External' })).toBeInTheDocument();
  });

  it('renders Purpose selector with Generic, Dataset, and Model options', async () => {
    renderRoute();

    // Heading + all three radio cards render with their labels and descriptions
    expect(await screen.findByText('Purpose')).toBeInTheDocument();
    expect(await screen.findByRole('radio', { name: 'Generic' })).toBeInTheDocument();
    expect(await screen.findByRole('radio', { name: 'Dataset' })).toBeInTheDocument();
    expect(await screen.findByRole('radio', { name: 'Model' })).toBeInTheDocument();

    // The lock-in note should be visible so users know purpose is immutable
    expect(
      await screen.findByText(
        /Purpose determines which metadata fields are available and can't be changed/i
      )
    ).toBeInTheDocument();
  });

  it('defaults Purpose selection to Dataset', async () => {
    renderRoute();

    const datasetCard = await screen.findByRole('radio', { name: 'Dataset' });
    expect(datasetCard).toBeChecked();
  });

  it('shows Create Fileset and Cancel in the footer', async () => {
    renderRoute();

    expect(await screen.findByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    expect(await screen.findByRole('button', { name: 'Create Fileset' })).toBeInTheDocument();
  });

  it('shows External storage URL and secret select when External tab is selected', async () => {
    const user = userEvent.setup();
    renderRoute();

    await user.click(await screen.findByRole('tab', { name: 'External' }));

    expect(await screen.findByRole('textbox', { name: 'URL' })).toBeInTheDocument();
    expect(await screen.findByRole('combobox', { name: 'Secret Key' })).toBeInTheDocument();
  });

  it('opens Create Secret modal when New Secret is chosen from the secret dropdown', async () => {
    const user = userEvent.setup();
    renderRoute();

    await user.click(await screen.findByRole('tab', { name: 'External' }));
    await user.click(await screen.findByRole('combobox', { name: 'Secret Key' }));
    await user.click(await screen.findByRole('menuitem', { name: 'New Secret' }));

    const secretDialog = await screen.findByRole('dialog', { name: 'Create Secret' });
    expect(secretDialog).toBeInTheDocument();
    await user.click(within(secretDialog).getByRole('button', { name: 'Cancel' }));
  });

  it('shows NGC API key label when External URL is an NGC catalog URL', async () => {
    const user = userEvent.setup();
    renderRoute();

    await user.click(await screen.findByRole('tab', { name: 'External' }));

    const urlInput = await screen.findByRole('textbox', { name: 'URL' });
    await user.clear(urlInput);
    await user.type(
      urlInput,
      'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ngc-apps/resources/some-dataset'
    );

    expect(screen.getAllByText(/NGC API key/i).length).toBeGreaterThan(0);
  });

  it('shows HuggingFace token (optional) label when External URL is a Hugging Face URL', async () => {
    const user = userEvent.setup();
    renderRoute();

    await user.click(await screen.findByRole('tab', { name: 'External' }));

    const urlInput = await screen.findByRole('textbox', { name: 'URL' });
    await user.clear(urlInput);
    await user.type(urlInput, 'https://huggingface.co/datasets/org/repo-name');

    expect(screen.getAllByText(/HuggingFace token \(optional\)/i).length).toBeGreaterThan(0);
  });

  it('shows inline validation error for uppercase letters in fileset name', async () => {
    const user = userEvent.setup();
    renderRoute();

    const nameInput = await screen.findByRole('textbox', { name: 'Fileset Name' });
    await user.type(nameInput, 'tiny-gpt2-A');
    await user.tab();

    expect(
      await screen.findByText(/Name must be lowercase\. Try "tiny-gpt2-a"\./)
    ).toBeInTheDocument();
  });

  it('shows inline validation error when fileset name starts with a digit', async () => {
    const user = userEvent.setup();
    renderRoute();

    const nameInput = await screen.findByRole('textbox', { name: 'Fileset Name' });
    await user.type(nameInput, '1invalid');
    await user.tab();

    expect(await screen.findByText(/must start with a lowercase letter/i)).toBeInTheDocument();
  });

  it('accepts a valid fileset name without showing an error', async () => {
    const user = userEvent.setup();
    renderRoute();

    const nameInput = await screen.findByRole('textbox', { name: 'Fileset Name' });
    await user.type(nameInput, 'tiny-gpt2-a');
    await user.tab();

    expect(screen.queryByText(/must start with a lowercase letter/i)).not.toBeInTheDocument();
  });

  describe('name uniqueness', () => {
    const FILESETS_URL = mockApiUrl(getFilesListFilesetsQueryKey, ':workspace');
    const FILESET_URL = mockApiUrl(getFilesRetrieveFilesetQueryKey, ':workspace', ':name');

    it('says where the name must be unique before anything is typed', async () => {
      renderRoute();

      expect(
        await screen.findByText(
          (_, element) =>
            element?.getAttribute('data-testid') === 'nv-form-field-helper' &&
            element.textContent === 'Fileset names must be unique within the default workspace.'
        )
      ).toBeInTheDocument();
    });

    it('flags a name already used in the workspace and blocks submit', async () => {
      const user = userEvent.setup();
      renderRoute();

      await user.type(await screen.findByRole('textbox', { name: 'Fileset Name' }), dataset.name);

      expect(
        await screen.findByText(
          `A fileset named ${dataset.name} already exists in the default workspace`
        )
      ).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Create Fileset' })).toBeDisabled();
    });

    it('confirms a free name with the workspace it will be created in', async () => {
      const user = userEvent.setup();
      renderRoute();

      await user.type(await screen.findByRole('textbox', { name: 'Fileset Name' }), 'fresh-name');

      await waitFor(() =>
        expect(screen.getByTestId('nv-form-field-helper')).toHaveTextContent(
          'Your fileset will be created as fresh-name in the default workspace'
        )
      );
      expect(screen.getByRole('button', { name: 'Create Fileset' })).toBeEnabled();
    });

    it('puts a 409 on submit on the name field and rechecks the name', async () => {
      // The check first sees `raced` as free; another create lands before ours.
      let createAttempted = false;
      server.use(
        http.get(FILESETS_URL, ({ request }) => {
          const name = new URL(request.url).searchParams.get('filter[name]');
          const taken = createAttempted && name === 'raced';
          return HttpResponse.json({
            data: taken ? [{ ...dataset, name: 'raced' }] : [],
            pagination: { total_results: taken ? 1 : 0 },
          });
        }),
        http.post(FILESETS_URL, () => {
          createAttempted = true;
          return HttpResponse.json({ detail: 'already exists' }, { status: 409 });
        })
      );
      const user = userEvent.setup();
      renderRoute();

      await user.type(await screen.findByRole('textbox', { name: 'Fileset Name' }), 'raced');
      await screen.findByText(/will be created as/);
      await user.click(screen.getByRole('button', { name: 'Create Fileset' }));

      expect(
        await screen.findByText('A fileset named raced already exists in the default workspace')
      ).toBeInTheDocument();
      // The cached "free" answer is dropped, so the button does not invite another 409.
      await waitFor(() =>
        expect(screen.getByRole('button', { name: 'Create Fileset' })).toBeDisabled()
      );
    });

    it('retries only the upload after it fails, without creating the fileset again', async () => {
      mockCheckDatasetQuality.mockResolvedValue(makeQualityReport());
      const navigate = vi.fn();
      mockUseNavigate(navigate);
      let creates = 0;
      let uploadFails = true;
      server.use(
        http.get(FILESETS_URL, ({ request }) => {
          const name = new URL(request.url).searchParams.get('filter[name]');
          const exists = creates > 0 && name === 'my-data';
          return HttpResponse.json({
            data: exists ? [{ ...dataset, name: 'my-data' }] : [],
            pagination: { total_results: exists ? 1 : 0 },
          });
        }),
        http.post(FILESETS_URL, () => {
          creates += 1;
          return creates > 1
            ? HttpResponse.json({ detail: 'already exists' }, { status: 409 })
            : HttpResponse.json({ ...dataset, name: 'my-data' });
        }),
        http.put(`${FILESET_URL}/-/*`, () =>
          uploadFails
            ? HttpResponse.json({ detail: 'upload failed' }, { status: 500 })
            : HttpResponse.json({})
        )
      );
      const user = userEvent.setup();
      renderRoute();

      await user.type(await screen.findByRole('textbox', { name: 'Fileset Name' }), 'my-data');
      await screen.findByText(/will be created as/);
      // querySelector is the only reliable way to reach the file input in jsdom.
      // eslint-disable-next-line testing-library/no-node-access
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      await user.upload(
        input,
        new File(['{"prompt":"q","completion":"a"}'], 'train.jsonl', {
          type: 'application/jsonl',
        })
      );
      await user.click(screen.getByRole('button', { name: 'Create Fileset' }));

      // The fileset it just created does not count against its own name.
      await waitFor(() =>
        expect(screen.getByRole('button', { name: 'Create Fileset' })).toBeEnabled()
      );
      expect(creates).toBe(1);
      expect(screen.queryByText(/already exists/)).not.toBeInTheDocument();

      uploadFails = false;
      await user.click(screen.getByRole('button', { name: 'Create Fileset' }));

      await waitFor(() => expect(navigate).toHaveBeenCalled());
      expect(creates).toBe(1);
    });
  });

  describe('dataset quality validation', () => {
    beforeEach(() => {
      mockCheckDatasetQuality.mockReset();
    });

    function makeJsonlFile(name = 'train.jsonl'): File {
      return new File(['{"prompt":"q","completion":"a"}'], name, {
        type: 'application/x-jsonlines',
      });
    }

    async function uploadFile(user: ReturnType<typeof userEvent.setup>, file: File) {
      // The Upload component renders a visually hidden file input with no accessible name;
      // querySelector is the only reliable way to reach it in jsdom.
      // eslint-disable-next-line testing-library/no-node-access
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      await user.upload(input, file);
    }

    it('shows quality report after uploading a JSONL file when purpose is Dataset', async () => {
      mockCheckDatasetQuality.mockResolvedValue(
        makeQualityReport({ hasErrors: false, hasWarnings: false })
      );
      const user = userEvent.setup();
      renderRoute();

      await uploadFile(user, makeJsonlFile());

      expect(await screen.findByText(/all quality checks passed/i)).toBeInTheDocument();
    });

    it('shows error issues from the quality report', async () => {
      mockCheckDatasetQuality.mockResolvedValue(
        makeQualityReport({
          hasErrors: true,
          issues: [
            {
              severity: 'error',
              code: 'INVALID_JSON_LINES',
              message: '2 lines could not be parsed as JSON objects.',
              affectedLines: [3, 7],
              count: 2,
            },
          ],
        })
      );
      const user = userEvent.setup();
      renderRoute();

      await uploadFile(user, makeJsonlFile());

      expect(
        await screen.findByText(/2 lines could not be parsed as JSON objects/i)
      ).toBeInTheDocument();
    });

    it('shows warning issues from the quality report', async () => {
      mockCheckDatasetQuality.mockResolvedValue(
        makeQualityReport({
          hasWarnings: true,
          issues: [
            {
              severity: 'warning',
              code: 'UNKNOWN_SCHEMA',
              message: 'No recognized fine-tuning schema detected.',
            },
          ],
        })
      );
      const user = userEvent.setup();
      renderRoute();

      await uploadFile(user, makeJsonlFile());

      expect(
        await screen.findByText(/No recognized fine-tuning schema detected/i)
      ).toBeInTheDocument();
    });

    it('disables the Create Fileset button when quality report has errors', async () => {
      mockCheckDatasetQuality.mockResolvedValue(
        makeQualityReport({
          hasErrors: true,
          issues: [{ severity: 'error', code: 'EMPTY_FILE', message: 'File is empty.' }],
        })
      );
      const user = userEvent.setup();
      renderRoute();

      await uploadFile(user, makeJsonlFile());

      await screen.findByText(/File is empty/i);
      expect(await screen.findByRole('button', { name: 'Create Fileset' })).toBeDisabled();
    });

    it('does not disable submit for warning-only reports', async () => {
      mockCheckDatasetQuality.mockResolvedValue(
        makeQualityReport({
          hasWarnings: true,
          issues: [
            {
              severity: 'warning',
              code: 'LONG_ENTRIES',
              message: '1 row may exceed context window.',
            },
          ],
        })
      );
      const user = userEvent.setup();
      renderRoute();

      await uploadFile(user, makeJsonlFile());

      await screen.findByText(/1 row may exceed context window/i);
      expect(await screen.findByRole('button', { name: 'Create Fileset' })).not.toBeDisabled();
    });

    it('does not show quality report section when purpose is not Dataset', async () => {
      mockCheckDatasetQuality.mockResolvedValue(makeQualityReport());
      const user = userEvent.setup();
      renderRoute();

      // Switch to Generic purpose
      await user.click(await screen.findByRole('radio', { name: 'Generic' }));
      await uploadFile(user, makeJsonlFile());

      expect(screen.queryByText(/all quality checks passed/i)).not.toBeInTheDocument();
      expect(mockCheckDatasetQuality).not.toHaveBeenCalled();
    });

    it('shows scanned-lines note when file has more lines than the scan limit', async () => {
      mockCheckDatasetQuality.mockResolvedValue(
        makeQualityReport({ scannedLines: 1000, totalLines: 5000 })
      );
      const user = userEvent.setup();
      renderRoute();

      await uploadFile(user, makeJsonlFile());

      expect(await screen.findByText(/Scanned first 1,000 of 5,000 lines/i)).toBeInTheDocument();
    });
  });
});

describe('FilesetNewRoute close behavior', () => {
  // This block needs the real router hooks, not the module-level mocks the suite
  // above installs — the bug only appears when a genuine navigation runs.
  beforeEach(async () => {
    const actual = await vi.importActual<typeof import('react-router')>('react-router');
    vi.mocked(reactRouter.useParams).mockImplementation(actual.useParams);
    vi.mocked(reactRouter.useNavigate).mockImplementation(actual.useNavigate);
  });

  it('closes exactly once without flashing back open', async () => {
    const user = userEvent.setup();

    // The `routes` option builds a data router, which is the only path that honors
    // `flushSync`; the declarative MemoryRouter fallback drops the option.
    renderRouteWithRouter(undefined, {
      history: `/workspaces/${workspace1.workspace}/filesets/new`,
      routes: [
        { path: ROUTES.workspace.filesetNew, element: <FilesetNewRoute /> },
        { path: ROUTES.workspace.filesets, element: <div>Filesets list</div> },
      ],
    });

    const dialog = await screen.findByRole('dialog');
    await waitFor(() => expect(dialog).toHaveAttribute('data-state', 'open'));

    // Count re-shows rather than watching `data-state`: the shared test setup stubs
    // `MutationObserver` to a no-op. `showModal` is the direct signal — `useDialog`
    // calls it when it sees a stale `open === true` against a closed <dialog>, which
    // is the flash.
    const showModalSpy = vi.spyOn(HTMLDialogElement.prototype, 'showModal');

    await user.click(screen.getByRole('button', { name: /close side panel/i }));

    // KUI's animate-out is a real 200ms timeout, so the resulting updates land outside
    // React's event loop. Advance real time inside `act` to flush them.
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 500));
    });

    expect(showModalSpy).not.toHaveBeenCalled();
    await screen.findByText('Filesets list');

    showModalSpy.mockRestore();
  });
});
