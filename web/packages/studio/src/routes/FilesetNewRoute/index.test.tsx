// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  checkDatasetQuality,
  type DatasetQualityReport,
} from '@nemo/common/src/utils/datasetQuality';
import { ROUTE_PARAMS, ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { FilesetNewRoute } from '@studio/routes/FilesetNewRoute';
import { mockUseNavigate, mockUseParams } from '@studio/tests/util/mockUseParams';
import { render, renderRoute as renderRouteWithRouter, screen } from '@studio/tests/util/render';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { act, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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

  it('renders Custom Fileset and Sample Dataset tabs', async () => {
    renderRoute();

    expect(await screen.findByRole('radio', { name: 'Custom Fileset' })).toBeInTheDocument();
    expect(await screen.findByRole('radio', { name: 'Sample Dataset' })).toBeInTheDocument();
  });

  it('shows custom fileset form when Custom Fileset tab is selected', async () => {
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

  it('switches to Sample Dataset tab and shows sample dataset cards', async () => {
    const user = userEvent.setup();
    renderRoute();

    await user.click(await screen.findByText('Sample Dataset'));

    expect(
      await screen.findByText('Choose from the following pre-configured sample datasets.')
    ).toBeInTheDocument();
    expect(await screen.findByText('Q&A Generation Dataset')).toBeInTheDocument();
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

    it('clears quality reports when switching to the Sample Dataset tab', async () => {
      mockCheckDatasetQuality.mockResolvedValue(
        makeQualityReport({ hasErrors: false, hasWarnings: false })
      );
      const user = userEvent.setup();
      renderRoute();

      await uploadFile(user, makeJsonlFile());
      await screen.findByText(/all quality checks passed/i);

      await user.click(await screen.findByText('Sample Dataset'));

      expect(screen.queryByText(/all quality checks passed/i)).not.toBeInTheDocument();
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
