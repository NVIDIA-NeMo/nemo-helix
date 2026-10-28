// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withOperators } from '@nemo/common/src/api/filterOperators';
import { UploadModalProvider } from '@nemo/common/src/components/UploadModal/Context/UploadModalProvider';
import { useUploadModalContext } from '@nemo/common/src/components/UploadModal/Context/useUploadModalContext';
import {
  uploadModalInitialState,
  type UploadModalState,
} from '@nemo/common/src/components/UploadModal/Context/useUploadModalReducer';
import { DatasetSelect } from '@nemo/common/src/components/UploadModal/DatasetUploader/Select';
import { filesListFilesetFiles, filesListFilesets } from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock the SDK hooks
vi.mock('@nemo/sdk/generated/platform/files', () => ({
  filesListFilesets: vi.fn(),
  getFilesListFilesetsQueryKey: vi.fn((workspace: string) => ['filesets', workspace]),
  filesListFilesetFiles: vi.fn(),
}));

const fileset = (name: string): FilesetOutput => ({
  id: `default/${name}`,
  name,
  workspace: 'default',
  description: '',
  purpose: 'dataset',
  storage: { type: 'local', path: '/data' } as const,
  metadata: {},
  custom_fields: {},
  project: 'default',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
});

const mockFilesets: FilesetOutput[] = [fileset('dataset1'), fileset('dataset2')];

const page = (data: FilesetOutput[], pageNumber = 1, totalPages = 1) =>
  ({ data, pagination: { page: pageNumber, total_pages: totalPages } }) as Awaited<
    ReturnType<typeof filesListFilesets>
  >;

// Helper component to access context in tests
const ContextReader = ({
  onContextChange,
}: {
  onContextChange: (state: UploadModalState) => void;
}) => {
  const [state] = useUploadModalContext();
  onContextChange(state);
  return null;
};

const createWrapper = (initialState?: Partial<UploadModalState>) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <UploadModalProvider initialState={{ ...uploadModalInitialState, ...initialState }}>
        {children}
      </UploadModalProvider>
    </QueryClientProvider>
  );
};

const filesetFile = (path: string) => ({ path, file_ref: `ref-${path}` });

describe('DatasetSelect', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(filesListFilesets).mockResolvedValue(page(mockFilesets));
    vi.mocked(filesListFilesetFiles).mockResolvedValue({ data: [] });
  });

  it('renders dataset select with datasets', () => {
    render(<DatasetSelect project="test-project" />, {
      wrapper: createWrapper(),
    });

    expect(screen.getByRole('combobox')).toBeInTheDocument();
  });

  it('queries filesets paginated, newest first, filtered by purpose', async () => {
    render(<DatasetSelect project="test-project" />, {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(filesListFilesets).toHaveBeenCalledWith(
        'test-project',
        expect.objectContaining({
          page: 1,
          page_size: 20,
          sort: '-created_at',
          filter: withOperators({ purpose: 'dataset' }),
        }),
        expect.anything()
      );
    });
  });

  it('searches filesets server-side', async () => {
    render(<DatasetSelect project="test-project" />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('combobox'));
    await user.type(screen.getByTestId('dataset-search'), 'data');

    await waitFor(() => {
      expect(filesListFilesets).toHaveBeenCalledWith(
        'test-project',
        expect.objectContaining({
          filter: withOperators({ name: { $like: '%data%' }, purpose: 'dataset' }),
        }),
        expect.anything()
      );
    });
  });

  it('shows loading state', async () => {
    vi.mocked(filesListFilesets).mockReturnValue(new Promise(() => {}));

    render(<DatasetSelect project="test-project" />, {
      wrapper: createWrapper({ allowNewDataset: false }),
    });

    await user.click(screen.getByRole('combobox'));
    expect(await screen.findByLabelText('Loading options')).toBeInTheDocument();
  });

  it('surfaces a failure message when the query fails', async () => {
    vi.mocked(filesListFilesets).mockRejectedValue(new Error('boom'));

    render(<DatasetSelect project="test-project" />, {
      wrapper: createWrapper({ allowNewDataset: false }),
    });

    await user.click(screen.getByRole('combobox'));
    expect(await screen.findByText('Failed to load filesets')).toBeInTheDocument();
    expect(screen.queryByText('No filesets found')).not.toBeInTheDocument();
  });

  it('updates context when dataset is selected', async () => {
    let contextState: UploadModalState | undefined;

    render(
      <>
        <DatasetSelect project="test-project" />
        <ContextReader onContextChange={(state) => (contextState = state)} />
      </>,
      {
        wrapper: createWrapper(),
      }
    );

    await user.click(screen.getByRole('combobox'));
    await user.click(await screen.findByRole('option', { name: 'dataset1' }));

    await waitFor(() => {
      expect(contextState?.dataset).toBeDefined();
    });

    expect(contextState?.dataset?.type).toBe('existing');
    expect(contextState?.dataset?.type === 'existing' && contextState.dataset.dataset.name).toBe(
      'dataset1'
    );
  });

  it('auto-selects the first root-level accepted file when autoSelectFirstAcceptable is set', async () => {
    vi.mocked(filesListFilesetFiles).mockResolvedValueOnce({
      data: [filesetFile('smaller_test.csv'), filesetFile('email_phishing_analyzer-eval.yml')],
    } as Awaited<ReturnType<typeof filesListFilesetFiles>>);

    let contextState: UploadModalState | undefined;
    render(
      <>
        <DatasetSelect project="test-project" />
        <ContextReader onContextChange={(state) => (contextState = state)} />
      </>,
      { wrapper: createWrapper({ autoSelectFirstAcceptable: true, acceptableFileTypes: ['.yml'] }) }
    );

    await user.click(screen.getByRole('combobox'));
    await user.click(await screen.findByRole('option', { name: 'dataset1' }));

    await waitFor(() => {
      expect(contextState?.selectedFiles).toHaveLength(1);
    });
    expect((contextState?.selectedFiles[0]?.file as { path?: string }).path).toBe(
      'email_phishing_analyzer-eval.yml'
    );
  });

  it('selects nothing when no root-level accepted file exists', async () => {
    vi.mocked(filesListFilesetFiles).mockResolvedValueOnce({
      data: [filesetFile('smaller_test.csv'), filesetFile('nested/config.yml')],
    } as Awaited<ReturnType<typeof filesListFilesetFiles>>);

    let contextState: UploadModalState | undefined;
    render(
      <>
        <DatasetSelect project="test-project" />
        <ContextReader onContextChange={(state) => (contextState = state)} />
      </>,
      { wrapper: createWrapper({ autoSelectFirstAcceptable: true, acceptableFileTypes: ['.yml'] }) }
    );

    await user.click(screen.getByRole('combobox'));
    await user.click(await screen.findByRole('option', { name: 'dataset1' }));

    await waitFor(() => {
      expect(contextState?.dataset?.type).toBe('existing');
    });
    expect(contextState?.selectedFiles).toHaveLength(0);
  });

  it('includes "New Dataset" option', async () => {
    render(<DatasetSelect project="test-project" />, {
      wrapper: createWrapper(),
    });

    await user.click(screen.getByRole('combobox'));

    // Queried by role, not text: a hidden native <option value="new"> carries the same
    // label, and only the menu item is exposed in the accessibility tree.
    expect(await screen.findByRole('option', { name: 'New Dataset' })).toBeInTheDocument();
  });

  it('updates context with new dataset type when New Dataset is selected', async () => {
    let contextState: UploadModalState | undefined;

    render(
      <>
        <DatasetSelect project="test-project" />
        <ContextReader onContextChange={(state) => (contextState = state)} />
      </>,
      {
        wrapper: createWrapper(),
      }
    );

    await user.click(screen.getByRole('combobox'));

    expect(await screen.findByRole('option', { name: 'New Dataset' })).toBeInTheDocument();

    await user.click(screen.getByRole('option', { name: 'New Dataset' }));

    await waitFor(() => {
      expect(contextState?.dataset).toBeDefined();
    });

    expect(contextState?.dataset?.type).toBe('new');
    expect(contextState?.dataset?.type === 'new' && contextState.dataset.name).toBe('');
  });

  it('can be disabled', async () => {
    render(<DatasetSelect project="test-project" disabled />, {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      const select = screen.getByRole('combobox');
      expect(select).toBeDisabled();
    });
  });
});
