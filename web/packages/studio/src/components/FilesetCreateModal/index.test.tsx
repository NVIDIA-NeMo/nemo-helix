// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MockToastProvider } from '@nemo/common/src/tests/MockToastProvider';
import { FilesetPurpose, type FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { FilesetCreateModal } from '@studio/components/FilesetCreateModal';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { mockUseNavigate, mockUseParams } from '@studio/tests/util/mockUseParams';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router';

// Hoisted mocks for hooks the modal calls.
const { mockMutate, mockUseRemoteRepoMetadata } = vi.hoisted(() => ({
  mockMutate: vi.fn<(args: unknown) => Promise<FilesetOutput>>(),
  mockUseRemoteRepoMetadata: vi.fn<
    (
      url: string | undefined,
      enabled: boolean
    ) => {
      data: { slug: string; description: string | null } | null | undefined;
      isFetching: boolean;
    }
  >(),
}));

vi.mock('@nemo/sdk/generated/platform/files', async () => {
  const actual = await vi.importActual<typeof import('@nemo/sdk/generated/platform/files')>(
    '@nemo/sdk/generated/platform/files'
  );
  return {
    ...actual,
    useFilesCreateFileset: () => ({ mutateAsync: mockMutate, isPending: false }),
  };
});

vi.mock('@studio/hooks/useRemoteRepoMetadata', () => ({
  useRemoteRepoMetadata: (url: string | undefined, enabled: boolean) =>
    mockUseRemoteRepoMetadata(url, enabled),
}));

// SecretSearchableSelect pulls in a heavier dep tree; replace with a stub
// since we only assert that the External fields appear, not their internals.
vi.mock('@studio/routes/SecretsListRoute/SecretSearchableSelect', () => ({
  SecretSearchableSelect: () => <div data-testid="secret-select-stub" />,
}));

vi.mock('@studio/routes/SecretsListRoute/CreateSecretModal', () => ({
  CreateSecretModal: () => null,
}));

const FILESET_RESPONSE: FilesetOutput = {
  id: 'fs-id',
  name: 'mod3',
  workspace: 'default',
  description: '',
  purpose: 'model',
  storage: { type: 'local', path: '/data' },
  metadata: {},
  custom_fields: {},
  project: 'default',
  created_at: '2026-05-29T00:00:00Z',
  updated_at: '2026-05-29T00:00:00Z',
};

function makeWrapper() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: React.ReactNode }) => (
    <MockToastProvider>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>{children}</BrowserRouter>
      </QueryClientProvider>
    </MockToastProvider>
  );
}

describe('FilesetCreateModal', () => {
  beforeEach(() => {
    mockUseParams({ [ROUTE_PARAMS.workspace]: 'default' });
    mockUseNavigate();
    mockMutate.mockReset();
    mockMutate.mockResolvedValue(FILESET_RESPONSE);
    mockUseRemoteRepoMetadata.mockReset();
    mockUseRemoteRepoMetadata.mockReturnValue({ data: undefined, isFetching: false });
  });

  describe('heading + submit copy by purpose', () => {
    it.each([
      [FilesetPurpose.dataset, 'Create Dataset'],
      [FilesetPurpose.model, 'Create Model Fileset'],
    ])('purpose=%s renders %s', (purpose, expected) => {
      const Wrapper = makeWrapper();
      render(
        <Wrapper>
          <FilesetCreateModal
            open
            onClose={vi.fn()}
            workspace="default"
            purpose={purpose as typeof FilesetPurpose.dataset | typeof FilesetPurpose.model}
          />
        </Wrapper>
      );
      // Heading + primary button both use the same copy.
      const matches = screen.getAllByText(expected);
      expect(matches.length).toBeGreaterThanOrEqual(2);
    });
  });

  describe('storage mode toggle', () => {
    it('shows only Name + Description in Local mode', () => {
      const Wrapper = makeWrapper();
      render(
        <Wrapper>
          <FilesetCreateModal
            open
            onClose={vi.fn()}
            workspace="default"
            purpose={FilesetPurpose.dataset}
          />
        </Wrapper>
      );
      expect(screen.getByRole('textbox', { name: /^name$/i })).toBeInTheDocument();
      expect(screen.getByRole('textbox', { name: /description/i })).toBeInTheDocument();
      expect(screen.queryByRole('textbox', { name: /^url$/i })).not.toBeInTheDocument();
      expect(screen.queryByTestId('secret-select-stub')).not.toBeInTheDocument();
    });

    it('shows URL + Secret above Name in External mode', async () => {
      const user = userEvent.setup();
      const Wrapper = makeWrapper();
      render(
        <Wrapper>
          <FilesetCreateModal
            open
            onClose={vi.fn()}
            workspace="default"
            purpose={FilesetPurpose.dataset}
          />
        </Wrapper>
      );
      await user.click(screen.getByRole('radio', { name: 'External' }));
      expect(screen.getByRole('textbox', { name: /^url$/i })).toBeInTheDocument();
      expect(screen.getByTestId('secret-select-stub')).toBeInTheDocument();
    });
  });

  describe('remote metadata auto-fill (External mode)', () => {
    it('fills the name from the slug when the field has not been edited', async () => {
      const user = userEvent.setup();
      mockUseRemoteRepoMetadata.mockReturnValue({
        data: { slug: 'Qwen3.6-35B-A3B-MTP-GGUF', description: null },
        isFetching: false,
      });
      const Wrapper = makeWrapper();
      render(
        <Wrapper>
          <FilesetCreateModal
            open
            onClose={vi.fn()}
            workspace="default"
            purpose={FilesetPurpose.dataset}
          />
        </Wrapper>
      );
      await user.click(screen.getByRole('radio', { name: 'External' }));
      const nameInput = screen.getByRole('textbox', { name: /^name$/i }) as HTMLInputElement;
      await waitFor(() => {
        // Sanitized: lowercase, valid chars
        expect(nameInput.value).toBe('qwen3.6-35b-a3b-mtp-gguf');
      });
    });

    it('does not overwrite a user-typed name', async () => {
      const user = userEvent.setup();
      // Start with no metadata; user types into Name; later metadata arrives.
      mockUseRemoteRepoMetadata.mockReturnValue({ data: undefined, isFetching: false });
      const Wrapper = makeWrapper();
      const { rerender } = render(
        <Wrapper>
          <FilesetCreateModal
            open
            onClose={vi.fn()}
            workspace="default"
            purpose={FilesetPurpose.dataset}
          />
        </Wrapper>
      );
      await user.click(screen.getByRole('radio', { name: 'External' }));
      const nameInput = screen.getByRole('textbox', { name: /^name$/i }) as HTMLInputElement;
      await user.type(nameInput, 'my-own-name');

      // Now simulate metadata arriving on a subsequent render.
      mockUseRemoteRepoMetadata.mockReturnValue({
        data: { slug: 'OtherSlug', description: null },
        isFetching: false,
      });
      rerender(
        <Wrapper>
          <FilesetCreateModal
            open
            onClose={vi.fn()}
            workspace="default"
            purpose={FilesetPurpose.dataset}
          />
        </Wrapper>
      );
      // User input survives.
      expect(nameInput.value).toBe('my-own-name');
    });

    it('disables Name + Description while fetch is in flight', async () => {
      const user = userEvent.setup();
      mockUseRemoteRepoMetadata.mockReturnValue({ data: undefined, isFetching: true });
      const Wrapper = makeWrapper();
      render(
        <Wrapper>
          <FilesetCreateModal
            open
            onClose={vi.fn()}
            workspace="default"
            purpose={FilesetPurpose.dataset}
          />
        </Wrapper>
      );
      await user.click(screen.getByRole('radio', { name: 'External' }));
      expect(screen.getByRole('textbox', { name: /^name$/i })).toBeDisabled();
      expect(screen.getByRole('textbox', { name: /description/i })).toBeDisabled();
    });
  });

  describe('navigation after create', () => {
    it('Dataset + Local navigates to fileset detail Files tab', async () => {
      const navigate = vi.fn();
      mockUseNavigate(navigate);
      mockMutate.mockResolvedValue({
        ...FILESET_RESPONSE,
        name: 'mydataset',
        purpose: 'dataset',
        storage: { type: 'local', path: '/x' },
      });
      const user = userEvent.setup();
      const Wrapper = makeWrapper();
      render(
        <Wrapper>
          <FilesetCreateModal
            open
            onClose={vi.fn()}
            workspace="default"
            purpose={FilesetPurpose.dataset}
          />
        </Wrapper>
      );
      await user.type(screen.getByRole('textbox', { name: /^name$/i }), 'mydataset');
      await user.click(screen.getByRole('button', { name: 'Create Dataset' }));
      await waitFor(() => {
        expect(navigate).toHaveBeenCalledWith(
          expect.stringContaining('/workspaces/default/filesets/mydataset/detail?tab=files')
        );
      });
    });

    it('Model + Local navigates to fileset detail Files tab', async () => {
      const navigate = vi.fn();
      mockUseNavigate(navigate);
      mockMutate.mockResolvedValue({
        ...FILESET_RESPONSE,
        name: 'mymodel',
        purpose: 'model',
      });
      const user = userEvent.setup();
      const Wrapper = makeWrapper();
      render(
        <Wrapper>
          <FilesetCreateModal
            open
            onClose={vi.fn()}
            workspace="default"
            purpose={FilesetPurpose.model}
          />
        </Wrapper>
      );
      await user.type(screen.getByRole('textbox', { name: /^name$/i }), 'mymodel');
      await user.click(screen.getByRole('button', { name: 'Create Model Fileset' }));
      await waitFor(() => {
        expect(navigate).toHaveBeenCalledWith(
          expect.stringContaining('/workspaces/default/filesets/mymodel/detail?tab=files')
        );
      });
    });
  });
});
