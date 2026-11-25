// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelEntity } from '@nemo/common/src/api/models/useModelEntity';
import { useModelSearch } from '@nemo/common/src/api/models/useModelSearch';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { ModelCompareRoute } from '@studio/routes/ModelCompareRoute';
import type { SharedModelEntry } from '@studio/routes/ModelCompareRoute/types';
import { fireEvent, renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { useNavigate } from 'react-router';

vi.mock('@nemo/common/src/api/models/useModelSearch', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('@nemo/common/src/api/models/useModelSearch')>();
  return { ...actual, useModelSearch: vi.fn() };
});

vi.mock('@nemo/common/src/api/models/useModelEntity', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('@nemo/common/src/api/models/useModelEntity')>();
  return { ...actual, useModelEntity: vi.fn() };
});

vi.mock('@studio/hooks/useWorkspaceFromPath', () => ({
  useWorkspaceFromPath: () => 'ws1',
}));

vi.mock('@studio/components/ModelCompareChat', () => ({
  ModelCompareChat: ({ models }: { models: SharedModelEntry[] }) => (
    <div data-testid="panels">{models.map((entry) => entry.modelURN ?? 'empty').join(',')}</div>
  ),
}));

vi.mock('@studio/components/ModelComparePrompts', () => ({
  ModelComparePrompts: () => null,
}));

const mockUseModelSearch = vi.mocked(useModelSearch);
const mockUseModelEntity = vi.mocked(useModelEntity);

const entity = (workspace: string, name: string, providers?: string[]): ModelEntity =>
  ({ id: `${workspace}/${name}`, workspace, name, model_providers: providers }) as ModelEntity;

const ENTITIES: Record<string, ModelEntity> = {
  'ws1/alpha': entity('ws1', 'alpha', ['ws1/build']),
  'ws1/beta': entity('ws1', 'beta', ['ws1/build']),
  'ws1/unserved': entity('ws1', 'unserved'),
  'other/foreign': entity('other', 'foreign', ['other/build']),
};

const searchResult = (overrides: Partial<ReturnType<typeof useModelSearch>> = {}) =>
  ({
    models: [ENTITIES['ws1/alpha']],
    groups: [{ workspace: 'ws1', models: [ENTITIES['ws1/alpha']] }],
    search: '',
    error: null,
    loading: false,
    onSearchChange: vi.fn(),
    onLoadMore: vi.fn(),
    hasMore: false,
    isLoadingMore: false,
    ...overrides,
  }) as ReturnType<typeof useModelSearch>;

beforeEach(() => {
  mockUseModelSearch.mockReset();
  mockUseModelEntity.mockReset();
  mockUseModelSearch.mockReturnValue(searchResult());
  mockUseModelEntity.mockImplementation((urn, options) =>
    options?.enabled === false || !urn ? undefined : ENTITIES[urn]
  );
});

describe('ModelCompareRoute availability', () => {
  it('shows the no-models state only for an exhausted, successful search', async () => {
    mockUseModelSearch.mockReturnValue(searchResult({ models: [], groups: [] }));

    renderRoute(<ModelCompareRoute />, { history: '/compare' });

    expect(await screen.findByText('No Models Available')).toBeInTheDocument();
  });

  it('shows an error state instead of the no-models state when the search fails', async () => {
    mockUseModelSearch.mockReturnValue(
      searchResult({ models: [], groups: [], error: new Error('gateway unreachable') })
    );

    renderRoute(<ModelCompareRoute />, { history: '/compare' });

    expect(await screen.findByText('gateway unreachable')).toBeInTheDocument();
    expect(screen.queryByText('No Models Available')).not.toBeInTheDocument();
  });
});

describe('ModelCompareRoute ?model= preselection', () => {
  it('preselects a model this workspace serves', async () => {
    renderRoute(<ModelCompareRoute />, { history: '/compare?model=alpha' });

    await waitFor(() => expect(screen.getByTestId('panels')).toHaveTextContent('ws1/alpha,empty'));
  });

  it('ignores a model from another workspace', async () => {
    renderRoute(<ModelCompareRoute />, { history: '/compare?model=other/foreign' });

    await waitFor(() => expect(screen.getByTestId('panels')).toHaveTextContent('empty,empty'));
  });

  it('ignores a model with no provider', async () => {
    renderRoute(<ModelCompareRoute />, { history: '/compare?model=unserved' });

    await waitFor(() => expect(screen.getByTestId('panels')).toHaveTextContent('empty,empty'));
  });

  it('applies ?model= added after the first render', async () => {
    const Harness = () => {
      const navigate = useNavigate();
      return (
        <>
          <button onClick={() => navigate('/compare?model=beta')}>select beta</button>
          <ModelCompareRoute />
        </>
      );
    };

    renderRoute(<Harness />, { history: '/compare' });

    await waitFor(() => expect(screen.getByTestId('panels')).toHaveTextContent('empty,empty'));

    fireEvent.click(screen.getByRole('button', { name: 'select beta' }));

    await waitFor(() => expect(screen.getByTestId('panels')).toHaveTextContent('ws1/beta,empty'));
  });
});
