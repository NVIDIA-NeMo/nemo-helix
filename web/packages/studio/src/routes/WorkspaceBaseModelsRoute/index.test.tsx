/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { suppressConsoleError } from '@nemo/testing/utils/suppress-console';
import { ROUTES } from '@studio/constants/routes';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { MemoryRouter, Route, Routes } from 'react-router';

const mockNavigate = vi.fn();
const mockEnvironment = vi.hoisted(() => ({
  customizerEnabled: false,
}));

vi.mock('@studio/constants/environment', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@studio/constants/environment')>();
  return {
    ...actual,
    get CUSTOMIZER_ENABLED() {
      return mockEnvironment.customizerEnabled;
    },
  };
});

vi.mock('react-router', async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...(actual as object),
    useNavigate: () => mockNavigate,
  };
});

const mockModel: ModelEntity = {
  id: 'model-id-1',
  name: 'my-model',
  workspace: 'ws1',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
  spec: {},
} as ModelEntity;

vi.mock('@nemo/common/src/api/entity-store/useBaseModels', () => ({
  useBaseModels: () => ({
    models: [mockModel],
    isLoading: false,
    isError: false,
    hasNextPage: false,
    fetchNextPage: vi.fn(),
    refetch: vi.fn(),
  }),
}));

vi.mock('@nemo/sdk/generated/platform/models', () => ({
  useModelsGetModel: () => ({
    data: undefined,
    isLoading: false,
  }),
}));

vi.mock('@nemo/common/src/hooks/useStudioDataViewState', () => ({
  useStudioDataViewState: () => ({
    debouncedSearchBar: '',
    debouncedColumnFilters: [],
    sorting: { state: [{ id: 'name', desc: false }], set: vi.fn() },
    searchBar: { state: '', set: vi.fn() },
    columnFiltering: { state: [], set: vi.fn() },
    pagination: { state: { pageIndex: 0, pageSize: 50 } },
    rowSelection: { state: {}, set: vi.fn() },
    resetFilters: vi.fn(),
    resetPagination: vi.fn(),
    apiFilter: {},
  }),
}));

vi.mock('@nemo/common/src/components/DataView/StudioDataView', () => ({
  StudioDataView: ({
    children,
    toolbarSlotEnd,
  }: {
    children: React.ReactNode;
    toolbarSlotEnd?: React.ReactNode;
  }) => (
    <div>
      {toolbarSlotEnd}
      {children}
    </div>
  ),
}));

// DataView.CustomContent renders its children function with mock row data.
// `mockRows` is a module-level mutable so individual tests can inject multiple
// rows to exercise per-row prop discrimination.
let mockRows: Array<{ original: ModelEntity }> = [{ original: mockModel }];
vi.mock('@nemo/common/src/components/DataView/internal', () => ({
  CustomContent: ({
    children,
  }: {
    children: (props: { rows: Array<{ original: ModelEntity }> }) => React.ReactNode;
  }) => <>{children({ rows: mockRows })}</>,
}));

vi.mock('@studio/components/VirtualizedCardGrid', () => ({
  VirtualizedCardGrid: <T,>({
    items,
    renderCard,
  }: {
    items: T[];
    renderCard: (item: T) => React.ReactNode;
  }) => (
    <>
      {items.map((item, i) => (
        <div key={i}>{renderCard(item)}</div>
      ))}
    </>
  ),
}));

vi.mock('@studio/components/BaseModelCard', () => ({
  BaseModelCard: ({
    model,
    isChatAvailable,
    onClick,
  }: {
    model: ModelEntity;
    isChatAvailable?: boolean;
    onClick?: () => void;
  }) => (
    <button
      type="button"
      data-testid="base-model-card"
      data-model-id={model.id ?? ''}
      data-is-chat-available={String(isChatAvailable ?? false)}
      onClick={onClick}
    >
      {model.name}
    </button>
  ),
}));

vi.mock('@studio/providers/breadcrumbs/useBreadcrumbs', () => ({
  useBreadcrumbs: () => {},
}));

// Mock ModelPanel to capture props via data attributes for assertion
vi.mock('@studio/components/sidePanels/ModelPanels/ModelPanel', () => ({
  ModelPanel: ({
    open,
    defaultTab,
    model,
    overviewProps,
    onOpenChange,
  }: {
    open?: boolean;
    defaultTab?: string;
    model?: ModelEntity;
    overviewProps?: {
      slotActions?: React.ReactNode;
    };
    onOpenChange?: (open: boolean) => void;
  }) => (
    <div
      data-testid="model-panel"
      data-open={String(open)}
      data-default-tab={defaultTab ?? ''}
      data-model-name={model?.name ?? ''}
    >
      {open && (
        <>
          {overviewProps?.slotActions}
          <button type="button" onClick={() => onOpenChange?.(false)}>
            Close panel
          </button>
        </>
      )}
    </div>
  ),
}));

const TestWrapper = ({
  initialEntry,
  children,
}: {
  initialEntry: string;
  children: React.ReactNode;
}) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return (
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialEntry]}>
        <Routes>
          <Route path={ROUTES.workspace.baseModels} element={children} />
          <Route path={ROUTES.workspace.baseModelsModel} element={children} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
};

describe('WorkspaceBaseModelsRoute deep linking', () => {
  // Import once to avoid per-test cold-start overhead that can exceed the test timeout
  // in the monorepo parallel test environment.
  let WorkspaceBaseModelsRoute: React.FC;
  beforeAll(async () => {
    ({ WorkspaceBaseModelsRoute } = await import('./index'));
  }, 30000);

  beforeEach(() => {
    vi.clearAllMocks();
    mockEnvironment.customizerEnabled = false;
    suppressConsoleError('was not wrapped in act');
  });

  it('opens model panel with model from path and passes defaultTab from tab query param', async () => {
    render(
      <TestWrapper initialEntry="/workspaces/ws1/base-models/my-model?tab=chat-playground">
        <WorkspaceBaseModelsRoute />
      </TestWrapper>
    );

    const panel = screen.getByTestId('model-panel');

    await waitFor(() => {
      expect(panel).toHaveAttribute('data-open', 'true');
    });
    expect(panel).toHaveAttribute('data-model-name', 'my-model');
    expect(panel).toHaveAttribute('data-default-tab', 'chat-playground');
  });

  it('opens model panel with model-details tab when tab param is model-details', async () => {
    render(
      <TestWrapper initialEntry="/workspaces/ws1/base-models/my-model?tab=model-details">
        <WorkspaceBaseModelsRoute />
      </TestWrapper>
    );
    const panel = screen.getByTestId('model-panel');

    await waitFor(() => {
      expect(panel).toHaveAttribute('data-open', 'true');
    });
    expect(panel).toHaveAttribute('data-model-name', 'my-model');
    expect(panel).toHaveAttribute('data-default-tab', 'model-details');
  });

  it('opens model panel with model-details tab when no tab query param (default)', async () => {
    render(
      <TestWrapper initialEntry="/workspaces/ws1/base-models/my-model">
        <WorkspaceBaseModelsRoute />
      </TestWrapper>
    );
    const panel = screen.getByTestId('model-panel');

    await waitFor(() => {
      expect(panel).toHaveAttribute('data-open', 'true');
    });
    expect(panel).toHaveAttribute('data-model-name', 'my-model');
    expect(panel).toHaveAttribute('data-default-tab', 'model-details');
  });

  it('does not render the dead evaluate action in the base model drawer', async () => {
    render(
      <TestWrapper initialEntry="/workspaces/ws1/base-models/my-model">
        <WorkspaceBaseModelsRoute />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByTestId('model-panel')).toHaveAttribute('data-open', 'true');
    });
    expect(screen.queryByRole('button', { name: 'Evaluate this Model' })).not.toBeInTheDocument();
  });

  it('does not open model panel when path has no model segment (list view)', async () => {
    render(
      <TestWrapper initialEntry="/workspaces/ws1/base-models">
        <WorkspaceBaseModelsRoute />
      </TestWrapper>
    );
    const panel = screen.getByTestId('model-panel');

    await waitFor(() => {
      expect(panel).toHaveAttribute('data-open', 'false');
    });
    expect(panel).toHaveAttribute('data-model-name', '');
  });

  it('preserves list query params when opening a model panel from the list', async () => {
    const queryParams = new URLSearchParams({
      s: 'llama',
      filters: JSON.stringify([{ id: 'customizable', value: { fine_tunable: true } }]),
      sort: '-created_at',
      page: '2',
    }).toString();

    render(
      <TestWrapper initialEntry={`/workspaces/ws1/base-models?${queryParams}`}>
        <WorkspaceBaseModelsRoute />
      </TestWrapper>
    );

    fireEvent.click(await screen.findByTestId('base-model-card'));

    expect(mockNavigate).toHaveBeenCalledWith(
      `/workspaces/ws1/base-models/my-model?${queryParams}`,
      { replace: true }
    );
  });

  it('preserves list query params and removes tab when closing a model panel', async () => {
    const listQueryParams = new URLSearchParams({
      s: 'llama',
      filters: JSON.stringify([{ id: 'customizable', value: { fine_tunable: true } }]),
      sort: '-created_at',
    }).toString();

    render(
      <TestWrapper
        initialEntry={`/workspaces/ws1/base-models/my-model?${listQueryParams}&tab=chat-playground`}
      >
        <WorkspaceBaseModelsRoute />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByTestId('model-panel')).toHaveAttribute('data-open', 'true');
    });
    fireEvent.click(screen.getByRole('button', { name: 'Close panel' }));

    expect(mockNavigate).toHaveBeenCalledWith(`/workspaces/ws1/base-models?${listQueryParams}`, {
      replace: true,
    });
  });
});

describe('WorkspaceBaseModelsRoute customizable filter gating', () => {
  let WorkspaceBaseModelsRoute: React.FC;
  beforeAll(async () => {
    ({ WorkspaceBaseModelsRoute } = await import('./index'));
  }, 30000);

  beforeEach(() => {
    vi.clearAllMocks();
    mockEnvironment.customizerEnabled = false;
    suppressConsoleError('was not wrapped in act');
  });

  it('hides the Fine-tunable checkbox when customizer is disabled', () => {
    render(
      <TestWrapper initialEntry="/workspaces/ws1/base-models">
        <WorkspaceBaseModelsRoute />
      </TestWrapper>
    );

    expect(screen.queryByRole('checkbox', { name: 'Fine-tunable' })).not.toBeInTheDocument();
  });

  it('shows the Fine-tunable checkbox when customizer is enabled', async () => {
    mockEnvironment.customizerEnabled = true;

    render(
      <TestWrapper initialEntry="/workspaces/ws1/base-models">
        <WorkspaceBaseModelsRoute />
      </TestWrapper>
    );

    expect(await screen.findByRole('checkbox', { name: 'Fine-tunable' })).toBeInTheDocument();
  });
});

describe('WorkspaceBaseModelsRoute card prop wiring', () => {
  let WorkspaceBaseModelsRoute: React.FC;
  beforeAll(async () => {
    ({ WorkspaceBaseModelsRoute } = await import('./index'));
  }, 30000);

  beforeEach(() => {
    vi.clearAllMocks();
    mockEnvironment.customizerEnabled = false;
    mockRows = [{ original: mockModel }];
    suppressConsoleError('was not wrapped in act');
  });

  it('forwards isChatAvailable derived from getModelEntityChatStatus per row', async () => {
    // `mockModel` has no `api_endpoint`, no `base_model`, and a `created_at` of
    // 2024-01-01 — well outside the 5-minute creation grace window — so
    // `getModelEntityChatStatus` returns 'enabled' via the optimistic standalone path.
    const chatModel = { ...mockModel, id: 'chat-id', name: 'chat-model' } as ModelEntity;
    // Inside the grace period → 'pending' (not 'enabled') so isChatAvailable=false.
    const recentModel = {
      ...mockModel,
      id: 'recent-id',
      name: 'recent-model',
      created_at: new Date().toISOString(),
    } as ModelEntity;
    mockRows = [{ original: chatModel }, { original: recentModel }];

    render(
      <TestWrapper initialEntry="/workspaces/ws1/base-models">
        <WorkspaceBaseModelsRoute />
      </TestWrapper>
    );

    const cards = await screen.findAllByTestId('base-model-card');
    const byId = new Map(cards.map((c) => [c.getAttribute('data-model-id'), c]));
    expect(byId.get('chat-id')).toHaveAttribute('data-is-chat-available', 'true');
    expect(byId.get('recent-id')).toHaveAttribute('data-is-chat-available', 'false');
  });
});
