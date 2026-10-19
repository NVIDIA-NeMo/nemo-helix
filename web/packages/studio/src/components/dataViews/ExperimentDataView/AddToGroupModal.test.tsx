// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MockToastProvider } from '@nemo/common/src/tests/MockToastProvider';
import type { ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { AddToGroupModal } from '@studio/components/dataViews/ExperimentDataView/AddToGroupModal';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Hoisted mocks for the SDK hooks the modal calls.
const { mockMutateAsync, mockCreateMutateAsync, mockUseListExperiments } = vi.hoisted(() => ({
  mockMutateAsync: vi.fn<(...args: unknown[]) => Promise<unknown>>(),
  mockCreateMutateAsync: vi.fn<(...args: unknown[]) => Promise<unknown>>(),
  mockUseListExperiments: vi.fn<() => { data: unknown; isLoading: boolean }>(),
}));

vi.mock('@nemo/sdk/generated/platform/experiments', async () => {
  const actual = await vi.importActual<typeof import('@nemo/sdk/generated/platform/experiments')>(
    '@nemo/sdk/generated/platform/experiments'
  );
  return {
    ...actual,
    useListExperiments: () => mockUseListExperiments(),
    useCreateExperiment: () => ({ mutateAsync: mockCreateMutateAsync, isPending: false }),
  };
});

vi.mock('@nemo/sdk/generated/platform/evaluations', async () => {
  const actual = await vi.importActual<typeof import('@nemo/sdk/generated/platform/evaluations')>(
    '@nemo/sdk/generated/platform/evaluations'
  );
  return {
    ...actual,
    usePatchEvaluation: () => ({ mutateAsync: mockMutateAsync, isPending: false }),
  };
});

const makeGroup = (id: string, name: string): ExperimentResponse => ({
  id,
  name,
  workspace: 'default',
  default_sort: '-created_at',
});

const GROUPS: ExperimentResponse[] = [
  makeGroup('g1', 'Alpha benchmarks'),
  makeGroup('g2', 'Beta benchmarks'),
  makeGroup('g3', 'Gamma benchmarks'),
];

const makeEvaluation = (name: string, experimentIds: string[]): EvaluationRow =>
  ({ id: name, name, experiment_ids: experimentIds }) as unknown as EvaluationRow;

// eval-1 is only in g1; eval-2 is in g1 and g2. So g1 (all members) is excluded, but g2 (only some
// members) is still offered, as is g3 (no members).
const EVALUATIONS = [makeEvaluation('eval-1', ['g1']), makeEvaluation('eval-2', ['g1', 'g2'])];

function makeWrapper() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: React.ReactNode }) => (
    <MockToastProvider>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </MockToastProvider>
  );
}

function renderModal(evaluations: EvaluationRow[] = EVALUATIONS) {
  const Wrapper = makeWrapper();
  const onSuccess = vi.fn();
  const utils = render(
    <Wrapper>
      <AddToGroupModal
        open
        onClose={vi.fn()}
        onSuccess={onSuccess}
        workspace="default"
        evaluations={evaluations}
      />
    </Wrapper>
  );
  return { ...utils, onSuccess };
}

describe('AddToGroupModal', () => {
  beforeEach(() => {
    mockMutateAsync.mockReset();
    mockMutateAsync.mockResolvedValue(undefined);
    mockCreateMutateAsync.mockReset();
    mockCreateMutateAsync.mockResolvedValue(makeGroup('g-new', 'Regression suite'));
    mockUseListExperiments.mockReset();
    mockUseListExperiments.mockReturnValue({
      data: { data: GROUPS },
      isLoading: false,
    });
  });

  it('offers "Create new experiment" plus only groups not every selected evaluation already belongs to', async () => {
    const user = userEvent.setup();
    renderModal();
    await user.click(screen.getByRole('combobox', { name: /experiment/i }));
    expect(
      await screen.findByRole('option', { name: /create new experiment/i })
    ).toBeInTheDocument();
    // g1: both evals are members -> excluded. g2: only eval-2 is a member -> still offered. g3: none.
    expect(screen.getByRole('option', { name: 'Beta benchmarks' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Gamma benchmarks' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Alpha benchmarks' })).not.toBeInTheDocument();
  });

  it('adds every selected evaluation to the chosen group and clears the selection', async () => {
    const user = userEvent.setup();
    const { onSuccess } = renderModal();

    await user.click(screen.getByRole('combobox', { name: /experiment/i }));
    await user.click(await screen.findByRole('option', { name: 'Gamma benchmarks' }));
    await user.click(screen.getByRole('button', { name: 'Add' }));

    expect(mockCreateMutateAsync).not.toHaveBeenCalled();
    expect(mockMutateAsync).toHaveBeenCalledTimes(2);
    expect(mockMutateAsync).toHaveBeenCalledWith({
      workspace: 'default',
      name: 'eval-1',
      data: { experiment_ids: ['g1', 'g3'] },
    });
    expect(mockMutateAsync).toHaveBeenCalledWith({
      workspace: 'default',
      name: 'eval-2',
      data: { experiment_ids: ['g1', 'g2', 'g3'] },
    });
    await waitFor(() => expect(onSuccess).toHaveBeenCalledTimes(1));
  });

  it('creates a new group then adds every selected evaluation to it', async () => {
    const user = userEvent.setup();
    const { onSuccess } = renderModal();

    await user.click(screen.getByRole('combobox', { name: /experiment/i }));
    await user.click(await screen.findByRole('option', { name: /create new experiment/i }));
    await user.type(screen.getByRole('textbox', { name: /^name$/i }), 'regression-suite');
    await user.click(screen.getByRole('button', { name: 'Create & add' }));

    expect(mockCreateMutateAsync).toHaveBeenCalledWith({
      workspace: 'default',
      data: expect.objectContaining({ name: 'regression-suite' }),
    });
    // Added to the id returned by the create call.
    expect(mockMutateAsync).toHaveBeenCalledTimes(2);
    expect(mockMutateAsync).toHaveBeenCalledWith({
      workspace: 'default',
      name: 'eval-1',
      data: { experiment_ids: ['g1', 'g-new'] },
    });
    expect(mockMutateAsync).toHaveBeenCalledWith({
      workspace: 'default',
      name: 'eval-2',
      data: { experiment_ids: ['g1', 'g2', 'g-new'] },
    });
    await waitFor(() => expect(onSuccess).toHaveBeenCalledTimes(1));
  });

  it('still offers "Create new experiment" when every group already contains all selected evaluations', async () => {
    const user = userEvent.setup();
    mockUseListExperiments.mockReturnValue({
      data: { data: [makeGroup('g1', 'Alpha benchmarks')] },
      isLoading: false,
    });
    renderModal([makeEvaluation('eval-1', ['g1']), makeEvaluation('eval-2', ['g1'])]);

    const trigger = screen.getByRole('combobox', { name: /experiment/i });
    expect(trigger).toBeEnabled();
    await user.click(trigger);
    expect(
      await screen.findByRole('option', { name: /create new experiment/i })
    ).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Alpha benchmarks' })).not.toBeInTheDocument();
  });
});
