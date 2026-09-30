// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { CustomizeModelButton } from '@studio/components/dataViews/CustomModelsDataView/CustomizeModelButton';
import { ROUTES } from '@studio/constants/routes';
import { useModelCustomizationEligibility } from '@studio/hooks/useModelCustomizationEligibility';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider } from 'react-router';

vi.mock('@studio/hooks/useModelCustomizationEligibility', () => ({
  useModelCustomizationEligibility: vi.fn(),
}));

const mockedUseEligibility = vi.mocked(useModelCustomizationEligibility);

const testModel = { id: 'model-1', name: 'my-model', workspace: 'ws' } as ModelEntity;

const setEligibility = (overrides: { canFineTune?: boolean; isLoading?: boolean } = {}) => {
  const canFineTune = overrides.canFineTune ?? false;
  mockedUseEligibility.mockReturnValue({
    canFineTune,
    canCustomize: canFineTune,
    isLoading: overrides.isLoading ?? false,
  });
};

const renderRoute = (props: { model?: ModelEntity } = {}) => {
  const router = createMemoryRouter(
    [
      {
        path: ROUTES.workspace.customizationJobList,
        element: (
          <>
            <CustomizeModelButton workspace={workspace1.workspace} model={props.model} />
            <LocationDisplay />
          </>
        ),
      },
      { path: ROUTES.workspace.newCustomizationJob, element: <LocationDisplay /> },
    ],
    { initialEntries: [ROUTES.workspace.customizationJobList] }
  );
  return render(
    <TestProviders>
      <RouterProvider router={router} />
    </TestProviders>
  );
};

describe('CustomizeModelButton', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setEligibility({ canFineTune: true });
  });

  describe('workspace-level (no model)', () => {
    it('renders "Customize a Model"', () => {
      renderRoute();
      expect(screen.getByRole('button', { name: 'Customize a Model' })).toBeInTheDocument();
    });

    it('navigates straight to the fine-tuning form on click', async () => {
      const user = userEvent.setup();
      renderRoute();
      await user.click(screen.getByRole('button', { name: 'Customize a Model' }));
      expect(await screen.findByTestId(LOCATION_DISPLAY_TEST_ID)).toHaveTextContent(
        `/workspaces/${workspace1.workspace}/fine-tune/new`
      );
    });

    it('stays enabled regardless of eligibility', () => {
      setEligibility({ canFineTune: false });
      renderRoute();
      expect(screen.getByRole('button', { name: 'Customize a Model' })).not.toBeDisabled();
    });
  });

  describe('per-model', () => {
    it('renders "Customize this Model" when a model is provided', () => {
      renderRoute({ model: testModel });
      expect(screen.getByRole('button', { name: /Customize this Model/ })).toBeInTheDocument();
    });

    it('disables the button while eligibility is loading', () => {
      setEligibility({ isLoading: true });
      renderRoute({ model: testModel });
      expect(screen.getByRole('button', { name: /Customize this Model/ })).toBeDisabled();
    });

    it('shows a spinner while eligibility is loading', () => {
      setEligibility({ isLoading: true });
      renderRoute({ model: testModel });
      expect(screen.getByRole('status')).toBeInTheDocument();
    });

    it('disables the button when the model cannot be fine-tuned', () => {
      setEligibility({ canFineTune: false });
      renderRoute({ model: testModel });
      expect(screen.getByRole('button', { name: /Customize this Model/ })).toBeDisabled();
    });

    it('navigates to the fine-tuning form with the model preselected', async () => {
      const user = userEvent.setup();
      renderRoute({ model: testModel });
      await user.click(screen.getByRole('button', { name: /Customize this Model/ }));
      expect(await screen.findByTestId(LOCATION_DISPLAY_TEST_ID)).toHaveTextContent(
        `/workspaces/${workspace1.workspace}/fine-tune/new`
      );
    });
  });
});
