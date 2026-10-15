// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CreateCustomizationStartProps } from '@studio/components/CreateCustomizationStart/types';
import { ROUTES } from '@studio/constants/routes';
import { NewCustomizationFormRoute } from '@studio/routes/NewCustomizationFormRoute';
import { NewCustomizationRoute } from '@studio/routes/NewCustomizationRoute';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { FORM_DEFAULTS, type CustomizationFormFields } from '@studio/util/forms/customization';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider } from 'react-router';

// Not the form's default backend, so the form showing it proves the values arrived.
const VALUES: CustomizationFormFields = {
  ...FORM_DEFAULTS,
  backend: 'unsloth',
  outputName: 'handed-over',
  unsloth: {
    ...FORM_DEFAULTS.unsloth,
    model: { ...FORM_DEFAULTS.unsloth.model, name: 'default/base-model' },
    dataset: { ...FORM_DEFAULTS.unsloth.dataset, path: 'default/data' },
  },
};

// Stands in for the picker, so each way in can hand over a known selection.
vi.mock('@studio/components/CreateCustomizationStart', () => ({
  CreateCustomizationStart: ({ onContinue }: CreateCustomizationStartProps) => (
    <>
      <button type="button" onClick={() => onContinue({ optionId: 'ai', initialValues: VALUES })}>
        AI
      </button>
      <button
        type="button"
        onClick={() => onContinue({ optionId: 'template', initialValues: VALUES })}
      >
        Template
      </button>
      <button
        type="button"
        onClick={() =>
          onContinue({
            optionId: 'template',
            initialValues: VALUES,
            savedTemplateName: 'my-sft-recipe',
          })
        }
      >
        Saved
      </button>
    </>
  ),
}));

const mountRoute = () => {
  const router = createMemoryRouter(
    [
      { path: ROUTES.workspace.newCustomizationJob, element: <NewCustomizationRoute /> },
      { path: ROUTES.workspace.newCustomizationJobScratch, element: <NewCustomizationFormRoute /> },
    ],
    { initialEntries: ['/workspaces/default/fine-tune/new'] }
  );
  render(
    <TestProviders>
      <RouterProvider router={router} />
    </TestProviders>
  );
  return router;
};

describe('NewCustomizationRoute hand-off', () => {
  it.each(['AI', 'Template'])('opens the form filled in from the %s values', async (way) => {
    mountRoute();
    await userEvent.click(screen.getByRole('button', { name: way }));

    expect(await screen.findByRole('radio', { name: /Unsloth/ })).toBeChecked();
    expect(screen.getByDisplayValue('handed-over')).toBeInTheDocument();
  });

  it('names the saved template, so the form can offer to delete it', async () => {
    const router = mountRoute();
    await userEvent.click(screen.getByRole('button', { name: 'Saved' }));

    await screen.findByRole('radio', { name: /Unsloth/ });
    expect(router.state.location.state).toMatchObject({ savedTemplateName: 'my-sft-recipe' });
  });
});
