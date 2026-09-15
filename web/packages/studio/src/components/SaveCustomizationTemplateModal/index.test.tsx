// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SaveCustomizationTemplateModal } from '@studio/components/SaveCustomizationTemplateModal';
import {
  getMockCustomizationJobTemplate,
  resetCustomizationJobTemplateMocks,
} from '@studio/mocks/handlers/customizationJobTemplates';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { FORM_DEFAULTS, type CustomizationFormFields } from '@studio/util/forms/customization';
import userEvent from '@testing-library/user-event';

const FIELDS: CustomizationFormFields = {
  ...FORM_DEFAULTS,
  automodel: {
    ...FORM_DEFAULTS.automodel,
    model: 'meta/llama-3.1-8b-instruct',
    dataset: { training: 'default/support-tickets' },
  },
};

beforeEach(() => {
  resetCustomizationJobTemplateMocks();
});

const renderModal = (onClose: () => void = vi.fn()) => {
  renderRoute(
    <SaveCustomizationTemplateModal open workspace="default" fields={FIELDS} onClose={onClose} />
  );
};

describe('SaveCustomizationTemplateModal', () => {
  it('persists the current form state, dataset included, under the given name', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    renderModal(onClose);

    await user.type(await screen.findByRole('textbox', { name: /template name/i }), 'my-recipe');
    await user.click(screen.getByRole('button', { name: /save template/i }));

    await waitFor(() => {
      expect(getMockCustomizationJobTemplate('my-recipe')).toBeDefined();
    });

    const saved = getMockCustomizationJobTemplate('my-recipe');
    expect(saved?.data.backend).toBe('automodel');
    expect(saved?.data.fields.automodel.dataset).toEqual({ training: 'default/support-tickets' });
    await waitFor(() => expect(onClose).toHaveBeenCalled());
  });

  it('requires a name before saving', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole('button', { name: /save template/i }));

    expect(await screen.findByText(/name is required/i)).toBeInTheDocument();
  });

  it('rejects a name with characters the entity-store will not accept', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.type(await screen.findByRole('textbox', { name: /template name/i }), 'not valid!');
    await user.click(screen.getByRole('button', { name: /save template/i }));

    expect(await screen.findByText(/must only contain alphanumeric/i)).toBeInTheDocument();
  });
});
