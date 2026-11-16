// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { FilesetActionMenu } from '@studio/components/FilesetActionMenu';
import { render } from '@studio/tests/util/render';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.mock('@studio/components/DatasetCreateModal', () => ({
  DatasetCreateModal: vi.fn(({ open }) => {
    if (!open) {
      return null;
    }
    return <div data-testid="dataset-create-modal" />;
  }),
}));

describe('FilesetActionMenu', () => {
  const fileset: FilesetOutput = {
    id: '123',
    name: 'test-fileset',
    workspace: 'test-workspace',
    description: 'Test fileset description',
    purpose: 'dataset',
    storage: { type: 'local', path: '/data/test-fileset' },
    metadata: {},
    custom_fields: {},
    project: 'test-workspace',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  };

  const openMenu = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(screen.getByRole('button', { name: /open fileset actions menu/i }));
  };

  it('shows Edit and Delete, with Delete styled as danger', async () => {
    const user = userEvent.setup();
    render(<FilesetActionMenu fileset={fileset} />);

    await openMenu(user);

    const items = screen.getAllByTestId('quick-actions-menu-item');
    expect(items.map((item) => item.textContent)).toEqual(['Edit', 'Delete']);
    expect(items.at(-1)).toHaveClass('nv-menu-item--danger');
  });

  it('opens the edit modal when Edit is clicked', async () => {
    const user = userEvent.setup();
    render(<FilesetActionMenu fileset={fileset} />);

    await openMenu(user);
    await user.click(screen.getByText('Edit'));

    expect(screen.getByTestId('dataset-create-modal')).toBeInTheDocument();
  });

  it('opens the delete confirmation modal when Delete is clicked', async () => {
    const user = userEvent.setup();
    render(<FilesetActionMenu fileset={fileset} />);

    await openMenu(user);
    await user.click(screen.getByText('Delete'));

    expect(
      await screen.findByRole('heading', { name: "Delete fileset 'test-fileset'?" })
    ).toBeInTheDocument();
  });
});
