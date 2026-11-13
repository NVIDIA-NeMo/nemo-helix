// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import { SecretsDataView } from '@studio/components/dataViews/SecretsDataView';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { XL_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { menuFor } from '@studio/tests/util/menuFor';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { MemoryRouter } from 'react-router';

const workspace = 'default';

const renderComponent = (props?: { onCreate?: () => void }) => {
  return render(
    <MemoryRouter>
      <TestProviders>
        <SecretsDataView workspace={workspace} {...props} />
      </TestProviders>
    </MemoryRouter>
  );
};

describe('SecretsDataView', () => {
  it('shows Edit and Delete, with Delete styled as danger, and opens the delete modal', async () => {
    const user = userEvent.setup();
    renderComponent();

    await screen.findByText('openai-api-key', undefined, { timeout: XL_SELECTOR_TIMEOUT });
    const trigger = screen.getAllByRole('button', { name: 'Open quick actions menu' })[0];
    await user.click(trigger);

    const menu = menuFor(trigger);
    const items = within(menu).getAllByTestId('quick-actions-menu-item');
    expect(items.map((item) => item.textContent)).toEqual(['Edit', 'Delete']);
    expect(items.at(-1)).toHaveClass('nv-menu-item--danger');

    await user.click(within(menu).getByRole('menuitem', { name: 'Delete' }));
    expect(await screen.findByRole('dialog')).toHaveTextContent(/Delete: /);
  });

  describe('Data display', () => {
    it('displays secrets from the API', async () => {
      renderComponent();

      expect(
        await screen.findByText('openai-api-key', undefined, { timeout: XL_SELECTOR_TIMEOUT })
      ).toBeInTheDocument();

      expect(screen.getByText('anthropic-api-key')).toBeInTheDocument();
      expect(screen.getByText('huggingface-token')).toBeInTheDocument();
    });

    it('displays secret descriptions', async () => {
      renderComponent();

      expect(
        await screen.findByText('openai-api-key', undefined, { timeout: XL_SELECTOR_TIMEOUT })
      ).toBeInTheDocument();

      expect(screen.getByText('OpenAI API key for GPT-4 integration')).toBeInTheDocument();
    });

    it('displays search placeholder', async () => {
      renderComponent();

      expect(
        await screen.findByPlaceholderText('Search Secrets...', undefined, {
          timeout: XL_SELECTOR_TIMEOUT,
        })
      ).toBeInTheDocument();
    });
  });

  describe('Empty state', () => {
    it('displays empty state when no secrets exist', async () => {
      server.use(
        http.get(`${PLATFORM_BASE_URL}/apis/secrets/v2/workspaces/:workspace/secrets`, () =>
          HttpResponse.json({
            data: [],
            pagination: {
              page: 1,
              page_size: 25,
              current_page_size: 0,
              total_pages: 0,
              total_results: 0,
            },
          })
        )
      );

      renderComponent({ onCreate: () => {} });

      expect(
        await screen.findByText(ENTITY_EMPTY_STATES.secrets.heading, undefined, {
          timeout: XL_SELECTOR_TIMEOUT,
        })
      ).toBeInTheDocument();

      expect(screen.getByText(ENTITY_EMPTY_STATES.secrets.subheading)).toBeInTheDocument();
      expect(
        screen.getByRole('button', { name: ENTITY_EMPTY_STATES.secrets.createAction?.label })
      ).toBeInTheDocument();
    });

    it('invokes onCreate when the create button is clicked', async () => {
      const user = userEvent.setup();
      const onCreate = vi.fn();

      server.use(
        http.get(`${PLATFORM_BASE_URL}/apis/secrets/v2/workspaces/:workspace/secrets`, () =>
          HttpResponse.json({
            data: [],
            pagination: {
              page: 1,
              page_size: 25,
              current_page_size: 0,
              total_pages: 0,
              total_results: 0,
            },
          })
        )
      );

      renderComponent({ onCreate });

      const createButton = await screen.findByRole(
        'button',
        { name: ENTITY_EMPTY_STATES.secrets.createAction?.label },
        { timeout: XL_SELECTOR_TIMEOUT }
      );
      await user.click(createButton);

      expect(onCreate).toHaveBeenCalledTimes(1);
    });
  });

  describe('Search', () => {
    it('filters secrets by name when searching', async () => {
      const user = userEvent.setup();
      renderComponent();

      expect(
        await screen.findByText('openai-api-key', undefined, { timeout: XL_SELECTOR_TIMEOUT })
      ).toBeInTheDocument();

      const searchInput = screen.getByPlaceholderText('Search Secrets...');
      await user.type(searchInput, 'openai');

      await waitFor(() => {
        expect(screen.queryByText('anthropic-api-key')).not.toBeInTheDocument();
        expect(screen.queryByText('huggingface-token')).not.toBeInTheDocument();
      });
    });

    it('shows no-results empty state when search has no matches', async () => {
      const user = userEvent.setup();
      renderComponent();

      expect(
        await screen.findByText('openai-api-key', undefined, { timeout: XL_SELECTOR_TIMEOUT })
      ).toBeInTheDocument();

      const searchInput = screen.getByPlaceholderText('Search Secrets...');
      await user.type(searchInput, 'nonexistent-secret-name');

      expect(
        await screen.findByText('No results found', undefined, { timeout: XL_SELECTOR_TIMEOUT })
      ).toBeInTheDocument();

      expect(
        screen.getByText('No items match your current search or filters.')
      ).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Clear Filters' })).toBeInTheDocument();
    });
  });
});
