/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { CreateInferenceProviderSidePanel } from '@studio/routes/InferenceProvidersListRoute/CreateInferenceProviderSidePanel';
import { render } from '@studio/tests/util/render';
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { delay, http, HttpResponse } from 'msw';

const mockOnClose = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

async function openModelProviderSelect(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('combobox', { name: /model provider/i }));
}

describe('CreateInferenceProviderSidePanel', () => {
  const defaultProps = {
    workspace: 'test-workspace',
    open: true,
    onClose: mockOnClose,
  };

  describe('Rendering', () => {
    it('renders nothing when closed', () => {
      render(<CreateInferenceProviderSidePanel {...defaultProps} open={false} />);
      expect(screen.queryByTestId('nv-side-panel-content')).not.toBeInTheDocument();
    });

    it('renders dialog with title and intro when open', async () => {
      render(<CreateInferenceProviderSidePanel {...defaultProps} />);
      const panel = await screen.findByTestId('nv-side-panel-content');
      expect(panel).toBeInTheDocument();
      expect(within(panel).getByText('Add Inference Provider')).toBeInTheDocument();
      expect(
        within(panel).getByText('Select a provider to add to your workspace.')
      ).toBeInTheDocument();
      expect(within(panel).getByRole('combobox', { name: /model provider/i })).toBeInTheDocument();
      expect(within(panel).getByText(/API Key Secret/)).toBeInTheDocument();
      expect(within(panel).getByRole('textbox', { name: 'Name' })).toBeInTheDocument();
      expect(within(panel).getByRole('textbox', { name: 'Host URL' })).toBeInTheDocument();
    });

    it('renders submit and cancel actions', async () => {
      render(<CreateInferenceProviderSidePanel {...defaultProps} />);
      const panel = await screen.findByTestId('nv-side-panel-content');
      expect(panel).toBeInTheDocument();
      expect(within(panel).getByRole('button', { name: 'Add Provider' })).toBeInTheDocument();
      expect(within(panel).getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    });

    it('shows search field and grouped sections when Model Provider menu is open', async () => {
      const user = userEvent.setup();
      render(<CreateInferenceProviderSidePanel {...defaultProps} />);
      await screen.findByTestId('nv-side-panel-content');
      await openModelProviderSelect(user);
      expect(screen.getByPlaceholderText('Search by name...')).toBeInTheDocument();
      expect(screen.getByText('Pre-configured Providers')).toBeInTheDocument();
      expect(screen.getByText('Custom')).toBeInTheDocument();
    });

    it('when OpenAI compatible endpoint is selected, shows Name and Host URL fields', async () => {
      const user = userEvent.setup();
      render(<CreateInferenceProviderSidePanel {...defaultProps} />);
      const dialog = await screen.findByTestId('nv-side-panel-content');
      await openModelProviderSelect(user);
      await user.click(screen.getByRole('option', { name: /openai compatible endpoint/i }));
      expect(within(dialog).getByRole('textbox', { name: 'Name' })).toBeInTheDocument();
      expect(within(dialog).getByRole('textbox', { name: 'Host URL' })).toBeInTheDocument();
    });
  });

  describe('Preset selection', () => {
    it('selecting NVIDIA Build pre-fills form so Add Provider is enabled', async () => {
      const user = userEvent.setup();
      render(<CreateInferenceProviderSidePanel {...defaultProps} />);
      await screen.findByTestId('nv-side-panel-content');
      await openModelProviderSelect(user);
      await user.click(screen.getByRole('option', { name: /^nvidia build$/i }));
      expect(await screen.findByRole('button', { name: 'Add Provider' })).toBeEnabled();
    });

    it('selecting OpenAI compatible endpoint clears name and host URL after NVIDIA Build', async () => {
      const user = userEvent.setup();
      render(<CreateInferenceProviderSidePanel {...defaultProps} />);
      const dialog = await screen.findByTestId('nv-side-panel-content');
      await openModelProviderSelect(user);
      await user.click(screen.getByRole('option', { name: /^nvidia build$/i }));
      await openModelProviderSelect(user);
      await user.click(screen.getByRole('option', { name: /openai compatible endpoint/i }));
      const nameInput = within(dialog).getByRole('textbox', { name: 'Name' });
      const hostInput = within(dialog).getByRole('textbox', { name: 'Host URL' });
      expect(nameInput).toHaveValue('');
      expect(hostInput).toHaveValue('');
    });
  });

  describe('Form validation', () => {
    async function typeName(value: string) {
      const user = userEvent.setup();
      render(<CreateInferenceProviderSidePanel {...defaultProps} />);
      const dialog = await screen.findByTestId('nv-side-panel-content');
      await openModelProviderSelect(user);
      await user.click(screen.getByRole('option', { name: /openai compatible endpoint/i }));
      const nameInput = within(dialog).getByRole('textbox', { name: 'Name' });
      fireEvent.change(nameInput, { target: { value } });
      fireEvent.blur(nameInput);
    }

    it('shows a live preview of the sanitized name for messy input', async () => {
      await typeName('My Provider 2');
      expect(await screen.findByText('my-provider-2')).toBeInTheDocument();
    });

    it('shows an error after blur when the name contains no letters or numbers', async () => {
      await typeName('!!!');
      expect(
        await screen.findByText('Name must start with a letter and be at least 2 characters.')
      ).toBeInTheDocument();
    });

    it('shows an error when the name is already taken', async () => {
      server.use(
        http.get(`${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/providers`, () =>
          HttpResponse.json({
            data: [{ name: 'taken', host_url: 'https://api.example.com/v1' }],
            pagination: {
              page: 1,
              page_size: 100,
              current_page_size: 1,
              total_pages: 1,
              total_results: 1,
            },
          })
        )
      );
      await typeName('taken');
      expect(
        await screen.findByText(/A provider with this name already exists/)
      ).toBeInTheDocument();
    });

    it('blocks submit until the existing providers have loaded', async () => {
      server.use(
        http.get(
          `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/providers`,
          async () => {
            await delay(200);
            return HttpResponse.json({
              data: [{ name: 'taken', host_url: 'https://api.example.com/v1' }],
              pagination: {
                page: 1,
                page_size: 100,
                current_page_size: 1,
                total_pages: 1,
                total_results: 1,
              },
            });
          }
        )
      );
      render(<CreateInferenceProviderSidePanel {...defaultProps} />);
      const dialog = await screen.findByTestId('nv-side-panel-content');
      const submit = within(dialog).getByRole('button', { name: 'Add Provider' });
      expect(submit).toBeDisabled();
      await waitFor(() => expect(submit).toBeEnabled());
    });

    it('shows validation error for invalid URL', async () => {
      const user = userEvent.setup();
      render(<CreateInferenceProviderSidePanel {...defaultProps} />);
      const dialog = await screen.findByTestId('nv-side-panel-content');
      await openModelProviderSelect(user);
      await user.click(screen.getByRole('option', { name: /openai compatible endpoint/i }));
      const nameInput = within(dialog).getByRole('textbox', { name: 'Name' });
      const hostInput = within(dialog).getByRole('textbox', { name: 'Host URL' });
      fireEvent.change(nameInput, { target: { value: 'myprovider' } });
      fireEvent.change(hostInput, { target: { value: 'not-a-url' } });
      fireEvent.blur(hostInput);
      expect(await screen.findByText(/Enter a valid URL/)).toBeInTheDocument();
    });
  });

  describe('Submit', () => {
    it('calls create provider with form data when Add Provider is clicked (custom)', async () => {
      const user = userEvent.setup();
      const createRequests: Array<{ workspace: string; data: unknown }> = [];
      server.use(
        http.post(
          `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/providers`,
          async ({ request, params }) => {
            createRequests.push({
              workspace: params.workspace as string,
              data: await request.json(),
            });
            return HttpResponse.json(
              {
                name: 'myprovider',
                workspace: params.workspace,
                host_url: 'https://api.example.com/v1',
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
              },
              { status: 201 }
            );
          }
        )
      );

      render(<CreateInferenceProviderSidePanel {...defaultProps} />);
      const dialog = await screen.findByTestId('nv-side-panel-content');
      await openModelProviderSelect(user);
      await user.click(screen.getByRole('option', { name: /openai compatible endpoint/i }));
      fireEvent.change(within(dialog).getByRole('textbox', { name: 'Name' }), {
        target: { value: 'myprovider' },
      });
      fireEvent.change(within(dialog).getByRole('textbox', { name: 'Host URL' }), {
        target: { value: 'https://api.example.com/v1' },
      });
      await user.click(screen.getByRole('button', { name: 'Add Provider' }));

      await waitFor(() => {
        expect(createRequests[0]).toEqual({
          workspace: 'test-workspace',
          data: {
            name: 'myprovider',
            host_url: 'https://api.example.com/v1',
            api_key_secret_name: undefined,
          },
        });
      });
    });

    it('calls create provider with preset data when Add Provider is clicked (NVIDIA Build)', async () => {
      const user = userEvent.setup();
      const createRequests: Array<{ workspace: string; data: unknown }> = [];
      server.use(
        http.post(
          `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/providers`,
          async ({ request, params }) => {
            createRequests.push({
              workspace: params.workspace as string,
              data: await request.json(),
            });
            return HttpResponse.json(
              {
                name: 'build',
                workspace: params.workspace,
                host_url: 'https://integrate.api.nvidia.com/v1',
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
              },
              { status: 201 }
            );
          }
        )
      );

      render(<CreateInferenceProviderSidePanel {...defaultProps} />);
      await screen.findByTestId('nv-side-panel-content');
      await openModelProviderSelect(user);
      await user.click(screen.getByRole('option', { name: /^nvidia build$/i }));
      await user.click(screen.getByRole('button', { name: 'Add Provider' }));

      await waitFor(() => {
        expect(createRequests[0].data).toEqual({
          name: 'build',
          host_url: 'https://integrate.api.nvidia.com/v1',
          api_key_secret_name: undefined,
        });
      });

      await waitFor(() => {
        expect(mockOnClose).toHaveBeenCalled();
      });
    });

    it('shows error message when create fails', async () => {
      const user = userEvent.setup();
      server.use(
        http.post(`${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/providers`, () =>
          HttpResponse.json({ detail: 'Provider already exists' }, { status: 409 })
        )
      );

      render(<CreateInferenceProviderSidePanel {...defaultProps} />);
      await screen.findByTestId('nv-side-panel-content');
      await openModelProviderSelect(user);
      await user.click(screen.getByRole('option', { name: /^nvidia build$/i }));
      await user.click(screen.getByRole('button', { name: 'Add Provider' }));

      expect(
        await screen.findByText(
          (content) => content.includes('Provider already exists') || content.includes('detail')
        )
      ).toBeInTheDocument();
      expect(mockOnClose).not.toHaveBeenCalled();
    });
  });

  describe('Close', () => {
    it('calls onClose when Cancel is clicked', async () => {
      const user = userEvent.setup();
      render(<CreateInferenceProviderSidePanel {...defaultProps} />);
      const panel = await screen.findByTestId('nv-side-panel-content');
      await user.click(within(panel).getByRole('button', { name: 'Cancel' }));
      expect(mockOnClose).toHaveBeenCalled();
    });
  });

  describe('API Key Secret', () => {
    it('opens Create Secret modal when New Secret is chosen from the secret dropdown', async () => {
      const user = userEvent.setup();
      render(<CreateInferenceProviderSidePanel {...defaultProps} />);
      const panel = await screen.findByTestId('nv-side-panel-content');

      await user.click(within(panel).getByRole('combobox', { name: /api key secret/i }));
      await user.click(await screen.findByRole('menuitem', { name: 'New Secret' }));

      await waitFor(() => {
        const modal = screen.getByTestId('nv-modal-content');
        expect(modal).toHaveAttribute('data-state', 'open');
        expect(within(modal).getByRole('heading', { name: 'Create Secret' })).toBeInTheDocument();
      });
    });
  });

  describe('defaultPreset prop', () => {
    it('shows NVIDIA Build as the selected provider in the combobox when defaultPreset="build"', async () => {
      render(<CreateInferenceProviderSidePanel {...defaultProps} defaultPreset="build" />);
      await screen.findByTestId('nv-side-panel-content');
      const combobox = screen.getByRole('combobox', { name: /model provider/i });
      expect(combobox).toHaveTextContent(/nvidia build/i);
      expect(await screen.findByRole('button', { name: 'Add Provider' })).toBeEnabled();
    });

    it('defaults to OpenAI Compatible Endpoint when no defaultPreset is provided', async () => {
      render(<CreateInferenceProviderSidePanel {...defaultProps} />);
      await screen.findByTestId('nv-side-panel-content');
      const combobox = screen.getByRole('combobox', { name: /model provider/i });
      expect(combobox).toHaveTextContent(/openai compatible endpoint/i);
    });
  });

  describe('Preset disabled when already added', () => {
    it('disables NVIDIA Build option when build provider exists', async () => {
      const user = userEvent.setup();
      server.use(
        http.get(
          `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/providers`,
          ({ params: { workspace } }) =>
            HttpResponse.json({
              data: [
                {
                  name: 'build',
                  workspace,
                  host_url: 'https://integrate.api.nvidia.com/v1',
                  created_at: new Date().toISOString(),
                  updated_at: new Date().toISOString(),
                },
              ],
              pagination: {
                page: 1,
                page_size: 100,
                current_page_size: 1,
                total_pages: 1,
                total_results: 1,
              },
            })
        )
      );

      render(<CreateInferenceProviderSidePanel {...defaultProps} />);
      await screen.findByTestId('nv-side-panel-content');
      await openModelProviderSelect(user);
      const buildOption = screen.getByRole('option', { name: /^nvidia build$/i });
      expect(buildOption).toBeDisabled();
    });
  });
});
