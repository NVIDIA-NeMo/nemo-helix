/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { type ModelProvider, ModelProviderStatus } from '@nemo/sdk/generated/platform/schema';
import { InferenceProviderDetailsSidePanel } from '@studio/routes/InferenceProvidersListRoute/InferenceProviderDetailsSidePanel';
import { render, screen } from '@studio/tests/util/render';

const mockOnClose = vi.fn();

const baseProvider = (overrides: Partial<ModelProvider> = {}): ModelProvider => ({
  name: 'build-nvidia',
  workspace: 'ws-1',
  created_at: '2024-06-01T12:00:00.000Z',
  updated_at: '2024-06-01T12:00:00.000Z',
  host_url: 'https://integrate.api.nvidia.com/v1',
  ...overrides,
});

describe('InferenceProviderDetailsSidePanel', () => {
  it('renders heading, host link, and core labels when open', async () => {
    render(
      <InferenceProviderDetailsSidePanel
        open
        onClose={mockOnClose}
        provider={baseProvider({ status: ModelProviderStatus.READY })}
      />
    );
    expect(await screen.findByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText('build-nvidia')).toBeInTheDocument();
    expect(screen.getByText('https://integrate.api.nvidia.com/v1')).toBeInTheDocument();
    expect(screen.getByText('Created')).toBeInTheDocument();
    expect(screen.getByText('Host URL')).toBeInTheDocument();
    expect(screen.getByText('Status')).toBeInTheDocument();
    expect(screen.getByText('Served models')).toBeInTheDocument();
  });

  it('renders API key secret row only when api_key_secret_name is set', async () => {
    const { rerender } = render(
      <InferenceProviderDetailsSidePanel open onClose={mockOnClose} provider={baseProvider()} />
    );
    await screen.findByRole('dialog');
    expect(screen.queryByText('API key secret')).not.toBeInTheDocument();

    rerender(
      <InferenceProviderDetailsSidePanel
        open
        onClose={mockOnClose}
        provider={baseProvider({ api_key_secret_name: 'my-secret' })}
      />
    );
    expect(screen.getByText('API key secret')).toBeInTheDocument();
    expect(screen.getByText('my-secret')).toBeInTheDocument();
  });

  it('renders status message in a banner when status_message is present', async () => {
    render(
      <InferenceProviderDetailsSidePanel
        open
        onClose={mockOnClose}
        provider={baseProvider({
          status: ModelProviderStatus.ERROR,
          status_message: 'Connection refused',
        })}
      />
    );
    await screen.findByRole('dialog');
    expect(screen.getByText('Status message')).toBeInTheDocument();
    expect(screen.getByText('Connection refused')).toBeInTheDocument();
  });

  it('renders served models', async () => {
    render(
      <InferenceProviderDetailsSidePanel
        open
        onClose={mockOnClose}
        provider={baseProvider({
          served_models: [
            {
              model_entity_id: 'gpt-5.4',
              served_model_name: 'gpt-5.4',
            },
            {
              model_entity_id: 'gpt-oss-120b',
              served_model_name: 'gpt-oss-120b',
            },
          ],
        })}
      />
    );
    await screen.findByRole('dialog');
    expect(screen.getByText('gpt-5.4', { exact: false })).toBeInTheDocument();
    expect(screen.getByText('gpt-oss-120b', { exact: false })).toBeInTheDocument();
  });
});
