// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { createToastMock } from '@nemo/common/src/providers/toast/useToastMock';
import {
  getInsightsGetAnalysisConfigQueryKey,
  useInsightsGetAnalysisConfig,
} from '@nemo/sdk/generated/insights/insights-analysis-configs';
import type { AnalysisConfig } from '@nemo/sdk/generated/insights/schema';
import { ThemeProvider } from '@nvidia/foundations-react-core';
import { queryClient } from '@studio/api/queryClient';
import { AnalysisConfigPanel } from '@studio/routes/agents/AgentDetailRoute/analysis/AnalysisConfigPanel';
import {
  AnalysisConfigPartialSaveError,
  saveAnalysisConfig,
} from '@studio/routes/agents/AgentDetailRoute/analysis/saveAnalysisConfig';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.mock('@nemo/common/src/providers/toast/useToast');

vi.mock('@nemo/sdk/generated/insights/insights-analysis-configs', () => ({
  useInsightsGetAnalysisConfig: vi.fn(),
  getInsightsGetAnalysisConfigQueryKey: vi.fn((workspace: string, agent: string) => [
    'analysis-config',
    workspace,
    agent,
  ]),
}));

vi.mock('@studio/api/queryClient', () => ({
  queryClient: { invalidateQueries: vi.fn() },
}));

vi.mock(
  '@studio/routes/agents/AgentDetailRoute/analysis/saveAnalysisConfig',
  async (importOriginal) => ({
    ...(await importOriginal<
      typeof import('@studio/routes/agents/AgentDetailRoute/analysis/saveAnalysisConfig')
    >()),
    saveAnalysisConfig: vi.fn(),
  })
);

vi.mock('@nemo/common/src/components/ModelSelectV2', () => ({
  WorkspaceModelSelect: ({
    value,
    onValueChange,
    ...props
  }: {
    value: { model: string } | null;
    onValueChange: (next: { model: string }) => void;
    'aria-label': string;
  }) => (
    <input
      aria-label={props['aria-label']}
      value={value?.model ?? ''}
      onChange={(event) => onValueChange({ model: event.target.value })}
    />
  ),
}));

const useConfig = vi.mocked(useInsightsGetAnalysisConfig);
const save = vi.mocked(saveAnalysisConfig);
const invalidateQueries = vi.mocked(queryClient.invalidateQueries);

const config = (overrides: Partial<AnalysisConfig> = {}): AnalysisConfig => ({
  id: 'insights-analysis-config-1',
  entity_id: 'insights-analysis-config-1',
  parent: 'ws-default',
  db_version: 1,
  name: 'email-security-triage',
  workspace: 'demo-epa',
  agent: 'email-security-triage',
  enabled: true,
  default_model: 'demo-epa/slow',
  fast_model: 'demo-epa/fast',
  created_at: '2026-08-14T09:00:00Z',
  created_by: 'user@example.com',
  updated_at: '2026-08-14T09:00:00Z',
  updated_by: 'user@example.com',
  ...overrides,
});

type ConfigQueryResult = ReturnType<typeof useInsightsGetAnalysisConfig>;

const queryResult = (data: AnalysisConfig | undefined): ConfigQueryResult =>
  ({
    data,
    isLoading: false,
    isError: data === undefined,
    error: data === undefined ? { response: { status: 404 } } : null,
  }) as ConfigQueryResult;

const renderPanel = (agent: string) =>
  render(
    <ThemeProvider>
      <AnalysisConfigPanel workspace="demo-epa" agent={agent} />
    </ThemeProvider>
  );

let toast = createToastMock();

beforeEach(() => {
  vi.clearAllMocks();
  toast = createToastMock();
  vi.mocked(useToast).mockReturnValue(toast);
  invalidateQueries.mockResolvedValue(undefined);
});

describe('AnalysisConfigPanel', () => {
  it('refreshes the config and reports a partial commit when the disable step fails', async () => {
    const user = userEvent.setup();
    useConfig.mockReturnValue(queryResult(config({ enabled: false })));
    save.mockRejectedValue(new AnalysisConfigPartialSaveError(new Error('boom')));

    renderPanel('email-security-triage');

    await user.click(screen.getByRole('button', { name: 'Edit' }));
    await user.clear(screen.getByLabelText('Default model'));
    await user.type(screen.getByLabelText('Default model'), 'demo-epa/new-slow');
    await user.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() => expect(toast.error).toHaveBeenCalled());
    expect(vi.mocked(toast.error).mock.calls[0][0]).toContain('may still be enabled');
    expect(toast.success).not.toHaveBeenCalled();

    expect(getInsightsGetAnalysisConfigQueryKey).toHaveBeenCalledWith(
      'demo-epa',
      'email-security-triage'
    );
    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: ['analysis-config', 'demo-epa', 'email-security-triage'],
    });

    // The save failed, so the panel keeps the draft open instead of claiming success.
    expect(screen.getByRole('button', { name: 'Save' })).toBeInTheDocument();
  });

  it('refreshes the config after a successful save', async () => {
    const user = userEvent.setup();
    useConfig.mockReturnValue(queryResult(config()));
    save.mockResolvedValue(config({ enabled: false }));

    renderPanel('email-security-triage');

    await user.click(screen.getByRole('button', { name: 'Edit' }));
    await user.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() => expect(toast.success).toHaveBeenCalled());
    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: ['analysis-config', 'demo-epa', 'email-security-triage'],
    });
  });

  it('drops the draft and leaves edit mode when the agent changes with no config on either', async () => {
    const user = userEvent.setup();
    useConfig.mockReturnValue(queryResult(undefined));

    const { rerender } = renderPanel('email-security-triage');

    await user.click(screen.getByRole('button', { name: 'Edit' }));
    await user.type(screen.getByLabelText('Default model'), 'demo-epa/slow');
    expect(screen.getByLabelText('Default model')).toHaveValue('demo-epa/slow');

    rerender(
      <ThemeProvider>
        <AnalysisConfigPanel workspace="demo-epa" agent="other-agent" />
      </ThemeProvider>
    );

    expect(screen.getByRole('button', { name: 'Edit' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Save' })).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Edit' }));
    expect(screen.getByLabelText('Default model')).toHaveValue('');
    expect(save).not.toHaveBeenCalled();
  });
});
