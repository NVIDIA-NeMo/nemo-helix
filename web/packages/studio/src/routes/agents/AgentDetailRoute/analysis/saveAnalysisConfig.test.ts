// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  insightsDisableAnalysisConfig,
  insightsEnableAnalysisConfig,
} from '@nemo/sdk/generated/insights/insights-analysis-configs';
import type { AnalysisConfig } from '@nemo/sdk/generated/insights/schema';
import {
  AnalysisConfigPartialSaveError,
  saveAnalysisConfig,
} from '@studio/routes/agents/AgentDetailRoute/analysis/saveAnalysisConfig';

vi.mock('@nemo/sdk/generated/insights/insights-analysis-configs', async (importOriginal) => ({
  ...(await importOriginal<
    typeof import('@nemo/sdk/generated/insights/insights-analysis-configs')
  >()),
  insightsEnableAnalysisConfig: vi.fn(),
  insightsDisableAnalysisConfig: vi.fn(),
}));

const enable = vi.mocked(insightsEnableAnalysisConfig);
const disable = vi.mocked(insightsDisableAnalysisConfig);

const stored = (overrides: Partial<AnalysisConfig> = {}): AnalysisConfig => ({
  id: 'insights-analysis-config-1',
  entity_id: 'insights-analysis-config-1',
  parent: 'ws-default',
  db_version: 1,
  name: 'email-security-triage',
  workspace: 'default',
  agent: 'email-security-triage',
  enabled: true,
  default_model: 'default/slow',
  fast_model: 'default/fast',
  created_at: '2026-08-14T09:00:00Z',
  created_by: 'user@example.com',
  updated_at: '2026-08-14T09:00:00Z',
  updated_by: 'user@example.com',
  ...overrides,
});

beforeEach(() => {
  vi.resetAllMocks();
  enable.mockResolvedValue(stored());
  disable.mockResolvedValue(stored({ enabled: false }));
});

describe('saveAnalysisConfig', () => {
  it('does nothing when neither the flag nor the models changed', async () => {
    const config = stored();

    const result = await saveAnalysisConfig(
      'demo-epa',
      'email-security-triage',
      { enabled: true, defaultModel: 'default/slow', fastModel: 'default/fast' },
      config
    );

    expect(result).toBe(config);
    expect(enable).not.toHaveBeenCalled();
    expect(disable).not.toHaveBeenCalled();
  });

  it('disables without touching the models when only the flag changed', async () => {
    await saveAnalysisConfig(
      'demo-epa',
      'email-security-triage',
      { enabled: false, defaultModel: 'default/slow', fastModel: 'default/fast' },
      stored()
    );

    expect(disable).toHaveBeenCalledWith('demo-epa', 'email-security-triage');
    expect(enable).not.toHaveBeenCalled();
  });

  it('re-enables with the stored pair when only the flag changed', async () => {
    await saveAnalysisConfig(
      'demo-epa',
      'email-security-triage',
      { enabled: true, defaultModel: 'default/slow', fastModel: 'default/fast' },
      stored({ enabled: false })
    );

    expect(enable).toHaveBeenCalledWith('demo-epa', 'email-security-triage', {
      default_model: 'default/slow',
      fast_model: 'default/fast',
    });
    expect(disable).not.toHaveBeenCalled();
  });

  it('writes changed models through enable', async () => {
    await saveAnalysisConfig(
      'demo-epa',
      'email-security-triage',
      { enabled: true, defaultModel: 'default/new-slow', fastModel: 'default/fast' },
      stored()
    );

    expect(enable).toHaveBeenCalledWith('demo-epa', 'email-security-triage', {
      default_model: 'default/new-slow',
      fast_model: 'default/fast',
    });
    expect(disable).not.toHaveBeenCalled();
  });

  it('writes models then disables when the pair changed and the agent stays off', async () => {
    await saveAnalysisConfig(
      'demo-epa',
      'email-security-triage',
      { enabled: false, defaultModel: 'default/new-slow', fastModel: 'default/fast' },
      stored({ enabled: false })
    );

    expect(enable).toHaveBeenCalledWith('demo-epa', 'email-security-triage', {
      default_model: 'default/new-slow',
      fast_model: 'default/fast',
    });
    expect(disable).toHaveBeenCalledWith('demo-epa', 'email-security-triage');
  });

  it('creates the config through enable when none is stored', async () => {
    await saveAnalysisConfig(
      'demo-epa',
      'email-security-triage',
      { enabled: true, defaultModel: 'default/slow', fastModel: 'default/fast' },
      undefined
    );

    expect(enable).toHaveBeenCalledWith('demo-epa', 'email-security-triage', {
      default_model: 'default/slow',
      fast_model: 'default/fast',
    });
  });

  it('creates a disabled config as enable followed by disable', async () => {
    await saveAnalysisConfig(
      'demo-epa',
      'email-security-triage',
      { enabled: false, defaultModel: 'default/slow', fastModel: 'default/fast' },
      undefined
    );

    expect(enable).toHaveBeenCalled();
    expect(disable).toHaveBeenCalledWith('demo-epa', 'email-security-triage');
  });

  it('reports a partial commit when the disable step fails after enabling', async () => {
    const disableError = new Error('boom');
    disable.mockRejectedValue(disableError);

    await expect(
      saveAnalysisConfig(
        'demo-epa',
        'email-security-triage',
        { enabled: false, defaultModel: 'default/new-slow', fastModel: 'default/fast' },
        stored({ enabled: false })
      )
    ).rejects.toBeInstanceOf(AnalysisConfigPartialSaveError);

    expect(enable).toHaveBeenCalled();
  });

  it('propagates a disable failure as-is when no enable preceded it', async () => {
    const disableError = new Error('boom');
    disable.mockRejectedValue(disableError);

    await expect(
      saveAnalysisConfig(
        'demo-epa',
        'email-security-triage',
        { enabled: false, defaultModel: 'default/slow', fastModel: 'default/fast' },
        stored()
      )
    ).rejects.toBe(disableError);

    expect(enable).not.toHaveBeenCalled();
  });
});
