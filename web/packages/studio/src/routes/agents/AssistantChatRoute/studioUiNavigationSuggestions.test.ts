// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getStudioUiNavigationSuggestion } from '@studio/routes/agents/AssistantChatRoute/studioUiNavigationSuggestions';
import { mockFeatureFlags } from '@studio/tests/util/mockFeatureFlags';

const workspace = 'default';

describe('getStudioUiNavigationSuggestion', () => {
  beforeEach(() => {
    mockFeatureFlags({
      agentsEnabled: true,
      customizerEnabled: true,
      dataDesignerEnabled: true,
      datasetsEnabled: true,
      deploymentsEnabled: true,
      evaluatorEnabled: true,
      guardrailsEnabled: true,
      inferenceProviderEnabled: true,
      intakeEnabled: true,
      jobsEnabled: true,
      modelCompareEnabled: true,
      safeSynthesizerEnabled: true,
      secretsEnabled: true,
      settingsEnabled: true,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('returns the matching Studio destination for a product workflow', () => {
    expect(getStudioUiNavigationSuggestion('Add guardrails to an agent', workspace)).toMatchObject({
      id: 'guardrails',
      href: '/workspaces/default/guardrails',
      title: 'Open Guardrails',
    });

    expect(
      getStudioUiNavigationSuggestion('Generate a synthetic dataset for fine tuning', workspace)
    ).toMatchObject({
      id: 'safe-synthesizer-new',
      href: '/workspaces/default/safe-synthesizer/new',
    });
  });

  it('keeps navigation shortcuts when prompts include Studio product context', () => {
    const cases = [
      {
        prompt: 'Manage workspace secrets',
        id: 'secrets',
      },
      {
        prompt: 'Open model playground',
        id: 'model-playground',
      },
      {
        prompt: 'Show workspace job history',
        id: 'jobs',
      },
      {
        prompt: 'Review intake annotations',
        id: 'traces',
      },
      {
        prompt: 'Open workspace settings',
        id: 'settings',
      },
      {
        prompt: 'Use the guardrails-plugin skill to debug guardrail middleware',
        id: 'guardrails',
      },
      {
        prompt: 'Use the inference skill to configure inference in this workspace',
        id: 'inference-providers',
      },
      {
        prompt: 'Use the nemo-build-agent skill to build an agent',
        id: 'agents',
      },
      {
        prompt: 'Use the nemo-evaluator skill to review eval history',
        id: 'evaluations',
      },
      {
        prompt: 'Use the safe-synthesizer skill to generate safety data',
        id: 'safe-synthesizer-new',
      },
      {
        prompt: 'Create a data generation workflow',
        id: 'data-designer-new',
      },
    ];

    for (const { prompt, id } of cases) {
      expect(getStudioUiNavigationSuggestion(prompt, workspace)).toMatchObject({ id });
    }
  });

  it('uses the traces destination for intake review prompts', () => {
    expect(getStudioUiNavigationSuggestion('Review intake annotations', workspace)).toMatchObject({
      id: 'traces',
      title: 'Open Traces',
      href: '/workspaces/default/intake/traces',
    });
  });

  it('returns undefined when the matching feature is disabled', () => {
    mockFeatureFlags({ guardrailsEnabled: false });

    expect(getStudioUiNavigationSuggestion('Create guardrails for this agent', workspace)).toBe(
      undefined
    );
  });

  it('does not interrupt ordinary assistant prompts', () => {
    const prompts = [
      'Review the current working tree',
      'Fix the settings page component',
      'Update settings page component',
      'Analyze token usage in this parser',
      'Add annotation support to the chart',
      'Open the playground component',
      'Create token validation helpers',
      'Tune model sizing calculations in this file',
      'Review job history component',
      'Build an agent class for this test helper',
      'Configure inference in this TypeScript module',
      'Generate synthetic test data in fixtures',
      'Review evaluator plugin imports',
    ];

    for (const prompt of prompts) {
      expect(getStudioUiNavigationSuggestion(prompt, workspace)).toBe(undefined);
    }
  });
});
