// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentConfig } from '@nemo/common/src/utils/agents';
import {
  applyModelToConfig,
  buildClonedAgentName,
  getPrimaryModelName,
} from '@studio/routes/agents/AgentsListRoute/CloneAgentModal/const';

describe('buildClonedAgentName', () => {
  it('appends a short random suffix to the source name', () => {
    expect(buildClonedAgentName('react-agent')).toMatch(/^react-agent-[a-z0-9]{6}$/);
  });

  it('produces a different suffix on each call', () => {
    expect(buildClonedAgentName('a')).not.toBe(buildClonedAgentName('a'));
  });
});

describe('getPrimaryModelName', () => {
  it('returns the model of the llm referenced by workflow.llm_name', () => {
    const config: AgentConfig = {
      llms: {
        llm: { _type: 'openai', model_name: 'primary-model' },
        embedding: { _type: 'openai', model_name: 'embed-model' },
      },
      workflow: { _type: 'react_agent', llm_name: 'llm' },
    };
    expect(getPrimaryModelName(config)).toBe('primary-model');
  });

  it('falls back to the first model when no workflow llm is identifiable', () => {
    const config: AgentConfig = {
      llms: { onlyLlm: { _type: 'openai', model_name: 'solo-model' } },
    };
    expect(getPrimaryModelName(config)).toBe('solo-model');
  });

  it('returns undefined when there are no models', () => {
    expect(getPrimaryModelName(undefined)).toBeUndefined();
    expect(getPrimaryModelName({})).toBeUndefined();
  });
});

describe('applyModelToConfig', () => {
  it('swaps only the workflow primary llm when it is identifiable', () => {
    const config: AgentConfig = {
      llms: {
        llm: { _type: 'openai', model_name: 'old-primary' },
        embedding: { _type: 'openai', model_name: 'embed-model' },
      },
      workflow: { _type: 'react_agent', llm_name: 'llm' },
    };
    const result = applyModelToConfig(config, 'new-model') as AgentConfig;
    expect(result.llms?.llm.model_name).toBe('new-model');
    // Non-primary llms (e.g. embedding) are left alone.
    expect(result.llms?.embedding.model_name).toBe('embed-model');
  });

  it('swaps every llm that declares a model when no primary is identifiable', () => {
    const config: AgentConfig = {
      llms: {
        a: { _type: 'openai', model_name: 'old-a' },
        b: { _type: 'openai', model_name: 'old-b' },
      },
    };
    const result = applyModelToConfig(config, 'new-model') as AgentConfig;
    expect(result.llms?.a.model_name).toBe('new-model');
    expect(result.llms?.b.model_name).toBe('new-model');
  });

  it('does not mutate the source config', () => {
    const config: AgentConfig = {
      llms: { llm: { _type: 'openai', model_name: 'old' } },
      workflow: { _type: 'react_agent', llm_name: 'llm' },
    };
    applyModelToConfig(config, 'new-model');
    expect(config.llms?.llm.model_name).toBe('old');
  });

  it('preserves config keys outside of llms', () => {
    const config = {
      function_groups: { calculator: { _type: 'calculator' } },
      llms: { llm: { _type: 'openai', model_name: 'old' } },
      workflow: { _type: 'react_agent', llm_name: 'llm', tool_names: ['calculator'] },
    } as AgentConfig;
    const result = applyModelToConfig(config, 'new-model') as Record<string, unknown>;
    expect(result.function_groups).toEqual({ calculator: { _type: 'calculator' } });
    expect((result.workflow as { tool_names: string[] }).tool_names).toEqual(['calculator']);
  });

  it('returns an empty config when the source has none', () => {
    expect(applyModelToConfig(undefined, 'new-model')).toEqual({});
  });
});
