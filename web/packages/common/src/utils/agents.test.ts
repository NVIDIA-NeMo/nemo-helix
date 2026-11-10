// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getAgentModelNames } from '@nemo/common/src/utils/agents';

describe('getAgentModelNames', () => {
  it('reads NAT workflow llms', () => {
    expect(
      getAgentModelNames({
        llms: { llm: { _type: 'openai', model_name: 'nvidia-llama-3-3-nemotron-super-49b-v1' } },
      })
    ).toEqual(['nvidia-llama-3-3-nemotron-super-49b-v1']);
  });

  it('reads nemo-agents-spec-v1 models and harness models', () => {
    expect(
      getAgentModelNames({
        models: { default: { provider: 'nvidia', model: 'nvidia-nemotron-3-nano-30b-a3b' } },
        harnesses: { deepagents: { kind: 'deepagents', model: { model: 'nvidia-judge-model' } } },
      })
    ).toEqual(['nvidia-nemotron-3-nano-30b-a3b', 'nvidia-judge-model']);
  });

  it('deduplicates repeated model names', () => {
    expect(
      getAgentModelNames({
        llms: {
          a: { _type: 'openai', model_name: 'shared' },
          b: { _type: 'openai', model_name: 'shared' },
        },
        models: { default: { model: 'shared' } },
      })
    ).toEqual(['shared']);
  });

  it('returns an empty list for a missing or model-less config', () => {
    expect(getAgentModelNames(undefined)).toEqual([]);
    expect(getAgentModelNames({ models: { default: { provider: 'nvidia' } } })).toEqual([]);
  });
});
