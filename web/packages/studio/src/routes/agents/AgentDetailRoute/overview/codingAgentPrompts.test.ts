// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { traceImportPrompt } from '@studio/routes/agents/AgentDetailRoute/overview/codingAgentPrompts';

const params = { workspace: 'default', baseUrl: 'http://localhost:8080' };

describe('traceImportPrompt', () => {
  it('asks for an analyst run once the agent it belongs to is known', () => {
    const prompt = traceImportPrompt({ ...params, agent: 'email-triage' });

    expect(prompt).toContain('## Then turn the traces into insights');
    expect(prompt).toContain('nemo insights analysis enable');
    expect(prompt).toContain('export AGENT_NAME=email-triage');
  });

  it('leaves analysis out when no agent is named, since it is enabled one agent at a time', () => {
    const prompt = traceImportPrompt(params);

    expect(prompt).not.toContain('## Then turn the traces into insights');
    expect(prompt).not.toContain('nemo insights analysis enable');
    expect(prompt).toContain('export AGENT_NAME=<agent-name>');
  });
});
