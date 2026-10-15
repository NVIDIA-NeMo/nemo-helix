// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  agentNameFromImage,
  imageReferenceProblem,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/image';

describe('imageReferenceProblem', () => {
  it.each([
    'docker.io/sgavrilovic951/sample_agent:0.1',
    'nvcr.io/nvidia/nemo/agent:1.0.0-rc1',
    'localhost:5000/agent',
    'agent',
    'org/agent@sha256:' + 'a'.repeat(64),
    'registry.example.com:443/team/agent:v2@sha256:' + 'b'.repeat(64),
  ])('accepts %s', (reference) => {
    expect(imageReferenceProblem(reference)).toBeUndefined();
  });

  it('says nothing about an empty field', () => {
    expect(imageReferenceProblem('  ')).toBeUndefined();
  });

  it('asks for the reference rather than a URL', () => {
    expect(imageReferenceProblem('https://docker.io/org/agent:1')).toMatch(/Leave out the scheme/);
  });

  it.each(['Org/Agent:1', 'org/agent:', 'org agent', 'org//agent'])('rejects %s', (reference) => {
    expect(imageReferenceProblem(reference)).toMatch(/is not an image reference/);
  });
});

describe('agentNameFromImage', () => {
  it.each([
    ['docker.io/sgavrilovic951/sample_agent:0.1', 'sample-agent'],
    ['localhost:5000/agent', 'agent'],
    ['org/my.agent@sha256:' + 'a'.repeat(64), 'my-agent'],
  ])('names %s as %s', (reference, name) => {
    expect(agentNameFromImage(reference)).toBe(name);
  });
});
