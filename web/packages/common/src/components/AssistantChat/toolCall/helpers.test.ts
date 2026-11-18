// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getSubtleToolDetail,
  getSubtleToolMessage,
} from '@nemo/common/src/components/AssistantChat/toolCall/helpers';

describe('Studio Assistant activity messages', () => {
  it.each([
    ['nemo_api', { summary: 'list workspaces' }, 'Called list workspaces'],
    [
      'search_docs',
      { query: 'guardrail virtual model' },
      'Searched docs for guardrail virtual model',
    ],
    ['check_status', { summary: 'evaluator job eval-1' }, 'Checked evaluator job eval-1'],
    [
      'deploy_guardrail',
      { summary: 'Created guardrail config' },
      'Guardrail: Created guardrail config',
    ],
  ])('describes %s activity', (toolName, args, expected) => {
    expect(getSubtleToolMessage(toolName, args)).toBe(expected);
  });

  it.each([
    ['nemo_api', 'Called platform API'],
    ['search_docs', 'Searched docs'],
    ['check_status', 'Checked job status'],
    ['deploy_guardrail', 'Deployed guardrail'],
  ])('falls back when %s activity has no summary', (toolName, expected) => {
    expect(getSubtleToolMessage(toolName, {})).toBe(expected);
  });

  it('uses the activity summary as the detail', () => {
    const message = getSubtleToolMessage('nemo_api', { summary: 'list workspaces' }) ?? '';

    expect(getSubtleToolDetail('nemo_api', { summary: 'list workspaces' }, message)).toBe(
      'list workspaces'
    );
  });
});
