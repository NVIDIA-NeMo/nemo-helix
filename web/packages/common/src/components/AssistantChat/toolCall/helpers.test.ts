// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getSubtleToolDetail,
  getSubtleToolGroupActions,
  getSubtleToolMessage,
  groupRepeatedSubtleToolActions,
  resolveToolName,
} from '@nemo/common/src/components/AssistantChat/toolCall/helpers';

describe('resolveToolName', () => {
  it.each([
    ['WebSearch', 'WebSearch'],
    ['web_search', 'WebSearch'],
    ['search-web', 'WebSearch'],
    ['WEB_SEARCH', 'WebSearch'],
    ['mcp__browser__web_search', 'WebSearch'],
    ['mcp__filesystem__read_file', 'Read'],
    ['read_file', 'Read'],
    ['write_file', 'Write'],
    ['list_directory', 'LS'],
    ['ls', 'LS'],
    ['run_command', 'Bash'],
    ['multi_edit', 'MultiEdit'],
    ['find-files', 'FindFiles'],
  ])('resolves %s to %s', (toolName, expected) => {
    expect(resolveToolName(toolName)).toBe(expected);
  });

  it.each(['get_weather', 'save_note', 'mcp__nemo_studio__job_progress', 'AssistantThinking'])(
    'leaves %s unchanged',
    (toolName) => {
      expect(resolveToolName(toolName)).toBe(toolName);
    }
  );
});

describe('getSubtleToolGroupActions', () => {
  it('resolves action tool names so variants of one tool group together', () => {
    const actions = getSubtleToolGroupActions({
      actions: [
        { toolCallId: 'a', toolName: 'read_file', args: { path: 'a.ts' } },
        { toolCallId: 'b', toolName: 'mcp__fs__Read', args: { path: 'b.ts' } },
      ],
    });

    expect(actions.map((action) => action.toolName)).toEqual(['Read', 'Read']);
    expect(groupRepeatedSubtleToolActions(actions)).toEqual([
      { actions, repeatedTitle: 'Read 2 files' },
    ]);
  });
});

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
    [
      'deploy_guardrail',
      { summary: 'ValueError: model not found', status: 'failed' },
      'Guardrail step failed: ValueError: model not found',
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

  it('keeps the failure wording when a failed guardrail step has no summary', () => {
    expect(getSubtleToolMessage('deploy_guardrail', { status: 'failed' })).toBe(
      'Guardrail step failed'
    );
    expect(getSubtleToolMessage('deploy_guardrail', { summary: '  ', status: 'failed' })).toBe(
      'Guardrail step failed'
    );
  });

  it('uses the activity summary as the detail', () => {
    const message = getSubtleToolMessage('nemo_api', { summary: 'list workspaces' }) ?? '';

    expect(getSubtleToolDetail('nemo_api', { summary: 'list workspaces' }, message)).toBe(
      'list workspaces'
    );
  });
});
