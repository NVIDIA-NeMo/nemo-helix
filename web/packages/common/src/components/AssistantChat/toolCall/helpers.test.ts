// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getSubtleToolGroupActions,
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
