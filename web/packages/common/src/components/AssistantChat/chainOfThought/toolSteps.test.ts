// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  FILE_EDIT_STEP,
  findToolStepDefinition,
  getFallbackToolStep,
  getRepeatedToolStepTitle,
  getToolStepTitle,
  getToolStepTitleParts,
  TOOL_STEP_DEFINITIONS,
} from '@nemo/common/src/components/AssistantChat/chainOfThought/toolSteps';

describe('findToolStepDefinition', () => {
  it.each(['WebSearch', 'web_search', 'web-search', 'SEARCH_WEB', 'mcp__browser__web_search'])(
    'matches %s regardless of case, separators, and MCP prefix',
    (toolName) => {
      expect(findToolStepDefinition(toolName)?.done).toBe('Searched web');
    }
  );

  it('returns undefined for tools without a definition', () => {
    expect(findToolStepDefinition('get_weather')).toBeUndefined();
  });

  it('maps every tool name to a single definition', () => {
    const names = TOOL_STEP_DEFINITIONS.flatMap((definition) =>
      definition.names.map((name) => name.toLowerCase())
    );
    expect(new Set(names).size).toBe(names.length);
  });
});

describe('getToolStepTitle', () => {
  const bash = findToolStepDefinition('Bash')!;

  it('uses the running or finished action', () => {
    expect(getToolStepTitle(bash, { args: {}, running: true })).toBe('Running command');
    expect(getToolStepTitle(bash, { args: {} })).toBe('Ran command');
  });

  it('appends the subject on one line', () => {
    expect(getToolStepTitle(bash, { args: { command: 'ls\n  -la' } })).toBe('Ran command ls -la');
  });

  it('returns the action and subject separately so the subject can be styled', () => {
    expect(getToolStepTitleParts(bash, { args: { command: 'pwd' } })).toEqual({
      action: 'Ran command',
      subject: 'pwd',
    });
    expect(getToolStepTitleParts(bash, { args: { command: '  ' } }).subject).toBeUndefined();
    expect(bash.subjectIsCode).toBe(true);
    expect(findToolStepDefinition('WebSearch')?.subjectIsCode).toBeFalsy();
  });

  it('shows file names rather than full paths', () => {
    expect(
      getToolStepTitle(findToolStepDefinition('Read')!, { args: { file_path: 'a/b/README.md' } })
    ).toBe('Read file README.md');
    expect(getToolStepTitle(FILE_EDIT_STEP, { args: { path: 'notes/MEMORY.md' } })).toBe(
      'Edited MEMORY.md'
    );
  });

  it('falls back to the tool name for unknown tools, with a common argument as detail', () => {
    const step = getFallbackToolStep('get_weather');
    expect(getToolStepTitle(step, { args: { city: 'Paris' } })).toBe('Called get_weather');
    expect(getToolStepTitle(step, { args: {}, running: true })).toBe('Calling get_weather');
    expect(getToolStepTitle(step, { args: { query: 'rain' } })).toBe('Called get_weather rain');
  });

  it('joins a search pattern with its path', () => {
    const grep = findToolStepDefinition('Grep')!;
    expect(getToolStepTitle(grep, { args: { pattern: 'TODO', path: 'src' } })).toBe(
      'Searched text TODO in src'
    );
    expect(getToolStepTitle(grep, { args: { pattern: 'TODO' } })).toBe('Searched text TODO');
  });
});

describe('getRepeatedToolStepTitle', () => {
  it('fills in the count, falling back to "{done} N times"', () => {
    expect(getRepeatedToolStepTitle(findToolStepDefinition('Read')!, 3)).toBe('Read 3 files');
    expect(getRepeatedToolStepTitle(getFallbackToolStep('search_configs'), 2)).toBe(
      'Called search_configs 2 times'
    );
    expect(
      getRepeatedToolStepTitle({ ...findToolStepDefinition('Grep')!, repeated: undefined }, 2)
    ).toBe('Searched text 2 times');
  });
});

describe('formatArgs', () => {
  it('shows a Bash command as a shell snippet', () => {
    expect(findToolStepDefinition('Bash')!.formatArgs!({ command: 'ls -la' })).toEqual({
      label: 'Command',
      language: 'bash',
      text: 'ls -la',
    });
    expect(findToolStepDefinition('Bash')!.formatArgs!({})).toBeUndefined();
  });
});
