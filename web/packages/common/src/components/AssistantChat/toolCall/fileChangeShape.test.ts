// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { inferFileChangeFromArgs } from '@nemo/common/src/components/AssistantChat/toolCall/fileChangeShape';
import {
  groupConsecutiveAssistantSubtleToolCalls,
  isAssistantSubtleToolCall,
} from '@nemo/common/src/components/AssistantChat/toolParts';

describe('inferFileChangeFromArgs', () => {
  it.each(['file_path', 'filePath', 'path', 'filename', 'file_name', 'file'])(
    'reads %s as a path',
    (pathKey) => {
      expect(inferFileChangeFromArgs('save_note', { [pathKey]: 'a.md', content: 'x' })).toEqual({
        kind: 'write',
        path: 'a.md',
      });
    }
  );

  it.each(['content', 'contents', 'new_content', 'text'])('reads %s as content', (contentKey) => {
    expect(inferFileChangeFromArgs('save_note', { path: 'a.md', [contentKey]: 'x' })).toEqual({
      kind: 'write',
      path: 'a.md',
    });
  });

  it('treats empty content as a write, since writing an empty file is valid', () => {
    expect(inferFileChangeFromArgs('save_note', { path: 'a.md', content: '' })).toEqual({
      kind: 'write',
      path: 'a.md',
    });
  });

  it.each([
    ['old_string', 'new_string'],
    ['oldString', 'newString'],
    ['old_str', 'new_str'],
  ])('reads %s and %s as an edit, ahead of content', (oldKey, newKey) => {
    expect(
      inferFileChangeFromArgs('patch', { path: 'a.md', [oldKey]: 'a', [newKey]: 'b', content: 'c' })
    ).toEqual({ kind: 'edit', path: 'a.md' });
  });

  it('needs both old and new strings for an edit', () => {
    expect(inferFileChangeFromArgs('patch', { path: 'a.md', old_string: 'a' })).toBeUndefined();
  });

  it('returns undefined without a path or without content', () => {
    expect(inferFileChangeFromArgs('save_note', { content: 'x' })).toBeUndefined();
    expect(inferFileChangeFromArgs('save_note', { path: 'a.md' })).toBeUndefined();
    expect(inferFileChangeFromArgs('save_note', { path: '  ', content: 'x' })).toBeUndefined();
    expect(inferFileChangeFromArgs('save_note', { path: 'a.md', content: 3 })).toBeUndefined();
  });

  it('infers an alias of a file-change tool from its arguments but never an alias of a read-only tool', () => {
    expect(inferFileChangeFromArgs('write_file', { path: 'a.md', contents: 'x' }, 'Write')).toEqual(
      { kind: 'write', path: 'a.md' }
    );
    expect(
      inferFileChangeFromArgs('read_file', { path: 'a.md', text: 'x' }, 'Read')
    ).toBeUndefined();
  });

  it.each(['Write', 'Edit', 'MultiEdit', 'Read', 'Grep', 'Bash', 'LS', 'WebFetch'])(
    'leaves the known tool %s to its own handling',
    (toolName) => {
      expect(inferFileChangeFromArgs(toolName, { path: 'a.md', content: 'x' })).toBeUndefined();
    }
  );
});

describe('isAssistantSubtleToolCall', () => {
  it('excludes a custom tool whose arguments look like a file change', () => {
    expect(isAssistantSubtleToolCall('save_note', { path: 'a.md', content: 'x' })).toBe(false);
  });

  it('keeps ordinary and unknown tools subtle', () => {
    expect(isAssistantSubtleToolCall('Bash', { command: 'pwd' })).toBe(true);
    expect(isAssistantSubtleToolCall('InspectWorkspace', { query: 'x' })).toBe(true);
  });

  it('keeps a shape-matched call out of a subtle group', () => {
    const grouped = groupConsecutiveAssistantSubtleToolCalls([
      {
        type: 'tool-call',
        toolCallId: 'a',
        toolName: 'Bash',
        args: { command: 'pwd' },
        argsText: '{}',
      },
      {
        type: 'tool-call',
        toolCallId: 'b',
        toolName: 'save_note',
        args: { path: 'a.md', content: 'x' },
        argsText: '{}',
      },
      {
        type: 'tool-call',
        toolCallId: 'c',
        toolName: 'Bash',
        args: { command: 'ls' },
        argsText: '{}',
      },
    ]);

    expect(grouped.map((part) => (part.type === 'tool-call' ? part.toolName : part.type))).toEqual([
      'Bash',
      'save_note',
      'Bash',
    ]);
  });
});
