// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AssistantToolArgs } from '@nemo/common/src/components/AssistantChat/toolParts';

export const FILE_PATH_ARG_KEYS = [
  'file_path',
  'filePath',
  'path',
  'filename',
  'file_name',
  'file',
] as const;
export const FILE_CONTENT_ARG_KEYS = ['content', 'contents', 'new_content', 'text'] as const;
export const OLD_STRING_ARG_KEYS = ['old_string', 'oldString', 'old_str', 'oldStr'] as const;
export const NEW_STRING_ARG_KEYS = ['new_string', 'newString', 'new_str', 'newStr'] as const;

/** Tools with the built-in file-change renderer; they keep their exact, name-based handling. */
const NAMED_FILE_CHANGE_TOOLS = new Set(['Edit', 'MultiEdit', 'Write']);

/**
 * Known tools that carry a path (or text) but never change a file. They are never inferred as file
 * changes, so e.g. a `Read` with a `path` argument stays a subtle row.
 */
const NON_FILE_CHANGE_TOOLS = new Set([
  'AskUserQuestion',
  'Bash',
  'FindFiles',
  'Glob',
  'Grep',
  'LS',
  'Read',
  'TaskCreate',
  'TaskUpdate',
  'TodoWrite',
  'ToolSearch',
  'WebFetch',
  'WebSearch',
]);

export interface InferredFileChange {
  readonly kind: 'edit' | 'write';
  readonly path: string;
}

const getTrimmedString = (args: AssistantToolArgs, keys: readonly string[]): string | undefined => {
  for (const key of keys) {
    const value = args[key];
    if (typeof value === 'string' && value.trim()) return value.trim();
  }
  return undefined;
};

const hasStringArg = (args: AssistantToolArgs, keys: readonly string[]): boolean =>
  keys.some((key) => typeof args[key] === 'string');

/**
 * Detects a file change from a tool call's argument shape, for tools we do not know by name (a
 * custom agent tool such as `save_note` or `write_file`). A path plus old/new strings reads as an
 * edit; a path plus content reads as a write. Returns undefined for built-in file tools, which are
 * matched by name, and for known tools that never change files.
 */
export const inferFileChangeFromArgs = (
  toolName: string,
  args: AssistantToolArgs,
  builtInToolName: string = toolName
): InferredFileChange | undefined => {
  // `toolName` is the name the agent emitted: only the exact built-in names keep name-based
  // handling, so an alias such as `write_file` is still inferred from its arguments. The resolved
  // `builtInToolName` is used to rule out aliases of read-only tools such as `read_file`.
  if (NAMED_FILE_CHANGE_TOOLS.has(toolName) || NON_FILE_CHANGE_TOOLS.has(builtInToolName)) {
    return undefined;
  }

  const path = getTrimmedString(args, FILE_PATH_ARG_KEYS);
  if (!path) return undefined;

  if (hasStringArg(args, OLD_STRING_ARG_KEYS) && hasStringArg(args, NEW_STRING_ARG_KEYS)) {
    return { kind: 'edit', path };
  }

  if (hasStringArg(args, FILE_CONTENT_ARG_KEYS)) {
    return { kind: 'write', path };
  }

  return undefined;
};
