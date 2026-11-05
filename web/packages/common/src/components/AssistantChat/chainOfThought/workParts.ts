// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
  ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
  ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
  ASSISTANT_THINKING_TOOL_NAME,
  isAssistantJobProgressToolName,
} from '@nemo/common/src/components/AssistantChat/toolParts';
import type { CodeSnippetLanguage } from '@nvidia/foundations-react-core';

/** Group key for a run of adjacent reasoning and tool-call parts. */
export const WORK_GROUP_KEY = 'group-work';

/** `message.metadata.custom` key holding how long each work group in the reply took, in order. */
export const WORK_DURATIONS_METADATA_KEY = 'workDurationsMs';

interface WorkPartLike {
  readonly type: string;
  readonly toolName?: string;
  readonly args?: unknown;
}

export const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

// Job progress cards stay visible next to the answer instead of hiding in the collapsed group.
export const isWorkPart = (part: WorkPartLike): boolean =>
  part.type === 'reasoning' ||
  (part.type === 'tool-call' && !isAssistantJobProgressToolName(part.toolName ?? ''));

export const groupWorkParts = (part: WorkPartLike): readonly [typeof WORK_GROUP_KEY] | null =>
  isWorkPart(part) ? [WORK_GROUP_KEY] : null;

/**
 * Counts the real tool calls a part stands for. Synthetic parts from the NeMo Assistant runtime
 * either hold thinking (no calls) or bundle several calls (a subtle group or collapsed details).
 */
export const countToolCalls = (part: WorkPartLike): number => {
  if (part.type !== 'tool-call') return 0;
  const args = isRecord(part.args) ? part.args : {};

  switch (part.toolName) {
    case ASSISTANT_THINKING_TOOL_NAME:
    case ASSISTANT_COLLAPSED_THINKING_TOOL_NAME:
      return 0;
    case ASSISTANT_SUBTLE_TOOL_GROUP_NAME:
      return Array.isArray(args.actions) ? args.actions.length : 0;
    case ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME:
      return Array.isArray(args.parts)
        ? args.parts.filter(isRecord).reduce(
            (total, nested) =>
              total +
              countToolCalls({
                type: String(nested.type),
                toolName: typeof nested.toolName === 'string' ? nested.toolName : undefined,
                args: nested.args,
              }),
            0
          )
        : 0;
    default:
      return 1;
  }
};

export const getWorkDurations = (custom: Record<string, unknown> | undefined): number[] => {
  const durations = custom?.[WORK_DURATIONS_METADATA_KEY];
  return Array.isArray(durations)
    ? durations.map((duration) => (typeof duration === 'number' ? duration : Number.NaN))
    : [];
};

/** The position of the work group starting at `startIndex` among the reply's work groups. */
export const getWorkGroupOrdinal = (parts: readonly WorkPartLike[], startIndex: number): number => {
  let ordinal = -1;
  for (let index = 0; index <= startIndex && index < parts.length; index += 1) {
    const startsGroup =
      isWorkPart(parts[index]!) && (index === 0 || !isWorkPart(parts[index - 1]!));
    if (startsGroup) ordinal += 1;
  }
  return Math.max(ordinal, 0);
};

export const formatWorkDuration = (durationMs: number): string => {
  const seconds = Math.round(durationMs / 1000);
  if (seconds < 1) return '<1s';
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`;
};

export const getWorkGroupLabel = ({
  durationMs,
  running,
  toolCallCount,
}: {
  readonly durationMs?: number;
  readonly running: boolean;
  readonly toolCallCount: number;
}): string => {
  if (running) return 'Thinking…';
  const summary =
    toolCallCount > 0
      ? `Ran ${toolCallCount} tool call${toolCallCount === 1 ? '' : 's'}`
      : 'Thought';
  return durationMs !== undefined && Number.isFinite(durationMs)
    ? `${summary} · ${formatWorkDuration(durationMs)}`
    : summary;
};

export const FILE_PATH_KEYS = ['file_path', 'filePath', 'path', 'filename', 'file_name', 'file'];
const FILE_CONTENT_KEYS = ['content', 'contents', 'new_string', 'newString', 'new_content', 'text'];

export interface FileEdit {
  readonly path: string;
  readonly content: string;
}

const getFirstString = (args: Record<string, unknown>, keys: readonly string[]) =>
  keys.map((key) => args[key]).find((value): value is string => typeof value === 'string');

/** Any tool whose arguments carry a file path and file content counts as a file edit. */
export const getFileEdit = (args: unknown): FileEdit | undefined => {
  if (!isRecord(args)) return undefined;
  const path = getFirstString(args, FILE_PATH_KEYS)?.trim();
  const content = getFirstString(args, FILE_CONTENT_KEYS);
  return path && content !== undefined ? { path, content } : undefined;
};

export const getFileName = (path: string): string => path.split(/[\\/]/).at(-1) || path;

const LANGUAGE_BY_EXTENSION: Record<string, CodeSnippetLanguage> = {
  bash: 'bash',
  css: 'css',
  go: 'go',
  html: 'html',
  js: 'javascript',
  json: 'json',
  jsonl: 'json',
  jsx: 'jsx',
  md: 'markdown',
  py: 'python',
  rs: 'rust',
  sh: 'shell',
  ts: 'typescript',
  tsx: 'tsx',
  yaml: 'yaml',
  yml: 'yaml',
};

export const getFileLanguage = (path: string): CodeSnippetLanguage =>
  LANGUAGE_BY_EXTENSION[getFileName(path).split('.').at(-1)?.toLowerCase() ?? ''] ?? 'text';

/** Pretty-prints a tool payload: JSON values and JSON strings as JSON, anything else as text. */
export const formatToolPayload = (
  value: unknown
): { readonly language: CodeSnippetLanguage; readonly text: string } => {
  if (typeof value === 'string') {
    try {
      return { language: 'json', text: JSON.stringify(JSON.parse(value), null, 2) };
    } catch {
      return { language: 'text', text: value };
    }
  }
  return { language: 'json', text: JSON.stringify(value, null, 2) ?? String(value) };
};
