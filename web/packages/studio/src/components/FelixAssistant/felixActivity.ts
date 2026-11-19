// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ThreadMessageLike } from '@assistant-ui/react';
import { ASSISTANT_SUBTLE_TOOL_GROUP_NAME } from '@nemo/common/src/components/AssistantChat/toolParts';

/**
 * What kind of work the assistant is doing right now, as far as Felix's poses are concerned:
 * `building` (creating or changing things) and `docs` (reading, searching, looking things up).
 */
export type FelixActivity = 'building' | 'docs';

/**
 * nemo_api actions that only look things up. Mirrors `_READ_ONLY_ACTIONS` in
 * agents/nemo-studio-assistant/src/nemo_studio_assistant/register.py, which is what decides whether
 * the backend asks for approval; keep the two in sync.
 */
const READ_ONLY_API_ACTIONS: ReadonlySet<string> = new Set([
  'check',
  'get',
  'get_logs',
  'get_status',
  'list',
  'read',
  'retrieve',
  'search',
]);

const NEMO_API_TOOL = 'nemoapi';

// Tool names are compared lower-cased with punctuation removed, so `Read`, `read_file` and
// `read-file` all match. The agent can report either Claude-style names (Bash, Edit, Grep) or
// Deep Agents-style names (execute, edit_file, grep), and MCP tools arrive as
// `mcp__<server>__<tool>`.
const BUILDING_TOOLS: ReadonlySet<string> = new Set([
  'bash',
  'deployguardrail',
  'edit',
  'editfile',
  'execute',
  'multiedit',
  'notebookedit',
  'shell',
  'write',
  'writefile',
]);

const DOCS_TOOLS: ReadonlySet<string> = new Set([
  'checkstatus',
  'findfiles',
  'glob',
  'grep',
  'ls',
  'read',
  'readfile',
  // The NeMo Studio Assistant's own documentation search (its MCP tool, not a harness built-in).
  'searchdocs',
  'skill',
  'toolsearch',
  'webfetch',
  'websearch',
]);

const normalizeToolName = (toolName: string): string =>
  toolName
    .replace(/^mcp__.+?__/, '')
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '');

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** Classifies a single tool call. Anything that isn't clearly one or the other gets no pose. */
export const getFelixActivityForTool = (
  toolName: string,
  args?: Record<string, unknown>
): FelixActivity | null => {
  const name = normalizeToolName(toolName);

  if (name === NEMO_API_TOOL) {
    const action = typeof args?.action === 'string' ? args.action.trim().toLowerCase() : '';
    if (!action) return null;
    return READ_ONLY_API_ACTIONS.has(action) ? 'docs' : 'building';
  }
  if (BUILDING_TOOLS.has(name)) return 'building';
  if (DOCS_TOOLS.has(name)) return 'docs';
  return null;
};

interface ToolCallLike {
  readonly toolName: string;
  readonly args?: Record<string, unknown>;
  readonly result?: unknown;
}

const isToolCallPart = (part: unknown): part is ToolCallLike & { type: 'tool-call' } =>
  isRecord(part) && part.type === 'tool-call' && typeof part.toolName === 'string';

const isBlankTextPart = (part: unknown): boolean =>
  isRecord(part) && part.type === 'text' && typeof part.text === 'string' && !part.text.trim();

/** A run of consecutive tool calls may be folded into one "subtle group" part; use its last call. */
const getLastCallInGroup = (part: ToolCallLike): ToolCallLike => {
  if (part.toolName !== ASSISTANT_SUBTLE_TOOL_GROUP_NAME) return part;

  const actions = isRecord(part.args) && Array.isArray(part.args.actions) ? part.args.actions : [];
  const last = actions.at(-1);
  return isRecord(last) && typeof last.toolName === 'string'
    ? {
        toolName: last.toolName,
        args: isRecord(last.args) ? last.args : undefined,
      }
    : part;
};

/**
 * Works out what the assistant is doing from the live thread.
 *
 * Only a tool call that is still *in flight* counts: it must be the last real part of the latest
 * assistant reply (text or reasoning after it means the tool is done and the model has moved on)
 * and must not have a result yet. Everything else returns null, which leaves Felix in his normal
 * pose.
 */
export const getFelixActivity = (messages: readonly ThreadMessageLike[]): FelixActivity | null => {
  const lastMessage = messages.at(-1);
  if (!lastMessage || lastMessage.role !== 'assistant' || typeof lastMessage.content === 'string') {
    return null;
  }

  const lastPart = lastMessage.content.findLast((part) => !isBlankTextPart(part));
  if (!isToolCallPart(lastPart) || lastPart.result !== undefined) return null;

  const call = getLastCallInGroup(lastPart);
  return getFelixActivityForTool(call.toolName, call.args);
};

/**
 * Whether the latest reply finished properly. A reply that failed (`incomplete` with an error) or
 * was stopped by the user (`incomplete`, cancelled) is not a success, and neither is one that
 * hasn't finished yet, so Felix only celebrates real completions.
 */
export const didLastReplySucceed = (messages: readonly ThreadMessageLike[]): boolean => {
  const lastMessage = messages.at(-1);
  return lastMessage?.role === 'assistant' && lastMessage.status?.type === 'complete';
};
