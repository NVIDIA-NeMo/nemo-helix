// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NvidiaGuiIconProps } from '@nemo/icons';
import {
  HelpCircle,
  IconClipboard,
  Pencil,
  IconDocument,
  World,
  ChartTree,
  MagnifyingGlass,
  WindowTerminal,
  CheckCircle,
  IconKeyboard,
} from '@nemo/icons/icons';
import type { FC } from 'react';

export const TOOL_LABELS: Record<string, string> = {
  Bash: 'Run command',
  check_status: 'Check job status',
  deploy_guardrail: 'Deploy guardrail',
  Edit: 'Edit file',
  Glob: 'Find files',
  Grep: 'Search text',
  LS: 'List directory',
  MultiEdit: 'Edit file',
  nemo_api: 'Platform API',
  Read: 'Read file',
  search_docs: 'Search docs',
  TodoWrite: 'Update todos',
  WebFetch: 'Fetch URL',
  WebSearch: 'Search web',
  Write: 'Write file',
};

/** Other names that model and MCP tools use for the built-in tools above. */
export const TOOL_NAME_ALIASES: Record<string, readonly string[]> = {
  AskUserQuestion: ['AskUser'],
  Bash: ['ExecuteCommand', 'RunCommand', 'Shell'],
  Edit: ['EditFile'],
  Grep: ['SearchText'],
  LS: ['ListDir', 'ListDirectory', 'ListFiles'],
  Read: ['ReadFile'],
  WebFetch: ['FetchUrl'],
  WebSearch: ['SearchWeb'],
  Write: ['CreateFile', 'WriteFile'],
};

export const TOOL_ICONS: Record<string, FC<Omit<NvidiaGuiIconProps, 'iconName'>>> = {
  Bash: WindowTerminal,
  check_status: CheckCircle,
  Edit: Pencil,
  Glob: MagnifyingGlass,
  Grep: MagnifyingGlass,
  LS: ChartTree,
  MultiEdit: Pencil,
  Read: IconDocument,
  search_docs: MagnifyingGlass,
  TodoWrite: CheckCircle,
  WebFetch: World,
  WebSearch: MagnifyingGlass,
  Write: Pencil,
};

export const TOOL_RUNNING_ACTIONS: Record<string, string> = {
  AskUserQuestion: 'Asking',
  Bash: 'Running',
  check_status: 'Checking job status',
  deploy_guardrail: 'Deploying guardrail',
  Edit: 'Editing',
  FindFiles: 'Searching files',
  Glob: 'Finding files',
  Grep: 'Searching text',
  LS: 'Listing directory',
  MultiEdit: 'Editing',
  nemo_api: 'Calling platform API',
  Read: 'Reading',
  search_docs: 'Searching docs',
  TaskCreate: 'Creating task',
  TaskUpdate: 'Updating task',
  TodoWrite: 'Updating todos',
  ToolSearch: 'Searching tools',
  WebFetch: 'Fetching URL',
  WebSearch: 'Searching web',
  Write: 'Writing',
};

export const SUBTLE_MESSAGE_MAX_LENGTH = 160;

export const SUBTLE_TOOL_ICONS: Record<string, FC<Omit<NvidiaGuiIconProps, 'iconName'>>> = {
  AskUserQuestion: HelpCircle,
  Bash: IconKeyboard,
  FindFiles: MagnifyingGlass,
  Grep: MagnifyingGlass,
  Read: IconDocument,
  TaskCreate: IconClipboard,
  TaskUpdate: IconClipboard,
  ToolSearch: MagnifyingGlass,
};
