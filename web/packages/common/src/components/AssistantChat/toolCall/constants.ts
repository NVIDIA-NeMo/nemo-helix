// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NvidiaGuiIconProps } from '@nemo/icons';
import {
  HelpCircle,
  IconClipboard,
  Rename,
  IconDocument,
  World,
  ChartTree,
  MagnifyingGlass,
  WindowTerminal,
  ListCheckmark,
} from '@nemo/icons/icons';
import type { FC } from 'react';

export const TOOL_LABELS: Record<string, string> = {
  Bash: 'Run command',
  Edit: 'Edit file',
  Glob: 'Find files',
  Grep: 'MagnifyingGlass text',
  LS: 'List directory',
  MultiEdit: 'Edit file',
  Read: 'Read file',
  TodoWrite: 'Update todos',
  WebFetch: 'Fetch URL',
  WebSearch: 'MagnifyingGlass web',
  Write: 'Write file',
};

export const TOOL_ICONS: Record<string, FC<Omit<NvidiaGuiIconProps, 'iconName'>>> = {
  Bash: WindowTerminal,
  Edit: Rename,
  Glob: MagnifyingGlass,
  Grep: MagnifyingGlass,
  LS: ChartTree,
  MultiEdit: Rename,
  Read: IconDocument,
  TodoWrite: ListCheckmark,
  WebFetch: World,
  WebSearch: MagnifyingGlass,
  Write: Rename,
};

export const CODE_BLOCK_SURFACE_CLASS = 'bg-gray-050 dark:bg-gray-900';
export const FILE_CHANGE_ADDITION_CLASS = 'text-feedback-success';
export const FILE_CHANGE_DELETION_CLASS = 'text-feedback-danger';
export const SUBTLE_MESSAGE_MAX_LENGTH = 160;

export const SUBTLE_TOOL_ICONS: Record<string, FC<Omit<NvidiaGuiIconProps, 'iconName'>>> = {
  AskUserQuestion: HelpCircle,
  Bash: WindowTerminal,
  FindFiles: MagnifyingGlass,
  Grep: MagnifyingGlass,
  Read: IconDocument,
  TaskCreate: IconClipboard,
  TaskUpdate: IconClipboard,
  ToolSearch: MagnifyingGlass,
};
