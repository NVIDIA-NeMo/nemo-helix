// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  CheckSquare,
  CircleHelp,
  ClipboardList,
  Command,
  FilePenLine,
  FileText,
  Globe,
  ListTree,
  Search,
  Terminal,
  type LucideIcon,
} from 'lucide-react';

export const TOOL_LABELS: Record<string, string> = {
  Bash: 'Run command',
  Edit: 'Edit file',
  Glob: 'Find files',
  Grep: 'Search text',
  LS: 'List directory',
  MultiEdit: 'Edit file',
  Read: 'Read file',
  TodoWrite: 'Update todos',
  WebFetch: 'Fetch URL',
  WebSearch: 'Search web',
  Write: 'Write file',
};

export const TOOL_ICONS: Record<string, LucideIcon> = {
  Bash: Terminal,
  Edit: FilePenLine,
  Glob: Search,
  Grep: Search,
  LS: ListTree,
  MultiEdit: FilePenLine,
  Read: FileText,
  TodoWrite: CheckSquare,
  WebFetch: Globe,
  WebSearch: Search,
  Write: FilePenLine,
};

export const CODE_BLOCK_SURFACE_CLASS = 'bg-gray-050 dark:bg-gray-900';
export const FILE_CHANGE_ADDITION_CLASS = 'text-feedback-success';
export const FILE_CHANGE_DELETION_CLASS = 'text-feedback-danger';
export const SUBTLE_MESSAGE_MAX_LENGTH = 160;

export const SUBTLE_TOOL_ICONS: Record<string, LucideIcon> = {
  AskUserQuestion: CircleHelp,
  Bash: Command,
  FindFiles: Search,
  Grep: Search,
  Read: FileText,
  TaskCreate: ClipboardList,
  TaskUpdate: ClipboardList,
  ToolSearch: Search,
};
