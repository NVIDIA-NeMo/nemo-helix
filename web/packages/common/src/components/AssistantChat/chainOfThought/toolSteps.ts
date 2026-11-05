// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  FILE_PATH_KEYS,
  getFileName,
  isRecord,
} from '@nemo/common/src/components/AssistantChat/chainOfThought/workParts';
import type { NvidiaGuiIconProps } from '@nemo/icons';
import {
  FolderOpen,
  HelpCircle,
  IconClipboard,
  IconDocument,
  Link,
  ListCheckmark,
  MagnifyingGlass,
  Pencil,
  Puzzle,
  Toolbox,
  WindowCode,
  WindowTerminal,
  World,
} from '@nemo/icons/icons';
import type { FC } from 'react';

export type ToolStepIcon = FC<Omit<NvidiaGuiIconProps, 'iconName'>>;
type ToolArgs = Record<string, unknown>;

/** How a tool call reads in the work group: its icon and the action it describes. */
export interface ToolStepDefinition {
  /** Tool names this covers. Matching ignores case, `_`, `-`, and an `mcp__<server>__` prefix. */
  readonly names: readonly string[];
  readonly icon: ToolStepIcon;
  /** Action text while the call runs, e.g. "Running command". */
  readonly running: string;
  /** Action text once the call finishes, e.g. "Ran command". */
  readonly done: string;
  /** Detail shown after the action, such as the command, file name, or query. */
  readonly subject?: (args: ToolArgs) => string | undefined;
  /** Show the subject in the code font: commands, paths, patterns, and URLs rather than prose. */
  readonly subjectIsCode?: boolean;
}

const firstString =
  (...keys: readonly string[]) =>
  (args: ToolArgs): string | undefined =>
    keys
      .map((key) => args[key])
      .find((value): value is string => typeof value === 'string' && value.trim() !== '');

const fileName = (args: ToolArgs) => {
  const path = firstString(...FILE_PATH_KEYS)(args);
  return path && getFileName(path.trim());
};

const firstQuestion = (args: ToolArgs) => {
  const questions = Array.isArray(args.questions) ? args.questions.filter(isRecord) : [];
  return (
    firstString('question', 'prompt')(args) ?? firstString('question', 'header')(questions[0] ?? {})
  );
};

const itemCount = (key: string, noun: string) => (args: ToolArgs) => {
  const items = args[key];
  return Array.isArray(items)
    ? `${items.length} ${noun}${items.length === 1 ? '' : 's'}`
    : undefined;
};

export const TOOL_STEP_DEFINITIONS: readonly ToolStepDefinition[] = [
  {
    names: ['Bash', 'Shell', 'RunCommand', 'ExecuteCommand'],
    icon: WindowTerminal,
    running: 'Running command',
    done: 'Ran command',
    subject: firstString('command', 'cmd'),
    subjectIsCode: true,
  },
  {
    names: ['Read', 'ReadFile'],
    icon: IconDocument,
    running: 'Reading file',
    done: 'Read file',
    subject: fileName,
    subjectIsCode: true,
  },
  {
    names: ['Write', 'WriteFile', 'CreateFile'],
    icon: Pencil,
    running: 'Writing file',
    done: 'Wrote file',
    subject: fileName,
    subjectIsCode: true,
  },
  {
    names: ['Edit', 'MultiEdit', 'EditFile'],
    icon: Pencil,
    running: 'Editing file',
    done: 'Edited file',
    subject: fileName,
    subjectIsCode: true,
  },
  {
    names: ['Glob', 'FindFiles'],
    icon: MagnifyingGlass,
    running: 'Finding files',
    done: 'Found files',
    subject: firstString('pattern', 'query', 'path'),
    subjectIsCode: true,
  },
  {
    names: ['Grep', 'SearchText'],
    icon: MagnifyingGlass,
    running: 'Searching text',
    done: 'Searched text',
    subject: firstString('pattern', 'query'),
    subjectIsCode: true,
  },
  {
    names: ['LS', 'ListDirectory', 'ListDir', 'ListFiles'],
    icon: FolderOpen,
    running: 'Listing directory',
    done: 'Listed directory',
    subject: firstString('path', 'directory'),
    subjectIsCode: true,
  },
  {
    names: ['WebSearch', 'SearchWeb'],
    icon: World,
    running: 'Searching web',
    done: 'Searched web',
    subject: firstString('query', 'q'),
  },
  {
    names: ['WebFetch', 'FetchUrl'],
    icon: Link,
    running: 'Fetching URL',
    done: 'Fetched URL',
    subject: firstString('url'),
    subjectIsCode: true,
  },
  {
    names: ['TodoWrite'],
    icon: ListCheckmark,
    running: 'Updating todos',
    done: 'Updated todos',
    subject: itemCount('todos', 'item'),
  },
  {
    names: ['TaskCreate'],
    icon: IconClipboard,
    running: 'Creating task',
    done: 'Created task',
    subject: firstString('description', 'task', 'prompt'),
  },
  {
    names: ['TaskUpdate'],
    icon: IconClipboard,
    running: 'Updating task',
    done: 'Updated task',
    subject: firstString('description', 'task', 'subject'),
  },
  {
    names: ['AskUserQuestion', 'AskUser'],
    icon: HelpCircle,
    running: 'Asking user',
    done: 'Asked user',
    subject: firstQuestion,
  },
  {
    names: ['ToolSearch'],
    icon: Toolbox,
    running: 'Searching tools',
    done: 'Searched tools',
    subject: firstString('query', 'pattern', 'name'),
  },
  {
    names: ['Skill'],
    icon: Puzzle,
    running: 'Using skill',
    done: 'Used skill',
    subject: firstString('skill', 'name', 'command'),
    subjectIsCode: true,
  },
];

/** Steps for tools with file-edit arguments but no definition, e.g. a custom `save_note` tool. */
export const FILE_EDIT_STEP: ToolStepDefinition = {
  names: [],
  icon: Pencil,
  running: 'Editing',
  done: 'Edited',
  subject: fileName,
  subjectIsCode: true,
};

export const getFallbackToolStep = (toolName: string): ToolStepDefinition => ({
  names: [toolName],
  icon: WindowCode,
  running: `Calling ${toolName}`,
  done: `Called ${toolName}`,
});

const normalizeToolName = (name: string) =>
  name
    .replace(/^mcp__.+?__/, '')
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '');

const DEFINITIONS_BY_NAME = new Map(
  TOOL_STEP_DEFINITIONS.flatMap((definition) =>
    definition.names.map((name) => [normalizeToolName(name), definition] as const)
  )
);

export const findToolStepDefinition = (toolName: string): ToolStepDefinition | undefined =>
  DEFINITIONS_BY_NAME.get(normalizeToolName(toolName));

interface ToolStepTitleOptions {
  readonly args: ToolArgs;
  readonly running?: boolean;
}

/** The step's action and, when the arguments have one, its subject collapsed onto one line. */
export const getToolStepTitleParts = (
  definition: ToolStepDefinition,
  { args, running }: ToolStepTitleOptions
): { readonly action: string; readonly subject?: string } => ({
  action: running ? definition.running : definition.done,
  subject: definition.subject?.(args)?.replace(/\s+/g, ' ').trim() || undefined,
});

/** The step title as plain text, e.g. "Ran command ls -la". */
export const getToolStepTitle = (
  definition: ToolStepDefinition,
  options: ToolStepTitleOptions
): string => {
  const { action, subject } = getToolStepTitleParts(definition, options);
  return subject ? `${action} ${subject}` : action;
};
