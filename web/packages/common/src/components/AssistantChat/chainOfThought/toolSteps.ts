// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type FileChangeKind,
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
import type { CodeSnippetLanguage } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

export type ToolStepIcon = FC<Omit<NvidiaGuiIconProps, 'iconName'>>;
type ToolArgs = Record<string, unknown>;

/** How a tool call reads in the work group: its icon and the action it describes. */
export interface ToolStepDefinition {
  /** The exact tool names this covers, e.g. `Edit` and `MultiEdit`. */
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
  /** Title for several calls folded into one step, with `{count}`: "Read {count} files". */
  readonly repeated?: string;
  /** Shows the arguments as something other than JSON, e.g. a Bash command as a shell snippet. */
  readonly formatArgs?: (args: ToolArgs) => ToolStepArgsView | undefined;
  /** Renders as a file change with line counts: a whole-file write, or an edit with replacements. */
  readonly fileChange?: FileChangeKind;
}

export interface ToolStepArgsView {
  readonly label: string;
  readonly language: CodeSnippetLanguage;
  readonly text: string;
}

const firstString =
  (...keys: readonly string[]) =>
  (args: ToolArgs): string | undefined =>
    keys
      .map((key) => args[key])
      .find((value): value is string => typeof value === 'string' && value.trim() !== '');

const fileName = (args: ToolArgs) => {
  const path = firstString('file_path', 'path')(args);
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

const patternInPath = (args: ToolArgs) => {
  const pattern = firstString('pattern', 'query')(args);
  const path = firstString('path')(args);
  return pattern && path ? `${pattern} in ${path}` : pattern;
};

const commandSnippet = (args: ToolArgs): ToolStepArgsView | undefined => {
  const command = firstString('command', 'cmd')(args);
  return command ? { label: 'Command', language: 'bash', text: command } : undefined;
};

export const TOOL_STEP_DEFINITIONS: readonly ToolStepDefinition[] = [
  {
    names: ['Bash'],
    icon: WindowTerminal,
    running: 'Running command',
    done: 'Ran command',
    repeated: 'Ran {count} commands',
    subject: firstString('command', 'cmd'),
    subjectIsCode: true,
    formatArgs: commandSnippet,
  },
  {
    names: ['Read'],
    icon: IconDocument,
    running: 'Reading file',
    done: 'Read file',
    repeated: 'Read {count} files',
    subject: fileName,
    subjectIsCode: true,
  },
  {
    names: ['Write'],
    icon: Pencil,
    running: 'Writing file',
    done: 'Wrote file',
    repeated: 'Wrote {count} files',
    subject: fileName,
    subjectIsCode: true,
    fileChange: 'write',
  },
  {
    names: ['Edit', 'MultiEdit'],
    icon: Pencil,
    running: 'Editing file',
    done: 'Edited file',
    repeated: 'Edited {count} files',
    subject: fileName,
    subjectIsCode: true,
    fileChange: 'edit',
  },
  {
    names: ['Glob', 'FindFiles'],
    icon: MagnifyingGlass,
    running: 'Finding files',
    done: 'Found files',
    repeated: 'Found files {count} times',
    subject: firstString('pattern', 'query', 'path'),
    subjectIsCode: true,
  },
  {
    names: ['Grep'],
    icon: MagnifyingGlass,
    running: 'Searching text',
    done: 'Searched text',
    repeated: 'Searched text {count} times',
    subject: patternInPath,
    subjectIsCode: true,
  },
  {
    names: ['LS'],
    icon: FolderOpen,
    running: 'Listing directory',
    done: 'Listed directory',
    repeated: 'Listed {count} directories',
    subject: firstString('path', 'directory'),
    subjectIsCode: true,
  },
  {
    names: ['WebSearch'],
    icon: World,
    running: 'Searching web',
    done: 'Searched web',
    repeated: 'Searched web {count} times',
    subject: firstString('query', 'q'),
  },
  {
    names: ['WebFetch'],
    icon: Link,
    running: 'Fetching URL',
    done: 'Fetched URL',
    repeated: 'Fetched {count} URLs',
    subject: firstString('url'),
    subjectIsCode: true,
  },
  {
    names: ['TodoWrite'],
    icon: ListCheckmark,
    running: 'Updating todos',
    done: 'Updated todos',
    repeated: 'Updated todos {count} times',
    subject: itemCount('todos', 'item'),
  },
  {
    names: ['TaskCreate'],
    icon: IconClipboard,
    running: 'Creating task',
    done: 'Created task',
    repeated: 'Created {count} tasks',
    subject: firstString('description', 'task', 'prompt'),
  },
  {
    names: ['TaskUpdate'],
    icon: IconClipboard,
    running: 'Updating task',
    done: 'Updated task',
    repeated: 'Updated {count} tasks',
    subject: firstString('description', 'task', 'subject'),
  },
  {
    names: ['AskUserQuestion'],
    icon: HelpCircle,
    running: 'Asking user',
    done: 'Asked user',
    repeated: 'Asked {count} questions',
    subject: firstQuestion,
  },
  {
    names: ['ToolSearch'],
    icon: Toolbox,
    running: 'Searching tools',
    done: 'Searched tools',
    repeated: 'Searched tools {count} times',
    subject: firstString('query', 'pattern', 'name'),
  },
  {
    names: ['Skill'],
    icon: Puzzle,
    running: 'Using skill',
    done: 'Used skill',
    repeated: 'Used {count} skills',
    subject: firstString('skill', 'name', 'command'),
    subjectIsCode: true,
  },
];

export const getFallbackToolStep = (toolName: string): ToolStepDefinition => ({
  names: [toolName],
  icon: WindowCode,
  running: `Calling ${toolName}`,
  done: `Called ${toolName}`,
  repeated: `Called ${toolName} {count} times`,
});

const DEFINITIONS_BY_NAME = new Map(
  TOOL_STEP_DEFINITIONS.flatMap((definition) =>
    definition.names.map((name) => [name, definition] as const)
  )
);

export const findToolStepDefinition = (toolName: string): ToolStepDefinition | undefined =>
  DEFINITIONS_BY_NAME.get(toolName);

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

/** Title for `count` calls of the same tool folded into one step, e.g. "Read 3 files". */
export const getRepeatedToolStepTitle = (definition: ToolStepDefinition, count: number): string =>
  (definition.repeated ?? `${definition.done} {count} times`).replace('{count}', String(count));
