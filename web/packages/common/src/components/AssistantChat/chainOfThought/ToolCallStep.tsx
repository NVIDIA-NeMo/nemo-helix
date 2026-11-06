// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WorkCodeSnippet } from '@nemo/common/src/components/AssistantChat/chainOfThought/WorkCodeSnippet';
import {
  formatToolPayload,
  getFileLanguage,
  getFileName,
} from '@nemo/common/src/components/AssistantChat/chainOfThought/workParts';
import {
  WORK_ICON_SIZE,
  WorkStep,
} from '@nemo/common/src/components/AssistantChat/chainOfThought/WorkStep';
import {
  TOOL_LABELS,
  TOOL_RUNNING_ACTIONS,
} from '@nemo/common/src/components/AssistantChat/toolCall/constants';
import {
  getFileChangeSummary,
  getSubtleToolDetail,
  getSubtleToolIcon,
  getSubtleToolMessage,
  getToolInvocation,
} from '@nemo/common/src/components/AssistantChat/toolCall/helpers';
import type { FileChangeSummary } from '@nemo/common/src/components/AssistantChat/toolCall/types';
import {
  type AssistantToolArgs,
  toAssistantToolArgs,
} from '@nemo/common/src/components/AssistantChat/toolParts';
import { DocumentNew, Pencil } from '@nemo/icons/icons';
import { Stack } from '@nvidia/foundations-react-core';
import cn from 'classnames';

// The detail sits one step dimmer than the action and brightens with it on hover and when the
// step is open. `leading-none` keeps the code font from growing the row past the title's height.
const DETAIL_CLASS =
  'leading-none text-subtle [summary:hover_&]:text-secondary [details[open]>summary_&]:text-secondary';
const CODE_DETAIL_CLASS = 'font-mono text-[13px]';

// Tools whose detail is a command, path, pattern, or URL rather than prose.
const CODE_DETAIL_TOOLS = new Set([
  'Bash',
  'Edit',
  'FindFiles',
  'Glob',
  'Grep',
  'LS',
  'MultiEdit',
  'Read',
  'WebFetch',
  'Write',
]);

export interface ToolCallStepProps {
  readonly args?: unknown;
  readonly argsText?: string;
  readonly defaultOpen?: boolean;
  readonly isError?: boolean;
  readonly result?: unknown;
  readonly running?: boolean;
  readonly toolName: string;
}

const PayloadSection = ({
  label,
  tone,
  value,
}: {
  readonly label: string;
  readonly tone?: 'default' | 'danger';
  readonly value: unknown;
}) => {
  const { language, text } = formatToolPayload(value);
  return <WorkCodeSnippet label={label} language={language} tone={tone} value={text} />;
};

/** Lines added and removed by a file change, "+12 −3". */
const FileChangeStats = ({
  additions,
  deletions,
}: Pick<FileChangeSummary, 'additions' | 'deletions'>) => (
  <span className="ml-1 text-[13px] tabular-nums" data-testid="assistant-work-edit-stats">
    <span className="text-feedback-success">+{additions}</span>{' '}
    <span className="text-feedback-danger">−{deletions}</span>
  </span>
);

/**
 * Splits a tool message such as "Read README.md" into its action and the call's detail, using
 * the same helpers the NeMo Assistant's tool rows used, so the detail can be styled on its own.
 */
const getTitleParts = (
  toolName: string,
  args: AssistantToolArgs,
  running: boolean | undefined,
  fileChange: FileChangeSummary | undefined
): { readonly action: string; readonly detail?: string } => {
  if (fileChange) {
    return {
      action: running ? (TOOL_RUNNING_ACTIONS[toolName] ?? 'Editing') : fileChange.action,
      detail: getFileName(fileChange.path),
    };
  }
  const message =
    getSubtleToolMessage(toolName, args) ?? `Used ${TOOL_LABELS[toolName] ?? toolName}`;
  const detail = getSubtleToolDetail(toolName, args, message);
  const hasDetail = detail !== message && message.endsWith(detail);
  const action = hasDetail ? message.slice(0, -detail.length).trim() : message;
  return {
    action: running
      ? (TOOL_RUNNING_ACTIONS[toolName] ?? `Using ${TOOL_LABELS[toolName] ?? toolName}`)
      : action,
    detail: hasDetail ? detail : undefined,
  };
};

/**
 * A tool call: the message the NeMo Assistant's tool rows used ("Ran pwd", "Read README.md",
 * "Used get_weather") with the per-tool icon, then the arguments and result (or error) when
 * expanded. A successful `Write`/`Edit`/`MultiEdit` shows as a file change with its line counts
 * and the written content (or the edit's arguments); a failed one keeps the arguments and error.
 */
export const ToolCallStep = ({
  args,
  argsText,
  defaultOpen,
  isError,
  result,
  running,
  toolName,
}: ToolCallStepProps) => {
  const toolArgs = toAssistantToolArgs(args);
  const fileChange = isError ? undefined : getFileChangeSummary(toolName, toolArgs, argsText ?? '');
  const { action, detail } = getTitleParts(toolName, toolArgs, running, fileChange);
  const Icon = fileChange
    ? fileChange.action === 'Wrote'
      ? DocumentNew
      : Pencil
    : getSubtleToolIcon(toolName);
  const hasArgs = Object.keys(toolArgs).length > 0;
  const command =
    toolName === 'Bash' && hasArgs ? getToolInvocation(toolName, toolArgs) : undefined;

  return (
    <WorkStep
      data-testid={fileChange ? 'assistant-work-file-edit' : 'assistant-work-tool-call'}
      defaultOpen={defaultOpen}
      icon={<Icon size={WORK_ICON_SIZE} />}
      title={
        <>
          {action}
          {detail ? (
            <>
              {' '}
              <span
                className={cn(DETAIL_CLASS, CODE_DETAIL_TOOLS.has(toolName) && CODE_DETAIL_CLASS)}
              >
                {detail}
              </span>
            </>
          ) : null}
          {fileChange ? (
            <FileChangeStats additions={fileChange.additions} deletions={fileChange.deletions} />
          ) : null}
        </>
      }
    >
      {fileChange ? (
        <WorkCodeSnippet
          label={fileChange.path}
          language={fileChange.action === 'Wrote' ? getFileLanguage(fileChange.path) : 'json'}
          value={fileChange.reviewContent}
        />
      ) : (
        <Stack gap="density-md">
          {command ? (
            <WorkCodeSnippet label="Command" language="bash" value={command} />
          ) : (
            <PayloadSection label="Arguments" value={hasArgs ? toolArgs : argsText || '{}'} />
          )}
          {result !== undefined ? (
            <PayloadSection
              label={isError ? 'Error' : 'Result'}
              tone={isError ? 'danger' : 'default'}
              value={result}
            />
          ) : null}
        </Stack>
      )}
    </WorkStep>
  );
};
