// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  TOOL_LABELS,
  TOOL_RUNNING_ACTIONS,
} from '@nemo/common/src/components/AssistantChat/toolCall/constants';
import {
  getBashCommand,
  getFileChangeSummary,
  getFileName,
  getSubtleToolDetail,
  getSubtleToolIcon,
  getSubtleToolMessage,
} from '@nemo/common/src/components/AssistantChat/toolCall/helpers';
import type { FileChangeSummary } from '@nemo/common/src/components/AssistantChat/toolCall/types';
import {
  type AssistantToolArgs,
  toAssistantToolArgs,
} from '@nemo/common/src/components/AssistantChat/toolParts';
import { WorkCodeSnippet } from '@nemo/common/src/components/AssistantChat/work/WorkCodeSnippet';
import { formatToolPayload } from '@nemo/common/src/components/AssistantChat/work/workParts';
import { WORK_ICON_SIZE, WorkStep } from '@nemo/common/src/components/AssistantChat/work/WorkStep';
import { getLanguageFromFilePath } from '@nemo/common/src/utils/codeSnippet';
import { DocumentNew, Pencil } from '@nemo/icons/icons';
import { Stack, Text } from '@nvidia/foundations-react-core';

const DETAIL_CLASS =
  'text-subtle group-hover/step:text-secondary group-data-[state=open]/step:text-secondary';

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

interface ToolCallStepProps {
  readonly args?: unknown;
  readonly argsText?: string;
  readonly defaultOpen?: boolean;
  readonly isError?: boolean;
  readonly result?: unknown;
  readonly running?: boolean;
  readonly toolName: string;
}

interface TitleParts {
  readonly action: string;
  readonly detail?: string;
}

const getRunningAction = (toolName: string, fallback: string): string =>
  TOOL_RUNNING_ACTIONS[toolName] ?? fallback;

/** Splits a tool message such as "Read README.md" into its action and the call's detail. */
const getTitleParts = (
  toolName: string,
  args: AssistantToolArgs,
  running: boolean | undefined,
  fileChange: FileChangeSummary | undefined
): TitleParts => {
  if (fileChange) {
    return {
      action: running ? getRunningAction(toolName, 'Editing') : fileChange.action,
      detail: getFileName(fileChange.path),
    };
  }
  const label = TOOL_LABELS[toolName] ?? toolName;
  const message = getSubtleToolMessage(toolName, args) ?? `Used ${label}`;
  const detail = getSubtleToolDetail(toolName, args, message);
  const hasDetail = detail !== message && message.endsWith(detail);
  const action = hasDetail ? message.slice(0, -detail.length).trim() : message;
  return {
    action: running ? getRunningAction(toolName, `Using ${label}`) : action,
    detail: detail !== message ? detail : undefined,
  };
};

const getStepIcon = (toolName: string, fileChange: FileChangeSummary | undefined) => {
  if (!fileChange) {
    return getSubtleToolIcon(toolName);
  }
  return fileChange.action === 'Wrote' ? DocumentNew : Pencil;
};

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
  const Icon = getStepIcon(toolName, fileChange);
  const hasArgs = Object.keys(toolArgs).length > 0;
  const command = toolName === 'Bash' ? getBashCommand(toolArgs) : undefined;
  const payload = formatToolPayload(hasArgs ? toolArgs : argsText || '{}');
  const resultPayload = result === undefined ? undefined : formatToolPayload(result);

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
              {CODE_DETAIL_TOOLS.has(toolName) ? (
                <Text asChild kind="mono/sm" className={DETAIL_CLASS}>
                  <span>{detail}</span>
                </Text>
              ) : (
                <span className={DETAIL_CLASS}>{detail}</span>
              )}
            </>
          ) : null}
          {fileChange ? (
            <span className="ml-density-xs tabular-nums" data-testid="assistant-work-edit-stats">
              <span className="text-feedback-success">+{fileChange.additions}</span>{' '}
              <span className="text-feedback-danger">−{fileChange.deletions}</span>
            </span>
          ) : null}
        </>
      }
    >
      {fileChange ? (
        <WorkCodeSnippet
          label={fileChange.path}
          language={
            fileChange.action === 'Wrote'
              ? (getLanguageFromFilePath(fileChange.path) ?? 'text')
              : 'json'
          }
          value={fileChange.reviewContent}
        />
      ) : (
        <Stack gap="density-md">
          {command ? (
            <WorkCodeSnippet label="Command" language="bash" value={command} />
          ) : (
            <WorkCodeSnippet label="Arguments" language={payload.language} value={payload.text} />
          )}
          {resultPayload ? (
            <WorkCodeSnippet
              label={isError ? 'Error' : 'Result'}
              labelClassName={isError ? 'text-feedback-danger' : undefined}
              language={resultPayload.language}
              value={resultPayload.text}
            />
          ) : null}
        </Stack>
      )}
    </WorkStep>
  );
};
