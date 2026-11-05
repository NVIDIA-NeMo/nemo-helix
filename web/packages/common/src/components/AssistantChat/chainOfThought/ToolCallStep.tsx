// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  FILE_EDIT_STEP,
  findToolStepDefinition,
  getFallbackToolStep,
  getToolStepTitleParts,
} from '@nemo/common/src/components/AssistantChat/chainOfThought/toolSteps';
import { WorkCodeSnippet } from '@nemo/common/src/components/AssistantChat/chainOfThought/WorkCodeSnippet';
import {
  formatToolPayload,
  getFileEdit,
  getFileLanguage,
  isRecord,
} from '@nemo/common/src/components/AssistantChat/chainOfThought/workParts';
import {
  WORK_ICON_SIZE,
  WorkStep,
} from '@nemo/common/src/components/AssistantChat/chainOfThought/WorkStep';
import { Stack } from '@nvidia/foundations-react-core';
import cn from 'classnames';

// The subject sits one step dimmer than the action and brightens with it on hover and when the
// step is open. `leading-none` keeps the code font from growing the row past the title's height.
const SUBJECT_CLASS =
  'leading-none text-subtle [summary:hover_&]:text-secondary [details[open]>summary_&]:text-secondary';
const CODE_SUBJECT_CLASS = 'font-mono text-[13px]';

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

/**
 * A tool call, with the icon and action text from its `TOOL_STEP_DEFINITIONS` entry, or
 * "Called {toolName}" for tools without one. Expanding it shows the arguments and the result.
 * Successful calls whose arguments carry a file path and content show the file instead; a failed
 * one keeps the arguments and error so the failure is visible.
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
  const fileEdit = isError ? undefined : getFileEdit(args);
  const step =
    findToolStepDefinition(toolName) ?? (fileEdit ? FILE_EDIT_STEP : getFallbackToolStep(toolName));
  const { icon: Icon } = step;
  const hasArgs = isRecord(args) ? Object.keys(args).length > 0 : args !== undefined;
  const { action, subject } = getToolStepTitleParts(step, {
    args: isRecord(args) ? args : {},
    running,
  });

  return (
    <WorkStep
      data-testid={fileEdit ? 'assistant-work-file-edit' : 'assistant-work-tool-call'}
      defaultOpen={defaultOpen}
      icon={<Icon size={WORK_ICON_SIZE} />}
      title={
        subject ? (
          <>
            {action}{' '}
            <span className={cn(SUBJECT_CLASS, step.subjectIsCode && CODE_SUBJECT_CLASS)}>
              {subject}
            </span>
          </>
        ) : (
          action
        )
      }
    >
      {fileEdit ? (
        <WorkCodeSnippet
          label={fileEdit.path}
          language={getFileLanguage(fileEdit.path)}
          value={fileEdit.content}
        />
      ) : (
        <Stack gap="density-md">
          <PayloadSection label="Arguments" value={hasArgs ? args : argsText || '{}'} />
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
