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
  type FileEdit,
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

/** Lines added and removed by a file edit, "+12 −3". */
const EditStats = ({ additions, deletions }: Pick<FileEdit, 'additions' | 'deletions'>) => (
  <span className="ml-1 text-[13px] tabular-nums" data-testid="assistant-work-edit-stats">
    <span className="text-feedback-success">+{additions}</span>{' '}
    <span className="text-feedback-danger">−{deletions}</span>
  </span>
);

/** The written file, the replaced and replacement text, or (for a list of edits) the arguments. */
const FileEditContent = ({ args, edit }: { readonly args: unknown; readonly edit: FileEdit }) => {
  const language = getFileLanguage(edit.path);
  if (edit.content !== undefined) {
    return <WorkCodeSnippet label={edit.path} language={language} value={edit.content} />;
  }
  if (edit.after !== undefined) {
    return (
      <Stack gap="density-md">
        {edit.before !== undefined ? (
          <WorkCodeSnippet
            label={`${edit.path} · before`}
            language={language}
            value={edit.before}
          />
        ) : null}
        <WorkCodeSnippet label={`${edit.path} · after`} language={language} value={edit.after} />
      </Stack>
    );
  }
  return <PayloadSection label="Arguments" value={args} />;
};

/**
 * A tool call, with the icon and action text from its `TOOL_STEP_DEFINITIONS` entry, or
 * "Called {toolName}" for tools without one. Expanding it shows the arguments and the result.
 * Successful calls whose arguments carry a file path and content show the file change instead,
 * with its line counts in the title; a failed one keeps the arguments and error so the failure
 * is visible.
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
  const toolArgs = isRecord(args) ? args : {};
  const fileEdit = isError ? undefined : getFileEdit(args);
  const step =
    findToolStepDefinition(toolName) ?? (fileEdit ? FILE_EDIT_STEP : getFallbackToolStep(toolName));
  const { icon: Icon } = step;
  const { action, subject } = getToolStepTitleParts(step, { args: toolArgs, running });
  const hasArgs = isRecord(args) ? Object.keys(args).length > 0 : args !== undefined;
  const argsView = hasArgs ? step.formatArgs?.(toolArgs) : undefined;

  return (
    <WorkStep
      data-testid={fileEdit ? 'assistant-work-file-edit' : 'assistant-work-tool-call'}
      defaultOpen={defaultOpen}
      icon={<Icon size={WORK_ICON_SIZE} />}
      title={
        <>
          {action}
          {subject ? (
            <>
              {' '}
              <span className={cn(SUBJECT_CLASS, step.subjectIsCode && CODE_SUBJECT_CLASS)}>
                {subject}
              </span>
            </>
          ) : null}
          {fileEdit ? (
            <EditStats additions={fileEdit.additions} deletions={fileEdit.deletions} />
          ) : null}
        </>
      }
    >
      {fileEdit ? (
        <FileEditContent args={args} edit={fileEdit} />
      ) : (
        <Stack gap="density-md">
          {argsView ? (
            <WorkCodeSnippet
              label={argsView.label}
              language={argsView.language}
              value={argsView.text}
            />
          ) : (
            <PayloadSection label="Arguments" value={hasArgs ? args : argsText || '{}'} />
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
