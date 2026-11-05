// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FileEditStep } from '@nemo/common/src/components/AssistantChat/chainOfThought/FileEditStep';
import { WorkCodeSnippet } from '@nemo/common/src/components/AssistantChat/chainOfThought/WorkCodeSnippet';
import {
  formatToolPayload,
  getFileEdit,
} from '@nemo/common/src/components/AssistantChat/chainOfThought/workParts';
import {
  WORK_ICON_SIZE,
  WorkStep,
} from '@nemo/common/src/components/AssistantChat/chainOfThought/WorkStep';
import { WindowCode } from '@nemo/icons/icons';
import { Stack } from '@nvidia/foundations-react-core';

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
 * A tool call: its name, then the arguments it was called with and the result it returned.
 * Successful tools whose arguments carry a file path and content render as a file edit instead;
 * a failed one stays a tool call so its error is shown.
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
  if (fileEdit) return <FileEditStep defaultOpen={defaultOpen} edit={fileEdit} />;

  const hasArgs =
    args !== undefined && !(typeof args === 'object' && args && !Object.keys(args).length);

  return (
    <WorkStep
      data-testid="assistant-work-tool-call"
      defaultOpen={defaultOpen}
      icon={<WindowCode size={WORK_ICON_SIZE} />}
      title={running ? `Calling ${toolName}` : `Called ${toolName}`}
    >
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
    </WorkStep>
  );
};
