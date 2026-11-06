// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  CodeSnippetActions,
  CodeSnippetCode,
  CodeSnippetCopyButton,
  CodeSnippetRoot,
  Text,
  type CodeSnippetLanguage,
} from '@nvidia/foundations-react-core';
import cn from 'classnames';

interface WorkCodeSnippetProps {
  readonly className?: string;
  readonly label: string;
  readonly labelClassName?: string;
  readonly language: CodeSnippetLanguage;
  readonly value: string;
}

/** A code block with its label (a file path, "Arguments", "Result") in the header bar. */
export const WorkCodeSnippet = ({
  className,
  label,
  labelClassName,
  language,
  value,
}: WorkCodeSnippetProps) => (
  <CodeSnippetRoot kind="block" rows={12}>
    <CodeSnippetActions>
      <Text kind="mono/sm" className={cn('mr-auto truncate', labelClassName)} title={label}>
        {label}
      </Text>
      <CodeSnippetCopyButton value={value} />
    </CodeSnippetActions>
    <CodeSnippetCode value={value} language={language} className={className} />
  </CodeSnippetRoot>
);
