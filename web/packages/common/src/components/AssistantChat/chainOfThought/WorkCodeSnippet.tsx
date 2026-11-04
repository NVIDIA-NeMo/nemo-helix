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
  readonly label: string;
  readonly language: CodeSnippetLanguage;
  readonly tone?: 'default' | 'danger';
  readonly value: string;
  /** Wraps long lines instead of scrolling, so the block never widens its container. */
  readonly wrap?: boolean;
}

/** A code block with its label (a file path, "Arguments", "Result") in the header bar. */
export const WorkCodeSnippet = ({
  label,
  language,
  tone = 'default',
  value,
  wrap = false,
}: WorkCodeSnippetProps) => (
  <CodeSnippetRoot kind="block" rows={12}>
    <CodeSnippetActions>
      <Text
        kind="mono/sm"
        className={cn('mr-auto truncate', tone === 'danger' && 'text-feedback-danger')}
        title={label}
      >
        {label}
      </Text>
      <CodeSnippetCopyButton value={value} />
    </CodeSnippetActions>
    <CodeSnippetCode
      value={value}
      language={language}
      className={cn(wrap && '[&_pre]:whitespace-pre-wrap [&_pre]:break-all')}
    />
  </CodeSnippetRoot>
);
