// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CodeSnippet, Text, type CodeSnippetLanguage } from '@nvidia/foundations-react-core';
import cn from 'classnames';

interface WorkCodeSnippetProps {
  readonly className?: string;
  readonly label: string;
  readonly labelClassName?: string;
  readonly language: CodeSnippetLanguage;
  readonly value: string;
}

export const WorkCodeSnippet = ({
  className,
  label,
  labelClassName,
  language,
  value,
}: WorkCodeSnippetProps) => (
  <CodeSnippet
    kind="block"
    rows={12}
    language={language}
    value={value}
    slotActions={
      <Text kind="mono/sm" className={cn('mr-auto truncate', labelClassName)} title={label}>
        {label}
      </Text>
    }
    attributes={{ CodeSnippetCode: { className } }}
  />
);
