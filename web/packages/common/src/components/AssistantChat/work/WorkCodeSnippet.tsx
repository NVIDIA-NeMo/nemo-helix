// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CopyButton } from '@nemo/common/src/components/CopyButton';
import { CopyGeneric } from '@nemo/icons/icons';
import {
  CodeSnippetActions,
  CodeSnippetCode,
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
      <CopyButton aria-label="Copy" color="neutral" kind="tertiary" size="tiny" text={value}>
        <CopyGeneric size="12" />
      </CopyButton>
    </CodeSnippetActions>
    <CodeSnippetCode className={className} language={language} value={value} />
  </CodeSnippetRoot>
);
