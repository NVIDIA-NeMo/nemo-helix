// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCopyToClipboard } from '@nemo/common/src/hooks/useCopyToClipboard';
import { Check, CopyGeneric } from '@nemo/icons/icons';
import {
  Button,
  CodeSnippetActions,
  CodeSnippetCode,
  CodeSnippetRoot,
  Text,
  Tooltip,
  type CodeSnippetLanguage,
} from '@nvidia/foundations-react-core';
import cn from 'classnames';
import { useEffect, useState } from 'react';

const COPY_FEEDBACK_MS = 3000;

type CopyState = 'idle' | 'copied' | 'failed';

interface WorkCodeSnippetProps {
  readonly className?: string;
  readonly label: string;
  readonly labelClassName?: string;
  readonly language: CodeSnippetLanguage;
  readonly value: string;
}

// KUI's built-in copy button swallows clipboard failures, which happen on plain http outside localhost.
const CopySnippetButton = ({ value }: { readonly value: string }) => {
  const [copyState, setCopyState] = useState<CopyState>('idle');
  const { copyToClipboard } = useCopyToClipboard({
    onSuccess: () => setCopyState('copied'),
    onError: () => setCopyState('failed'),
  });

  useEffect(() => {
    if (copyState === 'idle') {
      return undefined;
    }
    const timeout = setTimeout(() => setCopyState('idle'), COPY_FEEDBACK_MS);
    return () => clearTimeout(timeout);
  }, [copyState]);

  return (
    <Tooltip
      slotContent={copyState === 'failed' ? 'Could not copy: clipboard access was blocked' : 'Copy'}
    >
      <Button
        aria-label="Copy"
        kind="tertiary"
        size="tiny"
        type="button"
        onClick={() => void copyToClipboard(value)}
      >
        {copyState === 'copied' ? <Check size="12" /> : <CopyGeneric size="12" />}
      </Button>
    </Tooltip>
  );
};

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
      <CopySnippetButton value={value} />
    </CodeSnippetActions>
    <CodeSnippetCode className={className} language={language} value={value} />
  </CodeSnippetRoot>
);
