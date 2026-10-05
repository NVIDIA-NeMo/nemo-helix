// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { languageInCode } from '@nemo/common/src/utils/codeSnippet';
import { CodeSnippet, Text } from '@nvidia/foundations-react-core';
import cn from 'classnames';
import type { FC } from 'react';

export interface CodeDisplayProps {
  containerClassName?: string;
  children?: string;
}

const CODE_BLOCK_SURFACE_CLASS = '!bg-gray-050 !py-density-xs dark:!bg-gray-900';

export const CodeDisplay: FC<CodeDisplayProps> = ({ children, containerClassName }) => {
  const detectedLang = languageInCode(children || '');
  const code = detectedLang ? children?.slice(detectedLang.length).trim() || '' : children || '';

  return (
    <div className={cn('my-density-md', containerClassName)} data-testid="code-display">
      <CodeSnippet
        value={code || ''}
        language={detectedLang || 'markdown'}
        kind="block"
        attributes={{
          CodeSnippetCode: {
            className: CODE_BLOCK_SURFACE_CLASS,
          },
        }}
        slotActions={
          detectedLang && (
            <Text kind="mono/md" className="w-full">
              {detectedLang}
            </Text>
          )
        }
      />
    </div>
  );
};
