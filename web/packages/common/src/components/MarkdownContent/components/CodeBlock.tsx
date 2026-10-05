// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { extractLanguage } from '@nemo/common/src/components/MarkdownContent/utils';
import { isCodeSnippetLanguage } from '@nemo/common/src/utils/codeSnippet';
import { CodeSnippet } from '@nvidia/foundations-react-core';
import type { ComponentProps, FC } from 'react';
import { type ExtraProps } from 'react-markdown';

type CodeBlockProps = ComponentProps<'code'> & ExtraProps;

// `!` (important) beats CodeSnippet's own surface styles; `[&&]:` was used
// before but composes with `dark:` into a descendant selector that never
// matches, so the dark background silently fell back to the light one.
const CODE_BLOCK_SURFACE_CLASS = '!bg-gray-050 dark:!bg-gray-900';
const INLINE_CODE_SURFACE_CLASS =
  '!rounded !bg-gray-050 !px-1 !py-0.5 !font-sans dark:!bg-gray-900';

export const CodeBlock: FC<CodeBlockProps> = ({ className, children }) => {
  const isFenced = className?.includes('language-') ?? false;
  const extractedLanguage = extractLanguage(className);
  const language =
    extractedLanguage && isCodeSnippetLanguage(extractedLanguage) ? extractedLanguage : 'markdown';

  return (
    <CodeSnippet
      value={String(children).replace(/\n$/, '')}
      language={language}
      kind={isFenced ? 'block' : 'inline'}
      className={isFenced ? 'mb-density-md' : undefined}
      attributes={{
        CodeSnippetCode: {
          className: isFenced ? CODE_BLOCK_SURFACE_CLASS : INLINE_CODE_SURFACE_CLASS,
        },
      }}
    />
  );
};
