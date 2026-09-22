// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { markdownComponents } from '@nemo/common/src/components/MarkdownContent/components/markdownComponents';
import { remarkCallouts } from '@nemo/common/src/components/MarkdownContent/remarkCallouts';
import cn from 'classnames';
import { type FC } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export interface MarkdownContentProps {
  content: string;
  className?: string;
  /**
   * Drop markdown images (`![](…)`) instead of rendering them. Use for
   * untrusted, user/agent-authored content: an `<img src>` pointing at an
   * attacker-controlled host would make every viewer's browser fetch it,
   * leaking their IP + request metadata (CWE-200). `skipHtml` does not cover
   * markdown image syntax, so this is the guard for it.
   */
  disableImages?: boolean;
}

export const MarkdownContent: FC<MarkdownContentProps> = ({
  content,
  className,
  disableImages,
}) => {
  const components = disableImages
    ? { ...markdownComponents, img: () => null }
    : markdownComponents;
  return (
    <div className={cn('max-w-none', className)}>
      <Markdown remarkPlugins={[remarkGfm, remarkCallouts]} skipHtml components={components}>
        {content}
      </Markdown>
    </div>
  );
};
