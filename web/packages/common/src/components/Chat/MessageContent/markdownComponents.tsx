// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { INLINE_CODE_CLASS } from '@nemo/common/src/components/Chat/MessageContent/constants';
import { renderListItemChildren } from '@nemo/common/src/components/Chat/MessageContent/helpers';
import { MarkdownDataViewTable } from '@nemo/common/src/components/Chat/MessageContent/MarkdownDataViewTable';
import { MarkdownParagraph } from '@nemo/common/src/components/Chat/MessageContent/MarkdownParagraph';
import { Text } from '@nvidia/foundations-react-core';
import cn from 'classnames';
import type { Components } from 'react-markdown';

export const messageMarkdownComponents: Components = {
  h1: ({ children }) => (
    <Text asChild kind="title/xs">
      <h1 className="mb-density-sm mt-density-2xl">{children}</h1>
    </Text>
  ),
  h2: ({ children }) => (
    <Text asChild kind="label/bold/lg">
      <h2 className="mb-density-sm mt-density-xl">{children}</h2>
    </Text>
  ),
  h3: ({ children }) => (
    <Text asChild kind="label/bold/md">
      <h3 className="mb-density-xs mt-density-lg">{children}</h3>
    </Text>
  ),
  h4: ({ children }) => (
    <Text asChild kind="label/bold/md">
      <h4 className="mb-density-sm mt-density-md">{children}</h4>
    </Text>
  ),
  h5: ({ children }) => (
    <Text asChild kind="label/bold/sm">
      <h5 className="mb-density-sm mt-density-md">{children}</h5>
    </Text>
  ),
  h6: ({ children }) => (
    <Text asChild kind="label/bold/sm">
      <h6 className="mb-density-sm mt-density-md">{children}</h6>
    </Text>
  ),
  p: MarkdownParagraph,
  ul: ({ children, className }) => (
    <ul className={cn('my-density-md list-disc pl-density-2xl', className)}>{children}</ul>
  ),
  ol: ({ children, className, start }) => (
    <ol className={cn('my-density-md list-decimal pl-density-2xl', className)} start={start}>
      {children}
    </ol>
  ),
  li: ({ children, className }) => (
    <li
      className={cn(
        'mb-density-xs whitespace-normal pl-density-xs text-sm leading-6 last:mb-0 [&>p]:my-0',
        className
      )}
    >
      {renderListItemChildren(children)}
    </li>
  ),
  hr: () => <hr className="my-density-md border-base" />,
  blockquote: ({ children, className }) => (
    <blockquote
      className={cn(
        'my-density-sm border-l-2 border-base py-density-xs pl-density-md text-primary',
        className
      )}
    >
      {children}
    </blockquote>
  ),
  img: ({ src, alt }) => (
    <img src={src} alt={alt ?? ''} className="max-w-full rounded border border-base" />
  ),
  // Most chat surfaces keep model-supplied links inert. Consumers that know how
  // to handle links can provide their own renderer.
  a: ({ children }) => <span>{children}</span>,
  code: ({ children }) => <code className={INLINE_CODE_CLASS}>{children}</code>,
  table: ({ children }) => <MarkdownDataViewTable>{children}</MarkdownDataViewTable>,
};
