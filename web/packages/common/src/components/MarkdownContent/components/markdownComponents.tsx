// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Blockquote } from '@nemo/common/src/components/MarkdownContent/components/Blockquote';
import { CodeBlock } from '@nemo/common/src/components/MarkdownContent/components/CodeBlock';
import { Anchor, Divider, Text } from '@nvidia/foundations-react-core';
import cn from 'classnames';
import type { Components } from 'react-markdown';

// Map of HTML tag → React renderer used by react-markdown. Trivial markup
// wrappers stay inline; non-trivial handlers (`code`, `blockquote`) are
// extracted into their own component files.
export const markdownComponents: Components = {
  // Headings render Foundations typography on a semantic heading element via
  // radix's `asChild` pattern.
  h1: ({ children }) => (
    <Text asChild kind="title/2xl">
      <h1 className="mb-density-lg">{children}</h1>
    </Text>
  ),
  h2: ({ children }) => (
    <Text asChild kind="title/xl">
      <h2 className="mb-density-md mt-density-xl">{children}</h2>
    </Text>
  ),
  h3: ({ children }) => (
    <Text asChild kind="title/lg">
      <h3 className="mb-density-sm mt-density-lg">{children}</h3>
    </Text>
  ),
  h4: ({ children }) => (
    <Text asChild kind="title/md">
      <h4 className="mb-density-sm mt-density-lg">{children}</h4>
    </Text>
  ),
  h5: ({ children }) => (
    <Text asChild kind="title/sm">
      <h5 className="mb-density-sm mt-density-lg">{children}</h5>
    </Text>
  ),
  h6: ({ children }) => (
    <Text asChild kind="title/xs">
      <h6 className="mb-density-sm mt-density-lg">{children}</h6>
    </Text>
  ),

  p: ({ children }) => (
    <Text asChild kind="body/regular/md">
      <p className="mb-density-lg">{children}</p>
    </Text>
  ),

  a: ({ href, children }) => (
    <Anchor href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </Anchor>
  ),

  code: CodeBlock,
  pre: ({ children }) => <>{children}</>,

  ul: ({ children, className }) => (
    <ul className={cn('mb-density-lg list-disc pl-density-lg', className)}>{children}</ul>
  ),
  ol: ({ children, className }) => (
    <ol className={cn('mb-density-lg list-decimal pl-density-lg', className)}>{children}</ol>
  ),
  li: ({ children, className }) => <li className={cn('mb-density-xs', className)}>{children}</li>,

  table: ({ children }) => (
    <table className="mb-density-md min-w-full table-auto rounded-lg border border-base">
      {children}
    </table>
  ),
  thead: ({ children }) => <thead className="bg-surface-sunken">{children}</thead>,
  tr: ({ children }) => <tr className="border-t border-base">{children}</tr>,
  th: ({ children }) => (
    <th className="px-density-md py-density-sm text-left font-semibold">{children}</th>
  ),
  td: ({ children }) => (
    <td className="px-density-md py-density-sm text-left align-top">{children}</td>
  ),

  hr: () => <Divider className="my-density-lg" />,
  img: ({ src, alt }) => <img src={src} alt={alt ?? ''} className="max-w-full" />,

  blockquote: Blockquote,
};
