// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PropsWithChildren, ReactNode } from 'react';
import type { Components } from 'react-markdown';

export interface MarkdownTableOptions {
  expandableCells?: boolean;
}

export interface MessageContentProps {
  content?: string | null;
  markdownLinkComponent?: Components['a'];
  markdownTableOptions?: MarkdownTableOptions;
  renderAsMarkdown?: boolean;
}

export interface MarkdownTableColumn {
  id: string;
  header: ReactNode;
}

export interface MarkdownTableRow {
  id: string;
  cells: readonly ReactNode[];
  cellValues: readonly string[];
  expandedRowIds?: ReadonlySet<string>;
}

export interface ElementWithChildrenProps {
  children?: ReactNode;
}

export interface MarkdownAstNode {
  children?: MarkdownAstNode[];
  ordered?: boolean | null;
  spread?: boolean;
  start?: number | null;
  type: string;
  value?: unknown;
}

export interface MarkdownAstParent extends MarkdownAstNode {
  children: MarkdownAstNode[];
}

export interface MarkdownAstListNode extends MarkdownAstParent {
  ordered?: boolean | null;
  start?: number | null;
  type: 'list';
}

export interface MarkdownAstListItemNode extends MarkdownAstParent {
  spread?: boolean;
  type: 'listItem';
}

export interface MarkdownTableData {
  columns: readonly MarkdownTableColumn[];
  rows: readonly MarkdownTableRow[];
}

export interface MarkdownDataViewTableProps extends PropsWithChildren {
  options?: MarkdownTableOptions;
}

export interface MarkdownTableCellProps {
  children: ReactNode;
  expanded: boolean;
  expandable: boolean;
  onToggle: () => void;
}
