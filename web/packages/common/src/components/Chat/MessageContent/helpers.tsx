// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_MARKDOWN_TABLE_OPTIONS } from '@nemo/common/src/components/Chat/MessageContent/constants';
import { MarkdownParagraph } from '@nemo/common/src/components/Chat/MessageContent/MarkdownParagraph';
import type {
  ElementWithChildrenProps,
  MarkdownTableData,
  MarkdownTableOptions,
} from '@nemo/common/src/components/Chat/MessageContent/types';
import { Text } from '@nvidia/foundations-react-core';
import { childrenToText } from '@nvidia/foundations-react-core/lib';
import { Children, isValidElement, type ReactElement, type ReactNode } from 'react';

export const isElementWithChildren = (
  node: ReactNode
): node is ReactElement<ElementWithChildrenProps> => isValidElement<ElementWithChildrenProps>(node);

export const isElementNamed = (
  node: ReactNode,
  elementName: 'thead' | 'tbody' | 'tr' | 'th' | 'td' | 'p'
): node is ReactElement<ElementWithChildrenProps> =>
  isElementWithChildren(node) && node.type === elementName;

export const getChildNodes = (node: ReactElement<ElementWithChildrenProps>): ReactNode[] =>
  Children.toArray(node.props.children);

export const isWhitespaceTextNode = (node: ReactNode): node is string =>
  typeof node === 'string' && node.trim().length === 0;

export const isMarkdownParagraphElement = (
  node: ReactNode
): node is ReactElement<ElementWithChildrenProps> =>
  isElementWithChildren(node) && (node.type === MarkdownParagraph || isElementNamed(node, 'p'));

export const renderListItemChildren = (children: ReactNode): ReactNode => {
  const childNodes = Children.toArray(children);
  const firstContentIndex = childNodes.findIndex((child) => !isWhitespaceTextNode(child));
  const firstContent = childNodes[firstContentIndex];

  if (firstContentIndex === -1 || !isMarkdownParagraphElement(firstContent)) {
    return children;
  }

  return [
    <Text asChild kind="body/regular/md" key="leading-list-paragraph">
      <span className="text-sm leading-6">{getChildNodes(firstContent)}</span>
    </Text>,
    ...childNodes.slice(firstContentIndex + 1),
  ];
};

export const getRowCells = (row: ReactElement<ElementWithChildrenProps>): readonly ReactNode[] =>
  getChildNodes(row)
    .filter((cell) => isElementNamed(cell, 'th') || isElementNamed(cell, 'td'))
    .map((cell) => cell.props.children ?? '');

export const getSectionRows = (
  section: ReactElement<ElementWithChildrenProps> | undefined
): readonly (readonly ReactNode[])[] => {
  if (!section) return [];

  return getChildNodes(section)
    .filter((child) => isElementNamed(child, 'tr'))
    .map(getRowCells);
};

export const getNodeText = (node: ReactNode): string => childrenToText(node).trim();

export const getMarkdownTableOptions = (
  options: MarkdownTableOptions | undefined
): Required<MarkdownTableOptions> => ({
  ...DEFAULT_MARKDOWN_TABLE_OPTIONS,
  ...options,
});

export const parseMarkdownTable = (children: ReactNode): MarkdownTableData => {
  const tableChildren = Children.toArray(children);
  const head = tableChildren.find((child) => isElementNamed(child, 'thead'));
  const bodies = tableChildren.filter((child) => isElementNamed(child, 'tbody'));
  const headerRows = getSectionRows(head);
  const bodyRows = bodies.flatMap(getSectionRows);
  const headerCells = headerRows[0] ?? bodyRows[0] ?? [];
  const dataRows = headerRows.length > 0 ? bodyRows : bodyRows.slice(1);
  const columnCount = Math.max(headerCells.length, ...dataRows.map((row) => row.length));

  return {
    columns: Array.from({ length: columnCount }, (_, index) => ({
      id: `column-${index}`,
      header: headerCells[index] ?? `Column ${index + 1}`,
    })),
    rows: dataRows.map((cells, rowIndex) => ({
      id: `row-${rowIndex}`,
      cells: Array.from({ length: columnCount }, (_, cellIndex) => cells[cellIndex] ?? ''),
      cellValues: Array.from({ length: columnCount }, (_, cellIndex) =>
        getNodeText(cells[cellIndex] ?? '')
      ),
    })),
  };
};
