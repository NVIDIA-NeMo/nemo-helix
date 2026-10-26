// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { ChevronDown, ChevronRight } from '@nemo/icons/icons';
import {
  Button,
  CodeSnippet,
  Flex,
  PaginationArrowButton,
  PaginationNavigationGroup,
  PaginationRoot,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { type ComponentProps, type FC, useState } from 'react';

type CodeSnippetLanguage = NonNullable<ComponentProps<typeof CodeSnippet>['language']>;

interface Props {
  beforeValue: string;
  afterValue: string;
  beforeLabel?: string;
  afterLabel?: string;
  currentRow?: number;
  totalRows?: number;
  onRowChange?: (row: number) => void;
  language?: CodeSnippetLanguage;
}

export const PreviewOutputPanel: FC<Props> = ({
  beforeValue,
  afterValue,
  beforeLabel = 'SOURCE ROW',
  afterLabel = 'MODIFIED ROW',
  currentRow = 1,
  totalRows,
  onRowChange,
  language = 'json',
}) => {
  const [collapsed, setCollapsed] = useState(false);
  const toast = useToast();
  return (
    <Stack
      gap="3"
      paddingY="3"
      paddingX="3"
      className="w-full overflow-hidden rounded-xl border border-base bg-surface-base"
    >
      <Flex gap="2.5" align="center" className="w-full">
        <Button
          size="small"
          type="button"
          kind="tertiary"
          onClick={() => setCollapsed(!collapsed)}
          className="shrink-0 text-secondary transition-colors hover:text-primary"
          aria-label={collapsed ? 'Expand preview' : 'Collapse preview'}
          aria-expanded={!collapsed}
        >
          {collapsed ? <ChevronRight size="14" /> : <ChevronDown size="14" />}
        </Button>
        <Flex gap="2" align="center" className="min-w-0 flex-1" justify="between">
          <Text kind="body/bold/sm" className="shrink-0">
            Preview output
          </Text>
          {totalRows !== undefined && (
            <PaginationRoot
              page={currentRow}
              pageSize={1}
              totalItems={totalRows}
              onPageChange={onRowChange}
              className="justify-end"
            >
              <PaginationNavigationGroup className="gap-1">
                <PaginationArrowButton direction="previous" />
                <Text kind="body/regular/xs" className="px-1">
                  Row {currentRow} of {totalRows.toLocaleString()}
                </Text>
                <PaginationArrowButton direction="next" />
              </PaginationNavigationGroup>
            </PaginationRoot>
          )}
        </Flex>
      </Flex>

      {!collapsed && (
        <Flex gap="3" align="start" className="w-full">
          <Stack gap="1.5" className="min-w-0 flex-1">
            <CodeSnippet
              value={beforeValue}
              language={language}
              kind="block"
              onCopySuccess={() => {
                toast.success('Copied to clipboard!');
              }}
              slotActions={
                <Text kind="label/bold/sm" className="text-secondary">
                  {beforeLabel}
                </Text>
              }
              attributes={{
                CodeSnippetCode: { className: 'max-h-[240px]' },
                CodeSnippetActions: { className: 'justify-between' },
              }}
            />
          </Stack>
          <Stack gap="1.5" className="min-w-0 flex-1">
            <CodeSnippet
              value={afterValue}
              language={language}
              kind="block"
              onCopySuccess={() => {
                toast.success('Copied to clipboard!');
              }}
              slotActions={
                <Text kind="label/bold/sm" className="text-(--text-color-accent-green)">
                  {afterLabel}
                </Text>
              }
              attributes={{
                CodeSnippetCode: { className: 'max-h-[240px]' },
                CodeSnippetActions: { className: 'justify-between' },
              }}
            />
          </Stack>
        </Flex>
      )}
    </Stack>
  );
};
