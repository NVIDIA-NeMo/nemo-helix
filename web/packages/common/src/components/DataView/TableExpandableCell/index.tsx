// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Expand } from '@nemo/icons/icons';
import { Text } from '@nvidia/foundations-react-core';
import type { FC, ReactNode } from 'react';

export interface TableExpandableCellState {
  title: string;
  content: string;
}

export const TableExpandableCell: FC<{
  content: string;
  title: string;
  onExpand: (state: TableExpandableCellState) => void;
  footer?: ReactNode;
  boldContent?: boolean;
}> = ({ content, title, onExpand, footer, boldContent }) => {
  return (
    <div className="group relative flex h-full flex-col">
      <button
        onClick={() => onExpand({ title, content })}
        className="absolute right-1 top-1 z-10 cursor-pointer rounded bg-surface-base/80 p-1 opacity-0 hover:bg-surface-sunken group-hover:opacity-100 focus-visible:opacity-100"
        aria-label="Expand cell"
      >
        <Expand size="12" />
      </button>
      <div className="max-h-[130px] overflow-y-auto px-3 py-2">
        <Text
          kind="body/regular/md"
          className={`whitespace-pre-wrap${boldContent ? ' font-bold' : ''}`}
        >
          {content}
        </Text>
      </div>
      {footer && <div className="mt-auto">{footer}</div>}
    </div>
  );
};
