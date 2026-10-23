// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChevronDown } from '@nemo/icons/icons';
import { Text } from '@nvidia/foundations-react-core';
import { type ReactNode } from 'react';

interface FloatingPanelProps {
  children: ReactNode;
  onOpenChange: (open: boolean) => void;
  open: boolean;
  title: string;
}

export const FloatingPanel = ({ children, onOpenChange, open, title }: FloatingPanelProps) => {
  const actionLabel = `${open ? 'Collapse' : 'Expand'} ${title}`;

  return (
    <section
      aria-label={title}
      className={`overflow-hidden rounded border border-base bg-surface-base dark:bg-surface-raised ${open ? 'flex min-h-0 flex-1 flex-col' : 'shrink-0'}`}
    >
      <button
        aria-expanded={open}
        aria-label={actionLabel}
        className="flex w-full cursor-pointer items-center justify-between gap-density-sm px-density-md py-density-sm text-left text-secondary transition-colors hover:bg-surface-sunken focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        type="button"
        onClick={() => onOpenChange(!open)}
      >
        <Text kind="label/bold/md">{title}</Text>
        <ChevronDown
          aria-hidden="true"
          className={`size-4 shrink-0 transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>
      {open && (
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden border-t border-base">
          {children}
        </div>
      )}
    </section>
  );
};
