// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Panel } from '@nvidia/foundations-react-core';
import { Filter } from '@nvidia/react-gui-icons/icons';
import type { PropsWithChildren } from 'react';

export const FilterPanel = ({
  children,
  showFilters,
  containerTestId = 'table-filter-panel-container',
  panelTestId = 'table-filter-panel',
}: PropsWithChildren<{
  showFilters: boolean;
  containerTestId?: string;
  panelTestId?: string;
}>) => {
  return (
    <div
      data-testid={containerTestId}
      aria-hidden={!showFilters}
      className={`shrink-0 overflow-hidden transition-[width,opacity] duration-300 ease-out motion-reduce:transition-none ${
        showFilters
          ? 'w-[calc(300px+var(--spacing-density-xl))] opacity-100'
          : 'w-0 opacity-0 pointer-events-none'
      }`}
    >
      <div className="h-full pl-density-xl">
        <Panel
          data-testid={panelTestId}
          className="w-[300px] px-0 gap-0 h-full"
          slotHeading="Filter"
          slotIcon={<Filter width={24} height={24} />}
          elevation="high"
          density="compact"
          attributes={{
            PanelHeader: {
              className: 'px-4 pb-4 mb-0 border-base border-b-2 border-b',
            },
            PanelContent: {
              className: 'overflow-y-auto',
            },
          }}
        >
          {children}
        </Panel>
      </div>
    </div>
  );
};
