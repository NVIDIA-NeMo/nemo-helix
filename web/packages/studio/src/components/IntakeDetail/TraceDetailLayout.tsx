// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack } from '@nvidia/foundations-react-core';
import { ResizeablePanel } from '@studio/components/common/ResizeablePanel';
import type { FC, ReactNode } from 'react';

const SIDEBAR_MIN_WIDTH_PX = 18 * 16;
const DETAIL_MIN_WIDTH_PX = 20 * 16;

interface TraceDetailLayoutProps {
  navigation: ReactNode;
  children: ReactNode;
}

/** Shared two-pane shell for Session, trace, and span detail selections. */
export const TraceDetailLayout: FC<TraceDetailLayoutProps> = ({ navigation, children }) => (
  <ResizeablePanel
    slotLeft={
      <aside
        data-testid="trace-trajectory-sidebar"
        className="max-h-[calc(100vh-6rem)] overflow-y-auto rounded-lg bg-surface-raised p-density-xs"
      >
        {navigation}
      </aside>
    }
    slotRight={
      <Stack gap="density-lg" className="min-w-0">
        {children}
      </Stack>
    }
    defaultLeftWidth={SIDEBAR_MIN_WIDTH_PX}
    minLeftWidth={SIDEBAR_MIN_WIDTH_PX}
    minRightWidth={DETAIL_MIN_WIDTH_PX}
    resizeStep={16}
    separatorLabel="Resize trace trajectory sidebar"
    variant="plain"
    className="min-w-0 items-start"
    leftClassName="sticky top-density-lg hidden max-h-[calc(100vh-6rem)] self-start lg:block"
    separatorClassName="sticky top-density-lg hidden h-[calc(100vh-6rem)] self-start lg:flex"
  />
);
