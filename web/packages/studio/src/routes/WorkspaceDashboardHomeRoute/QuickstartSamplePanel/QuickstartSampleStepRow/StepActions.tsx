// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex } from '@nvidia/foundations-react-core';
import type { QuickstartSampleAction } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSamplePanel/quickstartSampleContent';
import type { FC } from 'react';
import { Link } from 'react-router';

export interface StepActionsProps {
  actions: readonly QuickstartSampleAction[];
}

/** The Studio view's body: the step's affordances, beside the description until it wraps. */
export const StepActions: FC<StepActionsProps> = ({ actions }) =>
  actions.length === 0 ? null : (
    <Flex gap="density-md" wrap="wrap" className="shrink-0">
      {actions.map((action) => (
        <Button
          key={`${action.label}::${action.href}`}
          color="brand"
          kind="tertiary"
          size="small"
          asChild
        >
          <Link to={action.href}>{action.label}</Link>
        </Button>
      ))}
    </Flex>
  );
