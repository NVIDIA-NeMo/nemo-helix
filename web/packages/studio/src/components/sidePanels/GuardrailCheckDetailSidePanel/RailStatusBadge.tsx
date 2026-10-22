// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge } from '@nvidia/foundations-react-core';
import { Shield } from '@nvidia/react-gui-icons/icons';
import type { Verdict } from '@studio/api/guardrail-checks/types';
import type { FC } from 'react';

export interface RailStatusBadgeProps {
  readonly status: Verdict;
}

/**
 * Verdict badge for a single rail.
 *
 * Deliberately not the shared ResultIndicator, which styles a check's *overall*
 * verdict: an individual rail that merely allowed the message is incidental
 * next to that, so it reads gray and unadorned and the shield is reserved for
 * a block.
 */
export const RailStatusBadge: FC<RailStatusBadgeProps> = ({ status }) => {
  if (status === 'blocked') {
    return (
      <Badge color="yellow" kind="solid">
        <Shield size="10" />
        Blocked
      </Badge>
    );
  }
  if (status === 'success') {
    return (
      <Badge color="gray" kind="solid">
        Allowed
      </Badge>
    );
  }
  return (
    <Badge color="gray" kind="outline">
      Unknown
    </Badge>
  );
};
