// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge } from '@nvidia/foundations-react-core';
import { ArrowRight, Clock, Shield } from '@nvidia/react-gui-icons/icons';
import type { Verdict } from '@studio/api/guardrail-checks/types';
import type { FC } from 'react';

/** Solid status badge for a check's latest-run verdict (yellow blocked / green allowed). */
export const ResultIndicator: FC<{ status: Verdict | undefined }> = ({ status }) => {
  if (status === 'blocked') {
    return (
      <Badge color="yellow" kind="solid">
        <Shield size="14" />
        Blocked
      </Badge>
    );
  }
  if (status === 'success') {
    return (
      <Badge color="green" kind="solid">
        <ArrowRight size="14" />
        Allowed
      </Badge>
    );
  }
  return (
    <Badge color="gray" kind="solid">
      <Clock size="14" />
      Not run
    </Badge>
  );
};
