// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge } from '@nvidia/foundations-react-core';
import type { TestCaseVerdict } from '@studio/components/evaluation/testCaseVerdict';
import { type FC } from 'react';

const VERDICT_BADGES: Record<
  TestCaseVerdict,
  { label: string; kind: 'solid' | 'outline'; color: 'green' | 'red' | 'gray' }
> = {
  passed: { label: 'Passed', kind: 'solid', color: 'green' },
  failed: { label: 'Failed', kind: 'solid', color: 'red' },
  error: { label: 'Error', kind: 'outline', color: 'red' },
  'not-scored': { label: 'Not scored', kind: 'outline', color: 'gray' },
};

export const TestCaseVerdictBadge: FC<{ verdict: TestCaseVerdict }> = ({ verdict }) => {
  const { label, kind, color } = VERDICT_BADGES[verdict];
  return (
    <Badge kind={kind} color={color}>
      {label}
    </Badge>
  );
};
