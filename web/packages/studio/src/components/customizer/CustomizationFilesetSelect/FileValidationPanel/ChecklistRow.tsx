// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@nvidia/foundations-react-core';
import { Warning, CheckCircle, CloseCircle } from '@nvidia/react-gui-icons/icons';
import { FC, ReactNode } from 'react';

// Validation rows are strictly binary: pass (green check), warn (yellow triangle),
// or fail (red X). Anything we can't actually evaluate is hidden by the panel
// rather than rendered as a third "neutral" state — that was confusing in
// practice ("did we pass? did we fail?").
export type ChecklistStatus = 'ok' | 'warning' | 'fail';

const StatusIcon: FC<{ status: ChecklistStatus }> = ({ status }) => {
  if (status === 'ok')
    return <CheckCircle width={16} height={16} className="text-feedback-success" />;
  if (status === 'warning')
    return <Warning width={16} height={16} className="text-feedback-warning" />;
  return <CloseCircle width={16} height={16} className="text-feedback-danger" />;
};

export interface ChecklistRowProps {
  status: ChecklistStatus;
  label: ReactNode;
}

export const ChecklistRow: FC<ChecklistRowProps> = ({ status, label }) => (
  <Flex align="center" gap="density-sm">
    <StatusIcon status={status} />
    <Text kind="body/regular/md">{label}</Text>
  </Flex>
);
