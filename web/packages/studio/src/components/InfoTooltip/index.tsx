// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { InfoCircle } from '@nemo/icons/icons';
import { Text, Tooltip } from '@nvidia/foundations-react-core';
import { tooltipClassName } from '@studio/styles/common';

export const InfoTooltip = ({
  message,
  position,
}: {
  message: string;
  position?: 'top' | 'bottom' | 'left' | 'right';
}) => {
  const content = (
    <div className={tooltipClassName}>
      <Text>{message}</Text>
    </div>
  );
  return (
    <Tooltip slotContent={content} side={position || 'top'}>
      <InfoCircle width="12px" height="12px" />
    </Tooltip>
  );
};
