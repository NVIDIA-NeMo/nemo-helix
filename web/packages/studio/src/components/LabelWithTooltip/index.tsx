// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { InfoCircle } from '@nemo/icons/icons';
import { Tooltip, Stack, Label, LabelProps } from '@nvidia/foundations-react-core';
import { tooltipClassName } from '@studio/styles/common';
import { ComponentProps, FC } from 'react';

interface Props extends LabelProps {
  label: string;
  tooltipMessage: string;
  position?: ComponentProps<typeof Tooltip>['side'];
}

export const LabelWithTooltip: FC<Props> = ({ label, tooltipMessage, position, ...labelProps }) => {
  return (
    <Stack direction="row" gap="density-xs">
      <Label {...labelProps}>{label}</Label>
      <Tooltip slotContent={tooltipMessage} side={position ?? 'right'} className={tooltipClassName}>
        <InfoCircle />
      </Tooltip>
    </Stack>
  );
};
