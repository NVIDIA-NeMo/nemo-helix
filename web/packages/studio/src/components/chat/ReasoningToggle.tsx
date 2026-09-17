// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Switch, Tooltip } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface ReasoningToggleProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  disabled?: boolean;
}

export const ReasoningToggle: FC<ReasoningToggleProps> = ({
  checked,
  onCheckedChange,
  disabled,
}) => (
  <Tooltip
    slotContent="Ask the model to skip reasoning. A model that does not support the parameter keeps reasoning."
    side="top"
  >
    <Switch
      size="small"
      name="reasoning"
      checked={checked}
      onCheckedChange={onCheckedChange}
      disabled={disabled}
      slotLabel="Reasoning"
    />
  </Tooltip>
);
