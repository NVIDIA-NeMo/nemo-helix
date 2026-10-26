// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ICON_COLOR_CLASS } from '@studio/components/AddColumnPalette/constants';
import type {
  AddColumnSelection,
  ColumnTypeOption,
} from '@studio/components/AddColumnPalette/types';
import { CardIconBadge, SelectableCard } from '@studio/components/common/SelectableCard';
import type { FC } from 'react';

interface ColumnTypeCardProps {
  option: ColumnTypeOption;
  onSelect: (selection: AddColumnSelection) => void;
  disabledReason?: string;
}

/**
 * A single column-type option, rendered as a {@link SelectableCard} so it is reachable and
 * activatable by keyboard (Tab to focus, Enter/Space to add) with no drag interaction.
 */
export const ColumnTypeCard: FC<ColumnTypeCardProps> = ({ option, onSelect, disabledReason }) => {
  const { icon: Icon, label, description, color, columnType, samplerType } = option;
  return (
    <SelectableCard
      className="w-full"
      title={label}
      subtitle={description}
      disabled={Boolean(disabledReason)}
      disabledReason={disabledReason}
      onActivate={() => onSelect({ columnType, samplerType })}
      leading={
        <CardIconBadge>
          <Icon size="15" className={ICON_COLOR_CLASS[color]} aria-hidden />
        </CardIconBadge>
      }
    />
  );
};
