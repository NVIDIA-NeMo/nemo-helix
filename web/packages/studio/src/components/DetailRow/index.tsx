// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Text } from '@nvidia/foundations-react-core';
import { Eye, Trash, Wrench } from '@nvidia/react-gui-icons/icons';
import { FC } from 'react';

export interface DetailRowProps {
  label: string;
  onView?: (name: string) => void;
  onDelete?: (name: string) => void;
  isEditable?: boolean;
  disabled?: boolean;
  icon?: React.ReactNode;
}

export const DetailRow: FC<DetailRowProps> = ({
  label,
  onView,
  onDelete,
  disabled,
  isEditable = false,
  icon,
}) => {
  const handleView = () => {
    onView?.(label);
  };

  const handleDelete = () => {
    onDelete?.(label);
  };

  return (
    <Flex
      align="center"
      justify="between"
      className={`h-[44px] border-b border-base first:border-b-0 border-t ${disabled ? 'opacity-50' : 'opacity-100'}`}
      data-testid="detail-row-container"
    >
      <Flex gap="density-md" align="center" className="overflow-hidden whitespace-pre">
        {icon ?? <Wrench size="18px" />}
        <Text className="overflow-hidden text-ellipsis leading-normal" kind="label/regular/md">
          {label}
        </Text>
      </Flex>
      <Flex gap="density-sm" align="center">
        {onView && (
          <Button
            type="button"
            kind="tertiary"
            size="small"
            onClick={handleView}
            aria-label="View metadata"
            disabled={disabled}
          >
            <Eye />
          </Button>
        )}

        {onDelete && isEditable && (
          <Button
            type="button"
            kind="tertiary"
            size="small"
            onClick={handleDelete}
            aria-label="Remove tool"
            disabled={disabled}
          >
            <Trash />
          </Button>
        )}
      </Flex>
    </Flex>
  );
};
