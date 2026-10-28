// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FolderOpen } from '@nemo/icons/icons';
import { StatusMessage, Flex } from '@nvidia/foundations-react-core';
import { FC } from 'react';

interface Props {
  className?: string;
  header: string;
  emptyMessage?: React.ReactNode;
  icon?: React.ReactNode;
  actions?: React.ReactNode;
}

/**
 * A generic component that renders a standard empty state in-place of a table when there is no data.
 */
export const TableEmptyState: FC<Props> = ({ className, header, emptyMessage, icon, actions }) => {
  return (
    <Flex className={`h-full w-full ${className}`} justify="center">
      <StatusMessage
        slotHeading={header}
        slotMedia={icon ?? <FolderOpen data-testid="icon-folder-open" className="size-12" />}
        slotSubheading={emptyMessage}
        slotFooter={actions ? <Flex gap="density-sm">{actions}</Flex> : null}
      />
    </Flex>
  );
};
