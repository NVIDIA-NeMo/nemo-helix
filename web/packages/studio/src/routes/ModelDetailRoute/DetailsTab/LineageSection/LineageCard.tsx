// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Card, Flex, Skeleton, Stack, Text } from '@nvidia/foundations-react-core';
import type { FC, ReactNode } from 'react';
import { Link } from 'react-router';

export interface LineageCardProps {
  label: string;
  value: string;
  /** A status badge shown before the note. */
  badge?: ReactNode;
  note?: string;
  /** Shown beside the label, such as a copy button. */
  action?: ReactNode;
  /** Makes the whole card a link. Omitted when there is nowhere to go. */
  to?: string;
  loading?: boolean;
}

export const LineageCard: FC<LineageCardProps> = ({
  label,
  value,
  badge,
  note,
  action,
  to,
  loading,
}) => {
  const body = (
    <Stack gap="density-sm" className="min-w-0">
      <Flex align="center" justify="between" gap="density-sm" className="min-h-6">
        <Text kind="label/regular/sm" className="text-secondary">
          {label}
        </Text>
        {action}
      </Flex>
      {loading ? (
        <Skeleton className="h-5 w-3/4" />
      ) : (
        <>
          <Text kind="body/semibold/md" className="truncate" title={value}>
            {value}
          </Text>
          {(badge || note) && (
            <Flex align="center" gap="density-sm" className="min-h-6">
              {badge}
              {note && (
                <Text kind="body/regular/sm" className="text-secondary">
                  {note}
                </Text>
              )}
            </Flex>
          )}
        </>
      )}
    </Stack>
  );

  if (to && !loading) {
    return (
      <Card asChild interactive>
        <Link to={to}>{body}</Link>
      </Card>
    );
  }
  return <Card>{body}</Card>;
};
