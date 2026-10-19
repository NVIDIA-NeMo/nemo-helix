// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Anchor, Badge, Button, Card, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { FC, ReactNode } from 'react';
import { useNavigate } from 'react-router';

interface DashboardCardProps {
  icon: ReactNode;
  title: string;
  description: string;
  docsUrl?: string;
  actionLabel?: string;
  actionHref?: string;
  comingSoon?: boolean;
}

export const DashboardCard: FC<DashboardCardProps> = ({
  icon,
  title,
  description,
  docsUrl,
  actionLabel,
  actionHref,
  comingSoon,
}) => {
  const navigate = useNavigate();

  const handleActionClick = () => {
    if (actionHref && !comingSoon) {
      navigate(actionHref);
    }
  };

  return (
    <Card className="h-full">
      <Stack gap="density-xl" className="h-full">
        {/* Icon and Badge row */}
        <Flex justify="between" align="start">
          {icon}
          {comingSoon && (
            <Badge kind="solid" color="gray">
              Coming Soon
            </Badge>
          )}
        </Flex>

        {/* Title */}
        <Text kind="title/sm">{title}</Text>

        {/* Description with Docs link */}
        <Stack gap="density-md" className="flex-1">
          <Text className="text-secondary" kind="body/regular/md">
            {description}{' '}
            {docsUrl && (
              <Anchor href={docsUrl} target="_blank" rel="noopener noreferrer">
                Docs
              </Anchor>
            )}
          </Text>
        </Stack>

        {/* Action button area - maintains consistent height */}
        <Flex className="min-h-[40px]">
          {!comingSoon && actionHref && actionLabel && (
            <Button kind="secondary" color="brand" onClick={handleActionClick}>
              {actionLabel}
            </Button>
          )}
        </Flex>
      </Stack>
    </Card>
  );
};
