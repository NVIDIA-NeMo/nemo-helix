// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EntityCardStat, EntityCardTag } from '@nemo/common/src/components/EntityCard/types';
import { OverflowGroup } from '@nemo/common/src/components/OverflowGroup';
import { Button, Card, Flex, Stack, Tag, Text } from '@nvidia/foundations-react-core';
import cn from 'classnames';
import type { ReactNode, Ref } from 'react';

export interface EntityCardProps {
  title: string;
  titleKind?: 'label/semibold/lg' | 'mono/md';
  /** Icon + text row under the title, e.g. a model creator or tool type, with optional trailing content. */
  subtitle?: { icon?: ReactNode; label: string; trailing?: ReactNode };
  description?: string;
  badges?: ReactNode;
  /** Footer-left tags, collapsed into a "+N" popover when they overflow. */
  tags?: readonly EntityCardTag[];
  /** Footer-right icon + label pairs. */
  stats?: readonly EntityCardStat[];
  onClick?: () => void;
  className?: string;
}

/** Shared card layout for entity grids: header, description, badges, and a tags/stats footer. */
export const EntityCard = ({
  title,
  titleKind = 'label/semibold/lg',
  subtitle,
  description,
  badges,
  tags = [],
  stats = [],
  onClick,
  className,
}: EntityCardProps) => {
  const Body = onClick ? 'button' : 'div';

  return (
    <Card
      asChild
      interactive={Boolean(onClick)}
      className={cn('[&_.nv-card-content-header]:overflow-hidden', className)}
      slotHeader={
        <Stack gap="density-sm" className="min-w-0">
          <Text kind={titleKind} className="leading-normal truncate block" title={title}>
            {title}
          </Text>
          {subtitle && (
            <Flex gap="density-sm" align="center">
              {subtitle.icon}
              <Text kind="body/regular/md" className="text-placeholder truncate">
                {subtitle.label}
              </Text>
              {subtitle.trailing}
            </Flex>
          )}
        </Stack>
      }
    >
      <Body onClick={onClick} className="text-left w-full h-full flex flex-col gap-2">
        {description && (
          <Text kind="body/regular/md" className="text-secondary line-clamp-2 leading-normal">
            {description}
          </Text>
        )}
        {badges && <Flex gap="density-sm">{badges}</Flex>}
        {(tags.length > 0 || stats.length > 0) && (
          <div className="grid grid-cols-[1fr_auto] items-center gap-4 border-t -mx-6 px-6 pt-4 -mb-2 mt-auto border-base">
            <div className="min-w-0 overflow-hidden">
              {tags.length > 0 && (
                <OverflowGroup
                  lines={1}
                  gap={4}
                  kind="popover"
                  renderSeeMoreButton={({ hiddenChildren, ref }) => (
                    <Button
                      ref={ref as Ref<HTMLButtonElement>}
                      kind="tertiary"
                      size="small"
                      onClick={(e) => e.stopPropagation()}
                    >
                      +{hiddenChildren.length}
                    </Button>
                  )}
                >
                  {tags.map((tag) => (
                    <Tag key={tag.key} kind="outline" color="gray" readOnly title={tag.title}>
                      {tag.label}
                    </Tag>
                  ))}
                </OverflowGroup>
              )}
            </div>
            <Flex gap="3" align="center">
              {stats.map((stat) => (
                <Flex key={stat.key} gap="1" align="center">
                  {stat.icon}
                  <Text kind="label/regular/sm">{stat.label}</Text>
                </Flex>
              ))}
            </Flex>
          </div>
        )}
      </Body>
    </Card>
  );
};
