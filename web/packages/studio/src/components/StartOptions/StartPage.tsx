// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Block, Divider, Flex, PageHeader, Skeleton, Stack, Text } from '@nvidia/foundations-react-core';
import { StartTile } from '@studio/components/StartOptions/StartTile';
import type { StartPageProps } from '@studio/components/StartOptions/types';
import type { FC } from 'react';

/** The column the whole flow sits in, per the design. */
const CONTENT_WIDTH = 'w-full max-w-[768px]';

/** The design's tile is `radius/md`; Card's own `radius-density-xl` is visibly rounder. */
const TILE_RADIUS = 'rounded-[var(--radius-md)]';

/** The tiles carry no visible prompt, so the group that holds them is named here. */
const GROUP_LABEL = 'How do you want to start?';

/** The design's tile sets its title at 14px semibold over a 12px description. */
const TILE_LABEL_KIND = 'label/semibold/md' as const;
const TILE_DESCRIPTION_KIND = 'label/regular/sm' as const;

/** Placeholders shown for a group that is still loading — the design's rows are pairs. */
const PLACEHOLDER_TILES = 2;

/** Picking any tile — option or template — starts that flow immediately. */
export const StartPage: FC<StartPageProps> = ({
  heading,
  headingDescription,
  options,
  templateGroups = [],
  onSelect,
  disabled = false,
  busyId = null,
  busyLabel,
  templatesTag,
  slotBanner,
}) => {
  const groups = templateGroups.filter((group) => group.loading || group.templates.length > 0);

  return (
    <Block className="h-full overflow-auto">
      <Stack gap="density-2xl" padding="density-2xl">
        <PageHeader slotHeading={heading} slotDescription={headingDescription} />

        <Flex justify="center" className="w-full">
          <Stack gap="density-2xl" className={CONTENT_WIDTH}>
            {slotBanner}

            <Stack gap="density-md" role="group" aria-label={GROUP_LABEL}>
              {options.map((option) => (
                <StartTile
                  key={option.id}
                  label={option.title}
                  description={option.description}
                  icon={<option.icon size={16} aria-hidden />}
                  slotEnd={
                    option.tag ? (
                      <Badge kind={option.tag.kind} color={option.tag.color} size="medium">
                        {option.tag.label}
                      </Badge>
                    ) : undefined
                  }
                  onSelect={() => onSelect(option.id)}
                  busy={busyId === option.id}
                  busyLabel={busyLabel}
                  labelKind={TILE_LABEL_KIND}
                  descriptionKind={TILE_DESCRIPTION_KIND}
                  className={TILE_RADIUS}
                  disabled={disabled || !option.enabled}
                />
              ))}
            </Stack>

            {groups.length > 0 && (
              <>
                <Flex align="center" gap="density-lg">
                  <Divider className="flex-1" aria-hidden />
                  <Text kind="label/regular/sm" className="whitespace-nowrap text-secondary">
                    OR START FROM A TEMPLATE
                  </Text>
                  {templatesTag && (
                    <Badge
                      kind={templatesTag.kind}
                      color={templatesTag.color}
                      size="medium"
                      className="shrink-0"
                    >
                      {templatesTag.label}
                    </Badge>
                  )}
                  <Divider className="flex-1" aria-hidden />
                </Flex>

                {groups.map((group) => (
                  <Stack
                    key={group.id}
                    gap="density-md"
                    role="group"
                    aria-labelledby={`${group.id}-heading`}
                  >
                    <Text kind="label/bold/md" id={`${group.id}-heading`}>
                      {group.title}
                    </Text>
                    <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                      {group.loading
                        ? Array.from({ length: PLACEHOLDER_TILES }, (_, i) => (
                            <Skeleton
                              key={i}
                              className={`h-[61px] w-full ${TILE_RADIUS}`}
                              aria-label={`Loading ${group.title}`}
                            />
                          ))
                        : group.templates.map((template) => (
                            <StartTile
                              key={template.id}
                              label={template.name}
                              description={template.description}
                              icon={<template.icon size={16} color={group.accent} aria-hidden />}
                              onSelect={() => onSelect(template.id)}
                              busy={busyId === template.id}
                              busyLabel={busyLabel}
                              labelKind={TILE_LABEL_KIND}
                              descriptionKind={TILE_DESCRIPTION_KIND}
                              className={TILE_RADIUS}
                              disabled={disabled}
                            />
                          ))}
                    </div>
                  </Stack>
                ))}
              </>
            )}
          </Stack>
        </Flex>
      </Stack>
    </Block>
  );
};
