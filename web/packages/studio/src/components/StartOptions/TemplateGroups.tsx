// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RadioCard } from '@nemo/common/src/components/RadioCard';
import { RadioGroupRoot, Skeleton, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import {
  TILE_DESCRIPTION_KIND,
  TILE_LABEL_KIND,
  TILE_RADIUS,
} from '@studio/components/StartOptions/tile';
import type { TemplateGroupsProps } from '@studio/components/StartOptions/types';
import type { FC } from 'react';

/** Placeholders shown for a group that is still loading — the design's rows are pairs. */
const PLACEHOLDER_TILES = 2;

/**
 * The recipes behind the "start from a template" option, grouped by section. Picking one
 * acts on it; while it is being acted on, its card shows progress in place of its icon and
 * description.
 *
 * Its own radio group: the option cards above choose a way in, these choose which recipe,
 * and one group spanning both would make picking a recipe clear the option that revealed it.
 */
export const TemplateGroups: FC<TemplateGroupsProps> = ({
  groups,
  onSelect,
  pendingId = null,
  pendingLabel,
  disabled = false,
}) => {
  const shown = groups.filter((group) => group.loading || group.templates.length > 0);

  if (shown.length === 0) return null;

  return (
    <RadioGroupRoot
      name="start-template"
      aria-label="Templates"
      value={pendingId ?? ''}
      onValueChange={onSelect}
      disabled={disabled}
      className="w-full"
    >
      <Stack gap="density-2xl">
        {shown.map((group) => (
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
                    // The action is a sibling laid over the card, not inside it: the card
                    // is one click target, and may not hold another.
                    <div key={template.id} className="relative">
                      <RadioCard
                        value={template.id}
                        label={template.name}
                        description={
                          template.id === pendingId ? (
                            <span role="status">{pendingLabel}</span>
                          ) : (
                            template.description
                          )
                        }
                        icon={
                          template.id === pendingId ? (
                            <Spinner size="small" aria-label={pendingLabel ?? 'Setting up'} />
                          ) : (
                            <template.icon size={16} color={group.accent} aria-hidden />
                          )
                        }
                        compact
                        labelKind={TILE_LABEL_KIND}
                        descriptionKind={TILE_DESCRIPTION_KIND}
                        showIndicator={false}
                        className={`${TILE_RADIUS} ${template.action ? '[&_.nv-card-content]:pr-10!' : ''}`}
                        disabled={disabled}
                      />
                      {template.action ? (
                        <div className="absolute right-2 top-2">{template.action}</div>
                      ) : null}
                    </div>
                  ))}
            </div>
          </Stack>
        ))}
      </Stack>
    </RadioGroupRoot>
  );
};
