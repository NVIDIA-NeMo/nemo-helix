// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { RadioCard } from '@nemo/common/src/components/RadioCard';
import {
  Badge,
  Block,
  Divider,
  Flex,
  PageHeader,
  RadioGroupRoot,
  Skeleton,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import type { StartPageProps } from '@studio/components/StartOptions/types';
import type { FC } from 'react';

/** The column the whole flow sits in, per the design. */
const CONTENT_WIDTH = 'w-full max-w-[768px]';

/** The design's tile is `radius/md`; Card's own `radius-density-xl` is visibly rounder. */
const TILE_RADIUS = 'rounded-[var(--radius-md)]';

/** Placeholders shown for a group that is still loading — the design's rows are pairs. */
const PLACEHOLDER_TILES = 2;

/**
 * "How do you want to start?" — the entry point shared by the create flows.
 *
 * Options and templates are one radio group rather than two steps: a template is picked
 * outright, so there is no "start from a template" option that then reveals a list.
 */
export const StartPage: FC<StartPageProps> = ({
  heading,
  headingDescription,
  options,
  templateGroups = [],
  value,
  onChange,
  disabled = false,
  emptyHint,
  templatesTag,
  continueLabel = 'Continue',
  continueLoading = false,
  canContinue,
  onContinue,
  slotBanner,
}) => {
  const groups = templateGroups.filter((group) => group.loading || group.templates.length > 0);

  const selectedLabel =
    options.find((option) => option.id === value)?.title ??
    groups
      .flatMap((group) => (group.loading ? [] : group.templates))
      .find((template) => template.id === value)?.name;

  return (
    <Stack className="h-full">
      <Block className="flex-1 overflow-auto">
        <Stack gap="density-2xl" padding="density-2xl">
          <PageHeader slotHeading={heading} slotDescription={headingDescription} />

          <Flex justify="center" className="w-full">
            <RadioGroupRoot
              name="start-option"
              value={value ?? ''}
              onValueChange={onChange}
              disabled={disabled}
              className={CONTENT_WIDTH}
            >
              <Stack gap="density-2xl">
                <Stack gap="density-md">
                  {options.map((option) => (
                    <RadioCard
                      key={option.id}
                      value={option.id}
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
                      showIndicator={false}
                      className={TILE_RADIUS}
                      disabled={disabled || !option.enabled}
                    />
                  ))}
                </Stack>

                {groups.length > 0 && (
                  <>
                    <Flex align="center" gap="density-lg" aria-hidden>
                      <Divider className="flex-1" />
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
                      <Divider className="flex-1" />
                    </Flex>

                    {groups.map((group) => (
                      <Stack key={group.id} gap="density-md">
                        <Text kind="label/bold/md">{group.title}</Text>
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
                                <RadioCard
                                  key={template.id}
                                  value={template.id}
                                  label={template.name}
                                  description={template.description}
                                  icon={
                                    <template.icon size={16} color={group.accent} aria-hidden />
                                  }
                                  showIndicator={false}
                                  className={TILE_RADIUS}
                                  disabled={disabled}
                                />
                              ))}
                        </div>
                      </Stack>
                    ))}
                  </>
                )}

                {slotBanner}
              </Stack>
            </RadioGroupRoot>
          </Flex>
        </Stack>
      </Block>

      <Flex justify="center" className="shrink-0 border-t border-base bg-surface-base px-10 py-3">
        <Flex align="center" justify="end" gap="density-2xl" className={CONTENT_WIDTH}>
          <Text kind="label/regular/md" className="text-secondary">
            {selectedLabel
              ? `Continue with ${selectedLabel} selected`
              : (emptyHint ?? 'Select an option above to continue')}
          </Text>
          <LoadingButton
            color="brand"
            kind="primary"
            loading={continueLoading}
            onClick={onContinue}
            disabled={!canContinue}
          >
            {continueLabel}
          </LoadingButton>
        </Flex>
      </Flex>
    </Stack>
  );
};
