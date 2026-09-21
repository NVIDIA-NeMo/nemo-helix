// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { RadioCard } from '@nemo/common/src/components/RadioCard';
import {
  Block,
  Divider,
  Flex,
  PageHeader,
  RadioGroupRoot,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import type { StartPageProps } from '@studio/components/StartOptions/types';
import { ArrowRight } from 'lucide-react';
import type { FC } from 'react';

/** The column the whole flow sits in, per the design. */
const CONTENT_WIDTH = 'w-full max-w-[768px]';

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
  blockedHint,
  continueLabel = 'Continue',
  continueLoading = false,
  canContinue,
  onContinue,
  slotBanner,
}) => {
  const groups = templateGroups.filter((group) => group.templates.length > 0);

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
                      showIndicator={false}
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
                      <Divider className="flex-1" />
                    </Flex>

                    {groups.map((group) => (
                      <Stack key={group.id} gap="density-md">
                        <Text kind="label/bold/md">{group.title}</Text>
                        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                          {group.templates.map((template) => (
                            <RadioCard
                              key={template.id}
                              value={template.id}
                              label={template.name}
                              description={template.description}
                              icon={<template.icon size={16} aria-hidden />}
                              showIndicator={false}
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

      <Flex justify="center" className="shrink-0 border-t border-base bg-surface-base px-10 py-4">
        <Flex align="center" justify="end" gap="density-lg" className={CONTENT_WIDTH}>
          {!canContinue && !continueLoading && (
            <Text kind="body/regular/sm" className="text-secondary">
              {blockedHint ?? 'Select an option above to continue'}
            </Text>
          )}
          <LoadingButton
            color="brand"
            kind="primary"
            loading={continueLoading}
            onClick={onContinue}
            disabled={!canContinue}
          >
            {continueLoading ? (
              continueLabel
            ) : (
              <>
                {continueLabel}
                <ArrowRight size={16} aria-hidden />
              </>
            )}
          </LoadingButton>
        </Flex>
      </Flex>
    </Stack>
  );
};
