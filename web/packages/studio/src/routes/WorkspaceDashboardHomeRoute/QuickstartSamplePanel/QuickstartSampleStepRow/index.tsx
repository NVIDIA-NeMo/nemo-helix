// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type {
  QuickstartSampleStep,
  QuickstartSampleView,
} from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSamplePanel/quickstartSampleContent';
import { StepActions } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSamplePanel/QuickstartSampleStepRow/StepActions';
import { StepCommands } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSamplePanel/QuickstartSampleStepRow/StepCommands';
import cn from 'classnames';
import type { FC } from 'react';

interface QuickstartSampleStepRowProps {
  step: QuickstartSampleStep;
  view: QuickstartSampleView;
  isLast: boolean;
}

/**
 * One entry in the quickstart timeline, following the KUI activity-feed pattern. Per the
 * design it uses glyph markers rather than severity dots, and carries no timestamp.
 */
export const QuickstartSampleStepRow: FC<QuickstartSampleStepRowProps> = ({
  step,
  view,
  isLast,
}) => {
  const Icon = step.icon;
  const isCli = view === 'cli';

  // Everything the two views differ by, in one place. Commands and actions stay separate
  // nodes rather than one swapped body because they occupy different slots below.
  const { title, description } = isCli ? step.cli : step.studio;
  const commands = isCli ? <StepCommands commands={step.cli.commands} title={title} /> : null;
  const actions = isCli ? null : <StepActions actions={step.studio.actions} />;

  return (
    <Flex gap="density-2xl" className="w-full">
      <Stack align="center" className="shrink-0 self-stretch" aria-hidden="true">
        <Flex
          align="center"
          justify="center"
          className="size-8 shrink-0 rounded-full bg-accent-gray-subtle"
        >
          <Icon className="size-4" />
        </Flex>
        {!isLast && (
          <div
            data-testid="quickstart-sample-step-connector"
            className="w-0.5 flex-1 bg-accent-gray-subtle"
          />
        )}
      </Stack>
      <Flex
        gap="density-md"
        wrap="wrap"
        className={cn('@container min-w-0 flex-1', isLast ? 'pb-0' : 'pb-density-2xl')}
      >
        <Stack gap="density-xs" className="min-w-0 flex-1 basis-full @xl:basis-0">
          <Text asChild kind="body/bold/md">
            <h3>{title}</h3>
          </Text>
          <Text kind="body/regular/md" className="text-secondary">
            {description}
          </Text>

          {commands}
        </Stack>

        {actions}
      </Flex>
    </Flex>
  );
};
