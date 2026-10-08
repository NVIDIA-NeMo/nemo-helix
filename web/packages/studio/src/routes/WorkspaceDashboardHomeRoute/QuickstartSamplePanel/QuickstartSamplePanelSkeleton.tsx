// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, Panel, Skeleton, Stack } from '@nvidia/foundations-react-core';
import {
  buildQuickstartSampleSteps,
  type QuickstartSampleAgent,
  type QuickstartSampleFeatures,
} from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSamplePanel/quickstartSampleContent';
import cn from 'classnames';
import type { FC } from 'react';

/** Only feeds the step count below; the hrefs and commands built from it are discarded. */
const PLACEHOLDER_AGENT: QuickstartSampleAgent = { name: '', description: '', status: '' };

/** Each flag defaults to the live value, as on `QuickstartSamplePanel`. */
export type QuickstartSamplePanelSkeletonProps = QuickstartSampleFeatures;

/**
 * Stands in for `QuickstartSamplePanel` while the sample agent and its deployments load, so
 * landing on a fresh sandbox does not open on an empty gap that the panel later pushes into.
 * Mirrors the panel's layout piece for piece; the Panel and step connectors are real chrome.
 */
export const QuickstartSamplePanelSkeleton: FC<QuickstartSamplePanelSkeletonProps> = ({
  intakeEnabled,
  agentOptimizationsEnabled,
}) => {
  // Steps are flag-gated, so count them the way the panel does rather than hardcoding four.
  const stepIds = buildQuickstartSampleSteps({
    workspace: '',
    agent: PLACEHOLDER_AGENT,
    intakeEnabled,
    agentOptimizationsEnabled,
  }).map((step) => step.id);

  return (
    <Panel aria-busy="true" data-testid="quickstart-sample-panel-skeleton">
      <Stack gap="density-2xl" className="w-full">
        <Stack gap="density-xl" className="w-full">
          {/* The agent row: icon chip, name and description, status badge. */}
          <Flex
            align="center"
            gap="density-lg"
            className="w-full rounded-lg border border-base bg-surface-raised p-density-lg"
          >
            <Skeleton className="size-10 shrink-0 rounded-md" />
            <Stack gap="density-xs" className="min-w-0 flex-1">
              <Skeleton className="h-4 w-48 max-w-full" />
              <Skeleton className="h-4 w-3/4" />
            </Stack>
            <Skeleton kind="pill" className="h-4 shrink-0" />
          </Flex>

          {/* The Studio / CLI switch, at its `small` height. */}
          <Skeleton className="h-8 w-full rounded-lg" />

          <Stack className="w-full">
            {stepIds.map((id, index) => {
              const isLast = index === stepIds.length - 1;
              return (
                <Flex
                  key={id}
                  gap="density-2xl"
                  className="w-full"
                  data-testid="quickstart-sample-step-skeleton"
                >
                  <Stack align="center" className="shrink-0 self-stretch">
                    <Skeleton kind="circle" />
                    {!isLast && <div className="w-0.5 flex-1 bg-accent-gray-subtle" />}
                  </Stack>
                  <Flex
                    gap="density-md"
                    wrap="wrap"
                    className={cn('@container min-w-0 flex-1', isLast ? 'pb-0' : 'pb-density-2xl')}
                  >
                    <Stack gap="density-xs" className="min-w-0 flex-1 basis-full @xl:basis-0">
                      <Skeleton className="h-5 w-40 max-w-full" />
                      <Skeleton className="h-5 w-2/3" />
                    </Stack>
                    <Skeleton kind="pill" className="h-7 w-24 shrink-0" />
                  </Flex>
                </Flex>
              );
            })}
          </Stack>
        </Stack>

        {/* The dashboard always wires the shared-workspace hand-off, so reserve its footer. */}
        <Divider />
        <Flex align="center" justify="center" wrap="wrap" gap="density-xl" className="w-full">
          <Skeleton className="h-5 w-56 max-w-full" />
          <Skeleton kind="pill" className="h-7 w-44" />
        </Flex>
      </Stack>
    </Panel>
  );
};
