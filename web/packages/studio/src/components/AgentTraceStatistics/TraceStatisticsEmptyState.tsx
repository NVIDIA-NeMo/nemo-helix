// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EntityAgentBridge } from '@nemo/common/src/components/EntityAgentBridge';
import { ChartPerformance, Play } from '@nemo/icons/icons';
import { Button, Card, Flex, Stack, StatusMessage } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface Props {
  /** Resolves `<workspace>` in the bridge's CLI command and skill prompt. */
  workspace?: string;
  /** Invoke the agent so it emits its first traces. */
  onRunAgent?: () => void;
  /** Widen the window — offered only when a longer one exists. */
  onExpandRange?: () => void;
}

/**
 * Deliberately not an `EntityEmptyState`. This panel escalates through three
 * distinct exits — run the agent, hand tracing setup to a coding agent, widen
 * the window — which overruns the empty-state footer's two-action rule, and its
 * "widen the range" exit is not a `no-results` clear-filters action either: the
 * panel never learns whether traces exist outside the window, only that none
 * landed inside it. Offering `onRunAgent` on both branches is the point.
 *
 * The registry still owns the handoff copy: {@link EntityAgentBridge} is the
 * same CLI/skill-prompt snippet every `telemetryTraces` empty state renders.
 */
export const TraceStatisticsEmptyState: FC<Props> = ({ workspace, onRunAgent, onExpandRange }) => (
  <Card>
    <Flex justify="center" padding="density-2xl">
      <Stack gap="density-lg" align="center" className="w-full max-w-[40rem]">
        <StatusMessage
          slotMedia={<ChartPerformance className="size-12 text-placeholder" />}
          slotHeading="No traces"
          slotSubheading={
            onExpandRange
              ? 'No traces in this date range.'
              : `Cost, token, and latency averages are built from instrumented agent runs — nothing has reported. Send the agent's traces to Intake, then invoke it to start filling this in.`
          }
          slotFooter={
            <Flex gap="density-sm" justify="center" wrap="wrap">
              {onRunAgent ? (
                <Button onClick={onRunAgent}>
                  <Play size="16" aria-hidden />
                  Run the agent
                </Button>
              ) : null}
              {onExpandRange ? (
                <Button kind="tertiary" onClick={onExpandRange}>
                  Look back further
                </Button>
              ) : null}
            </Flex>
          }
        />
        <EntityAgentBridge entity="telemetryTraces" workspace={workspace} className="w-full" />
      </Stack>
    </Flex>
  </Card>
);
