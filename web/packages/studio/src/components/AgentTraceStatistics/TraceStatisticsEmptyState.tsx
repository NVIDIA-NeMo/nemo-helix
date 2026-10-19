// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Card, Flex, StatusMessage } from '@nvidia/foundations-react-core';
import { ChartNoAxesCombined, Play } from 'lucide-react';
import { type FC } from 'react';

interface Props {
  /** Invoke the agent so it emits its first traces. */
  onRunAgent?: () => void;
  /** Widen the window — offered only when a longer one exists. */
  onExpandRange?: () => void;
  /** Docs on turning on tracing for an agent. */
  onLearnMore?: () => void;
}

export const TraceStatisticsEmptyState: FC<Props> = ({
  onRunAgent,
  onExpandRange,
  onLearnMore,
}) => (
  <Card>
    <Flex justify="center" padding="density-2xl">
      <StatusMessage
        slotMedia={<ChartNoAxesCombined className="size-12 text-placeholder" />}
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
                <Play size={16} aria-hidden />
                Run the agent
              </Button>
            ) : null}
            {onLearnMore ? (
              <Button kind="secondary" onClick={onLearnMore}>
                Set up tracing
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
    </Flex>
  </Card>
);
