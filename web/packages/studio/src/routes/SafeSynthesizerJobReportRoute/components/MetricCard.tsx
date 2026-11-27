// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ScoreGauge } from '@nemo/common/src/components/ScoreGauge';
import {
  AccordionContent,
  AccordionItem,
  AccordionRoot,
  AccordionTrigger,
  Flex,
  Stack,
  Text,
  useAccordionItemContext,
} from '@nvidia/foundations-react-core';
import { InfoTip } from '@studio/routes/SafeSynthesizerJobReportRoute/components/InfoTip';
import type { MetricCopy } from '@studio/routes/SafeSynthesizerJobReportRoute/reportCopy';
import type { MetricScore } from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';
import type { FC, ReactNode } from 'react';

/** Mounts details only while the card is open, so closed cards never build their charts. */
const OpenOnly: FC<{ children: ReactNode }> = ({ children }) => {
  const { open } = useAccordionItemContext();
  return open ? children : null;
};

interface MetricCardProps {
  id: string;
  copy: MetricCopy;
  score?: MetricScore | null;
  stat?: ReactNode;
  /** Card details; a card without them cannot be expanded. */
  children?: ReactNode;
}

export const MetricCard: FC<MetricCardProps> = ({ id, copy, score, stat, children }) => {
  const expandable = children !== undefined && children !== null && children !== false;
  return (
    <AccordionItem value={id} disabled={!expandable} data-testid={`metric-card-${id}`}>
      <AccordionTrigger
        chevronPosition="start"
        aria-label={`Toggle details for ${copy.name}`}
        className="[&>.nv-accordion-label-text]:min-w-0 [&>.nv-accordion-label-text]:flex-1"
      >
        <Flex justify="between" align="center" gap="density-lg" className="w-full">
          <Stack gap="density-xs" className="min-w-0">
            <Flex align="center" gap="density-sm">
              <Text kind="body/semibold/md">{copy.name}</Text>
              {copy.tooltip ? <InfoTip label={copy.name} text={copy.tooltip} /> : null}
            </Flex>
            <Text kind="body/regular/sm" className="text-secondary">
              {copy.description}
            </Text>
            {stat ? <Text kind="body/semibold/sm">{stat}</Text> : null}
          </Stack>
          {score !== undefined ? (
            <span className="shrink-0 [&_svg]:!size-full">
              <ScoreGauge score={score?.score ?? undefined} size="sm" />
            </span>
          ) : null}
        </Flex>
      </AccordionTrigger>
      {expandable ? (
        <AccordionContent>
          <OpenOnly>
            <Stack gap="density-lg" className="pt-density-md">
              {children}
            </Stack>
          </OpenOnly>
        </AccordionContent>
      ) : null}
    </AccordionItem>
  );
};

interface MetricGroupProps {
  title: string;
  description: string;
  icon: ReactNode;
  children: ReactNode;
}

/** A titled list of metric cards; each card expands on its own. */
export const MetricGroup: FC<MetricGroupProps> = ({ title, description, icon, children }) => (
  <Stack gap="density-md" data-testid="metric-group">
    <Flex align="start" gap="density-md">
      {icon}
      <Stack gap="density-xs">
        <Text kind="body/bold/lg">{title}</Text>
        <Text kind="body/regular/sm" className="text-secondary">
          {description}
        </Text>
      </Stack>
    </Flex>
    <AccordionRoot multiple>{children}</AccordionRoot>
  </Stack>
);
