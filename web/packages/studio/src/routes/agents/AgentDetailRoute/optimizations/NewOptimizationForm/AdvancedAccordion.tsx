// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Accordion, Flex, Text } from '@nvidia/foundations-react-core';
import { SearchSpaceFields } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/SearchSpaceFields';
import {
  formatRange,
  type SearchParameter,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizationCatalog';
import { type FC } from 'react';

/** Collapsed summary line: enough to see the search space without opening the panel. */
const summarize = (parameters: SearchParameter[]): string =>
  [
    ...parameters.map((parameter) => `${parameter.label} ${formatRange(parameter)}`),
    `${parameters.length} parameter${parameters.length === 1 ? '' : 's'}`,
  ].join(' · ');

export interface AdvancedAccordionProps {
  searchSpace: SearchParameter[];
}

/**
 * The escape hatch from the guided form: the search space the intent chose, opened up for editing.
 *
 * Collapsed by default because the intent already answered this for most users. A preview of the
 * generated config belongs beside it, and lands with the config builder.
 */
export const AdvancedAccordion: FC<AdvancedAccordionProps> = ({ searchSpace }) => (
  <Accordion
    items={[
      {
        value: 'search-space',
        chevronPosition: 'start',
        slotTrigger: (
          <Flex align="center" gap="density-md" wrap="wrap">
            <Text kind="body/semibold/sm">Advanced — edit the search space</Text>
            <Text kind="body/regular/xs" color="secondary">
              {summarize(searchSpace)}
            </Text>
          </Flex>
        ),
        slotContent: <SearchSpaceFields />,
      },
    ]}
  />
);
