// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { QuickstartAgent } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartPanel/quickstartContent';
import { ChevronRight } from 'lucide-react';
import type { FC } from 'react';
import { Link } from 'react-router';

interface QuickstartAgentRowProps {
  agent: QuickstartAgent;
  href: string;
}

/**
 * The provisioned sample agent, as one full-row link: a single tab stop with a single
 * accessible name. Giving the chevron its own link would nest two links in the row.
 */
export const QuickstartAgentRow: FC<QuickstartAgentRowProps> = ({ agent, href }) => (
  <Link
    to={href}
    data-testid="quickstart-agent-row"
    // The borders are load-bearing: surface-base, -raised and -overlay are all the same
    // white in the light theme, so without them the row and chip vanish into the Panel.
    className="flex w-full items-center gap-density-lg rounded-lg border border-base bg-surface-raised p-density-lg no-underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
  >
    <Flex
      align="center"
      justify="center"
      className="shrink-0 rounded-md border border-base bg-surface-overlay p-density-md"
    >
      <ENTITY_ICONS.agents className="size-6" aria-hidden="true" />
    </Flex>

    <Stack className="min-w-0 flex-1">
      <Text kind="label/semibold/md">{agent.name}</Text>
      {/* `Text` renders a span, so `truncate` needs `block`; `title` keeps the full
          description reachable once it is clipped. */}
      <Text
        kind="body/regular/md"
        data-testid="quickstart-agent-description"
        className="block truncate text-secondary"
        title={agent.description}
      >
        {agent.description}
      </Text>
    </Stack>

    <StatusBadge status={agent.status} />
    <ChevronRight className="size-4 shrink-0 text-secondary" aria-hidden="true" />
  </Link>
);
